import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
import pandas as pd
import numpy as np


from catboost import CatBoostRegressor
from nptyping import NDArray, Float, Shape
from xgboost import XGBRegressor
import lightgbm as lgb

# tensorflow give GPU warning that process can be speede up by Nvidia GPU with TensorRT
# remove this warning by os.environ


from src.data_preprocessing import normalize_data_for_ANN, N_ROWS
from sklearn.metrics import mean_absolute_percentage_error as mape
from src.data_preprocessing import (
    all_filtered_experimental_data_not_normalized,
    split_data_into_training_and_testing,
    convert_current_to_log,
)
from tensorflow import keras
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def return_test_data() -> tuple[NDArray[Shape["N_ROWS, 2"], Float], NDArray[Shape["1, N_ROWS"], Float]]:
    """
    Returs testing data X_test (potential) with shape (N,1) and y_test log10((abs(current density)))
    with shape (N,) (1 column, N_ROWS rows) since models only accept (N,) shaped arrays when predicting
    """

    # all_data is np.ndarray type (N,3) with filtered data (negative values are included)
    all_data = all_filtered_experimental_data_not_normalized()
    # return testing data [1], where training is [0]
    testing_data = split_data_into_training_and_testing(all_data)[1]

    X_test, y_test = testing_data[:, :2].reshape(-1, 2), convert_current_to_log(testing_data[:, -1])
    # ph_mask = X_test[:, 1] == ph
    # X_test, y_test = X_test[ph_mask], y_test[ph_mask]
    return (X_test, y_test)


def filter_x_y_boolean_mask(ph) -> list[np.ndarray]:
    X_test, y_test = return_test_data()
    ph_mask = X_test[:, 1] == ph
    X_test, y_test = X_test[ph_mask], y_test[ph_mask]
    return [X_test, y_test]


def plot_scatter_if_early_stopping(model: str, iterations: int, train_loss_last_iter, val_loss_last_iter) -> None:
    df = pd.read_csv("model_figures/max_iterations_GBDTs.csv", sep="\t")
    max_iterations = df.loc[df["model"] == model, "max_iterations"].values[0]
    if iterations < max_iterations:
        ax_loss_trees.scatter(iterations, train_loss_last_iter, label="Early stopping triggered")
        ax_val_loss_trees.scatter(iterations, val_loss_last_iter, label="Early stopping triggered")


def linestyles_and_markers_for_model_comparisons(model_str: str) -> list[str]:
    """
    Function to create unique linestyle and marker for each model
    """
    models = ["cb", "rf", "xgb", "lgb", "ann"]
    linestyles = ["-", "--", ":", "--", ":"]
    colors = ["#777777", "b", "b", "orange", "orange"]
    try:
        idx = models.index(model_str)
        return [linestyles[idx], colors[idx]]
    except ValueError:
        raise ValueError


if __name__ == "__main__":
    # test ML algorithms on unseen data
    fig_pred = plt.figure(figsize=(10, 10))
    ax_pred = fig_pred.subplots()

    # figure for comparison of best and second best ann tuned model
    fig_compare_anns = plt.figure(figsize=(10, 10))
    ax_compare_anns = fig_compare_anns.subplots()

    # training loss GBDTs
    fig_loss_trees = plt.figure(figsize=(10, 10))
    ax_loss_trees = fig_loss_trees.subplots()
    ax_loss_trees.set_xlabel("Iterations")
    ax_loss_trees.set_ylabel("Training loss, RMSE")

    # validation loss GBDTs
    fig_val_loss_trees = plt.figure(figsize=(10, 10))
    ax_val_loss_trees = fig_val_loss_trees.subplots()
    ax_val_loss_trees.set_xlabel("Iterations")
    ax_val_loss_trees.set_ylabel("Validation loss, RMSE")

    # ANN losses
    fig_loss_ANN = plt.figure(figsize=(10, 10))
    ax_loss_ANN = fig_loss_ANN.subplots()
    ax_loss_ANN.set_xlabel("Epochs")
    ax_loss_ANN.set_ylabel("Error, RMSE")


def plot_experimental_testing_data(ph) -> None:
    """
    Plots the experimental filtered curves for all pHs in the test data set
    """

    # potential, abs(current density)
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)
    ax_pred.semilogx(10**y_test_ph, X_test_ph[:, 0], label=f"Exp data, pH = {ph}", color="k", linestyle="--")
    ax_compare_anns.semilogx(10**y_test_ph, X_test_ph[:, 0], label=f"Exp data, pH = {ph}", color="k", linestyle="--")


def random_forest_comparison(ph) -> None:
    """
    The RF data is trained and stored from train_models.py
    """
    X_test_ph, _ = filter_x_y_boolean_mask(ph)
    path = "models_data/random_forest_output"
    # load current density prediction
    # current_density_pred = np.loadtxt(f"{path}/current_density_pred.csv", skiprows=1, usecols=2)
    current_density_pred = pd.read_csv(f"{path}/current_density_pred_ph_{ph}.csv", sep="\t")["Current density [A/cm2]"]

    linestyle, color = linestyles_and_markers_for_model_comparisons("rf")
    ax_pred.semilogx(
        current_density_pred,
        X_test_ph[:, 0],
        label="Random Forest",
        linestyle=linestyle,
        color=color,
    )


def catboost_comparison(ph, store_mape: list) -> None:
    cb = CatBoostRegressor()
    cb.load_model("models_saved/catboost_model.cbm")
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)
    y_pred_ph = cb.predict(X_test_ph)
    linestyle, color = linestyles_and_markers_for_model_comparisons("cb")
    ax_pred.semilogx(10**y_pred_ph, X_test_ph[:, 0], label="CatBoost", linestyle=linestyle, color=color)

    store_mape.append(mape(y_pred_ph, y_test_ph))


def plot_train_val_loss_catboost():
    df_train_loss = pd.read_csv("catboost_info/learn_error.tsv", sep="\t")
    ax_loss_trees.plot(df_train_loss["iter"], df_train_loss["RMSE"], label="CatBoost")
    df_val_loss = pd.read_csv("catboost_info/test_error.tsv", sep="\t")
    ax_val_loss_trees.plot(df_val_loss["iter"], df_val_loss["RMSE"], label="Catboost", linestyle="--")

    plot_scatter_if_early_stopping(
        "cb", df_train_loss["iter"].iloc[-1], df_train_loss["RMSE"].iloc[-1], df_val_loss["RMSE"].iloc[-1]
    )


def xgboost_comparison(ph, store_mape) -> None:
    xgb = XGBRegressor()
    xgb.load_model("models_saved/xgboost.txt")
    X_test, y_test = filter_x_y_boolean_mask(ph)
    # prediction of log10(abs(current density))
    y_pred = xgb.predict(X_test)
    store_mape.append(mape(y_pred, y_test) * 100)
    y_pred = 10**y_pred
    linestyle, color = linestyles_and_markers_for_model_comparisons("xgb")
    ax_pred.semilogx(y_pred, X_test[:, 0], label="XGBoost", linestyle=linestyle, color=color)


def plot_train_val_loss_xgboost():
    # plot rmse for each iter
    df = pd.read_csv("models_data/xgboost_info/train_val_loss.csv", sep="\t")
    ax_loss_trees.plot(df["iter"], df["train_loss_rmse"], label="XGBoost")
    ax_val_loss_trees.plot(df["iter"], df["val_loss_rmse"], label="XGBoost", linestyle="--")

    plot_scatter_if_early_stopping(
        "xgb", df["iter"].iloc[-1], df["train_loss_rmse"].iloc[-1], df["val_loss_rmse"].iloc[-1]
    )


def lgbm_comparison(ph, store_mape) -> None:
    lgbm = lgb.Booster(model_file="models_saved/lgbm.txt")
    X_test, y_test = filter_x_y_boolean_mask(ph)
    # prediction of log10(abs(current density))
    y_pred = lgbm.predict(X_test)
    store_mape.append(mape(y_pred, y_test) * 100)
    y_pred = 10**y_pred
    linestyle, color = linestyles_and_markers_for_model_comparisons("lgb")
    ax_pred.semilogx(y_pred, X_test[:, 0], label="LightGBM", linestyle=linestyle, color=color)


def plot_train_val_loss_lgbm():
    # plot rmse for each iter
    # plot rmse for each iter
    df = pd.read_csv("models_data/lgbm_info/train_val_loss.csv", sep="\t")
    ax_loss_trees.plot(df["iter"], df["train_loss_rmse"], label="LightGBM")
    ax_val_loss_trees.plot(df["iter"], df["val_loss_rmse"], label="LightGBM", linestyle="--")

    plot_scatter_if_early_stopping(
        "lgbm", df["iter"].iloc[-1], df["train_loss_rmse"].iloc[-1], df["val_loss_rmse"].iloc[-1]
    )


def ANN_comparison(ph, store_mape) -> None:
    _, _, _, _, x_scaler, y_scaler = normalize_data_for_ANN()
    X_test, y_test = filter_x_y_boolean_mask(ph)

    # plot the two best models to compare hyperparameters
    files: list[str] = os.listdir("tuning_results")
    for file in files:
        if file.endswith(".h5"):
            model = keras.models.load_model(f"tuning_results/{file}")  # type: ignore
            y_pred_log = y_scaler.inverse_transform(model.predict(x_scaler.fit_transform(X_test)))
            y_pred = 10**y_pred_log
            which_model: str = file.split("_")[0]

            # store error only if best model
            if file == "first_best_model.h5":
                linestyle, color = linestyles_and_markers_for_model_comparisons("ann")
                ax_pred.semilogx(y_pred, X_test[:, 0], label=f"ANN {which_model}", linestyle=linestyle, color=color)
                ax_compare_anns.semilogx(y_pred, X_test[:, 0], label=f"ANN {which_model}")
                store_mape.append(mape(y_pred_log, y_test) * 100)
            else:
                ax_compare_anns.semilogx(y_pred, X_test[:, 0], label=f"ANN {which_model}")


def plot_train_val_loss_ann_best_model():
    # plot loss from training for best model
    df_loss = pd.read_csv("models_data/ANN_info/training_val_loss0", sep="\t")
    epochs = [iter for iter in range(1, len(df_loss["val_rmse"]) + 1, 1)]
    # plot epochs vs rmse (root of mse which is the loss given in df) for best model
    ax_loss_ANN.semilogy(epochs, df_loss["rmse"], "s-", label="ANN training loss best model", color="r")
    ax_loss_ANN.semilogy(epochs, df_loss["val_rmse"], "s--", label="ANN validation loss best model", color="r")


def plot_histogram_mape_models(best_scores_mape_log: pd.DataFrame):
    fig, ax = plt.subplots()

    bar_width = 0.07
    labels = ["RF", "CB", "XGB", "LGB", "ANN"]
    colors = ["k", "#555555", "#777777", "#999999", "#CCCCCC"]  # grey scales
    hatches = [None, "/", None, "/", None]
    locs = [1, 2, 3, 4]
    # Loop over each row
    for i, row in best_scores_mape_log.iterrows():
        # Get the values for the current row and remove the pH value
        values = row.drop("pH").values
        sorting_indices = np.argsort(values)  # type: ignore
        # all rows are sorted in ascending order, rearange all used lists
        # sorted_colors = [colors[i] for i in sorting_indices]
        sorted_values = [values[i] for i in sorting_indices]
        sorted_labels = [labels[i] for i in sorting_indices]

        if i == 0:
            # Plot the bars
            for idx, val in enumerate(sorted_values):
                ax.bar(
                    locs[i] + idx * 2 * bar_width,
                    val,
                    bar_width,
                    label=sorted_labels[idx],
                    color=colors[idx],
                    hatch=hatches[idx],
                )
        else:
            # Plot the bars
            for idx, val in enumerate(sorted_values):
                ax.bar(locs[i] + idx * 2 * bar_width, val, bar_width, color=colors[idx], hatch=hatches[idx])

    ax.set_xlim(locs[0] / 2, locs[-1] + 1)

    # Add legend and axis labels
    ax.legend()
    ax.set_ylabel("Mean Absolute Percentage Error of log10(|i|) [%]")
    ax.set_ylim(0, 15)
    ax.set_xticks(
        [(i + 2 * 2 * bar_width) for i in locs], labels=[f"pH = {ph}" for ph in best_scores_mape_log["pH"]], rotation=45
    )
    fig.tight_layout()
    fig.savefig("model_figures/mape_log_%_histogram.png")


if __name__ == "__main__":
    # prediction loss of log(abs(current density)) in Mean Absolute Percentage Errror
    best_scores_mape_log = pd.DataFrame()
    best_scores_mape_log["pH"] = pd.read_csv("testing_pHs.csv", sep="\t")["test_pHs"]
    # store mape_log for RF
    best_scores_mape_log["rf"] = (
        pd.read_csv("models_data/random_forest_output/errors.csv", sep="\t")["(MAPE of log)"] * 100
    )
    plot_train_val_loss_catboost()
    plot_train_val_loss_xgboost()
    plot_train_val_loss_lgbm()
    plot_train_val_loss_ann_best_model()

    try:
        # plot losses
        fig_loss_trees.legend(loc="upper right")
        fig_loss_trees.savefig("model_figures/training_loss_GBDTS.png")

        fig_val_loss_trees.legend(loc="upper right")
        fig_val_loss_trees.savefig("model_figures/validation_loss_GBDTS.png")

        fig_loss_ANN.legend(loc="upper right")
        fig_loss_ANN.savefig("model_figures/train_val_loss_ANN.png")
    except Exception as e:
        raise e

    df = pd.read_csv("testing_pHs.csv", sep="\t")
    # compare results for all pHs
    store_mape_catboost = []
    store_mape_xgboost = []
    store_mape_lgbm = []
    store_mape_ann = []
    # do preds for each pH in the test data set
    for ph in df["test_pHs"]:
        plot_experimental_testing_data(ph)
        catboost_comparison(ph, store_mape_catboost)
        random_forest_comparison(ph)
        xgboost_comparison(ph, store_mape_xgboost)
        lgbm_comparison(ph, store_mape_lgbm)
        ANN_comparison(ph, store_mape_ann)

        # save figs
        try:
            ax_pred.set_xlabel("|i| [A/cm$^2$]")
            ax_pred.set_ylabel("E [V]")
            ax_pred.legend(loc="upper right")
            fig_pred.savefig(f"model_figures/comparison_with_exp_results/comparison_with_exp_data_ph_{ph}.png")
            ax_pred.clear()

            ax_compare_anns.set_xlabel("|i| [A/cm$^2$]")
            ax_compare_anns.set_ylabel("E [V]")
            ax_compare_anns.legend(loc="upper right")
            fig_compare_anns.savefig(f"model_figures/comparison_of_anns_with_exp_ph_{ph}.png")
            ax_compare_anns.clear()

        except Exception as e:
            raise e

    best_scores_mape_log["cb"] = store_mape_catboost
    best_scores_mape_log["xgb"] = store_mape_xgboost
    best_scores_mape_log["lgb"] = store_mape_lgbm
    best_scores_mape_log["ann"] = store_mape_ann
    best_scores_mape_log = best_scores_mape_log.sort_values(by="pH").reset_index(drop=True)
    best_scores_mape_log.to_csv("model_figures/mape_of_pred_log%.csv", sep="\t", index=False)
    plot_histogram_mape_models(best_scores_mape_log)
