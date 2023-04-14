import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
import numpy as np
import pandas as pd
from src.data_preprocessing import (
    all_filtered_experimental_data_not_normalized,
    split_data_into_training_and_testing,
    ph_for_testing,
    convert_current_to_log,
)

from catboost import CatBoostRegressor
from nptyping import NDArray, Float, Shape
from xgboost import XGBRegressor
import lightgbm as lgb

# tensorflow give GPU warning that process can be speede up by Nvidia GPU with TensorRT
# remove this warning by os.environ


from src.data_preprocessing import normalize_data_for_ANN, N_ROWS
from sklearn.metrics import mean_absolute_percentage_error as mape
from tensorflow import keras
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# func OK
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
    return (X_test, y_test)


if __name__ == "__main__":
    fig_pred = plt.figure(figsize=(10, 10))
    ax_pred = fig_pred.subplots()
    ax_pred.set_xlabel("|i| [A/cm$^2$]")
    ax_pred.set_ylabel("E [V]")

    fig_loss_trees = plt.figure(figsize=(10, 10))
    ax_loss_trees = fig_loss_trees.subplots()
    ax_loss_trees.set_xlabel("Iterations")
    ax_loss_trees.set_ylabel("Loss, RMSE")

    fig_loss_ANN = plt.figure(figsize=(10, 10))
    ax_loss_ANN = fig_loss_ANN.subplots()
    ax_loss_ANN.set_xlabel("Epochs")
    ax_loss_ANN.set_ylabel("Error, MSE")

# prediction loss of log(abs(current density)) in Mean Absolute Percentage Errror
best_scores_mape = {}


def plot_experimental_testing_data() -> None:
    # potential, abs(current density)
    X_test, y_test = return_test_data()
    # slice pH column for plotting, also [:,0] reshapes the matrix to (N,) from (N,2)
    X_test = X_test[:, 0]
    ax_pred.semilogx(10**y_test, X_test, label=f"Exp data, pH = {ph_for_testing()}", color="k")


def catboost_comparison() -> None:
    cb = CatBoostRegressor()
    cb.load_model("models_saved/catboost_model.cbm")
    X_test, y_test = return_test_data()
    # prediction of log10(abs(current density))
    y_pred = cb.predict(X_test)
    # reshape to (N,1) (N rows, 1 column), since training data is shaped (N,2)
    # also convert to 10**current as this is what we evaluate against in the log plot
    best_scores_mape["cb"] = mape(y_pred, y_test)
    y_pred = 10**y_pred
    ax_pred.semilogx(y_pred, X_test[:, 0], label="CatBoost")

    # loss(iterations)
    df_loss = np.loadtxt("catboost_info/learn_error.tsv", skiprows=1)
    ax_loss_trees.plot(df_loss[:, 0], df_loss[:, 1], label="CatBoost")


def random_forest_comparison() -> None:
    X_test, y_test = return_test_data()
    path = "models_data/random_forest_output"
    # load current density prediction
    # current_density_pred = np.loadtxt(f"{path}/current_density_pred.csv", skiprows=1, usecols=2)
    current_density_pred = pd.read_csv(f"{path}/current_density_pred.csv", sep="\t")["Current density [A/cm2]"]

    ax_pred.semilogx(current_density_pred, X_test[:, 0], label="Random Forest")
    # load MAPE (scalar)
    best_scores_mape["rf"] = pd.read_csv(f"{path}/mape.csv", sep="\t")["(MAPE of log)"][0]

    # load rmse for all estimators (trees)
    df_loss = pd.read_csv(f"{path}/rmse.csv", sep="\t")
    ax_loss_trees.plot(df_loss["Trees"], df_loss["rmse of log"], label="RF")


def xgboost_comparison() -> None:
    xgb = XGBRegressor()
    xgb.load_model("models_saved/xgboost.txt")
    X_test, y_test = return_test_data()
    # prediction of log10(abs(current density))
    y_pred = xgb.predict(X_test)
    best_scores_mape["xgb"] = mape(y_pred, y_test)
    y_pred = 10**y_pred
    ax_pred.semilogx(y_pred, X_test[:, 0], label="XGBoost")

    # plot rmse for each iter
    df = pd.read_csv("models_data/xgboost_info/training_loss.csv", sep="\t")

    ax_loss_trees.plot(df["iterations"], df["rmse"], label="xgboost")


def lgbm_comparison() -> None:
    lgbm = lgb.Booster(model_file="models_saved/lgbm.txt")
    X_test, y_test = return_test_data()
    # prediction of log10(abs(current density))
    y_pred = lgbm.predict(X_test)
    best_scores_mape["lgbm"] = mape(y_pred, y_test)
    y_pred = 10**y_pred
    ax_pred.semilogx(y_pred, X_test[:, 0], label="lightgbm")

    # plot rmse for each iter
    df = pd.read_csv("models_data/lgbm_info/training_loss.csv", sep="\t")

    ax_loss_trees.plot(df["iterations"], df["rmse"], label="lightgbm")


def ANN_comparison() -> None:
    _, _, _, _, x_scaler, y_scaler = normalize_data_for_ANN()
    X_test, y_test = return_test_data()
    model = keras.models.load_model("models_saved/ANN")

    y_pred = y_scaler.inverse_transform(model.predict(x_scaler.fit_transform(X_test)))
    best_scores_mape["ANN"] = mape(y_pred, y_test)

    y_pred = 10**y_pred

    ax_pred.semilogx(y_pred, X_test[:, 0], label="ANN")

    # plot loss
    df_loss = pd.read_csv("models_data/ANN_info/training_val_loss", sep="\t")
    epochs = [iter for iter in range(1, len(df_loss["val_loss"]) + 1, 1)]
    # plot epochs vs rmse (root of mse which is the loss given in df)
    ax_loss_ANN.plot(epochs, df_loss["loss"], label="ANN training loss")
    ax_loss_ANN.plot(epochs, df_loss["val_loss"], label="ANN validation loss")


def store_mape_from_models_into_csv() -> None:
    df = pd.DataFrame(list(best_scores_mape.items()), columns=["Model", "MAPE of log"]).sort_values("MAPE of log")
    df["MAPE of current density"] = 10 ** df["MAPE of log"]
    df.to_csv("model_figures/df_with_models_MAPE_errors", sep="\t", index=False)


if __name__ == "__main__":
    plot_experimental_testing_data()
    catboost_comparison()
    random_forest_comparison()
    xgboost_comparison()
    lgbm_comparison()
    # ANN_comparison()
    store_mape_from_models_into_csv()

    # save figs
    fig_pred.legend(loc="upper right")
    fig_pred.savefig("model_figures/comparison_of_models.png")
    fig_loss_trees.legend(loc="upper right")
    fig_loss_trees.savefig("model_figures/iteration_loss.png")
    fig_loss_ANN.legend(loc="upper right")
    fig_loss_ANN.savefig("model_figures/train_val_loss_ANN.png")