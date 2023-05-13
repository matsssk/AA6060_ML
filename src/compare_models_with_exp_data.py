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
from src.get_selected_features import plot_confidence_interval
from typing import Any
from src.get_selected_features import linreg_tafel_line_ORR_or_HER, get_ocps_machine_learning_models
from tensorflow import keras
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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

    # Appendix: evaluate each model with exp data. 2x2 plot
    fig_rf, ax_individual_model_vs_exp_rf = plt.subplots(2, 2, figsize=(15, 15))
    fig_rf.supxlabel("|i| [A/cm$^2$]")
    fig_rf.supylabel("E [V]")

    fig_cb, ax_individual_model_vs_exp_cb = plt.subplots(2, 2, figsize=(15, 15))
    fig_cb.supxlabel("|i| [A/cm$^2$]")
    fig_cb.supylabel("E [V]")

    fig_lgb, ax_individual_model_vs_exp_lgb = plt.subplots(2, 2, figsize=(15, 15))
    fig_lgb.supxlabel("|i| [A/cm$^2$]")
    fig_lgb.supylabel("E [V]")

    fig_ann, ax_individual_model_vs_exp_ann = plt.subplots(2, 2, figsize=(15, 15))
    fig_ann.supxlabel("|i| [A/cm$^2$]")
    fig_ann.supylabel("E [V]")

    fig_xgb, ax_individual_model_vs_exp_xgb = plt.subplots(2, 2, figsize=(15, 15))
    fig_xgb.supxlabel("|i| [A/cm$^2$]")
    fig_xgb.supylabel("E [V]")


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


def filter_x_y_boolean_mask(ph) -> list[np.ndarray]:
    X_test, y_test = return_test_data()
    ph_mask = X_test[:, 1] == ph
    X_test, y_test = X_test[ph_mask], y_test[ph_mask]
    return [X_test, y_test]


def plot_scatter_if_early_stopping(model: str, iterations: int, train_loss_last_iter, val_loss_last_iter) -> None:
    df = pd.read_csv("summarized_data_figures_datafiles/csv_files/max_iterations_GBDTs.csv", sep="\t")
    max_iterations = df.loc[df["model"] == model, "max_iterations"].values[0]
    if iterations < max_iterations:
        ax_loss_trees.scatter(iterations, train_loss_last_iter, label="Early stopping", color="k", marker="x")
        ax_val_loss_trees.scatter(iterations, val_loss_last_iter, label="Early stopping", color="k", marker="x")


def linestyles_and_markers_for_model_comparisons(model_str: str) -> list[str]:
    """
    Function to create unique linestyle and marker for each model
    """
    models = ["cb", "rf", "xgb", "lgb", "ann"]
    linestyles = ["-", "-", "-", "-", "-"]
    colors = ["r", "g", "b", "m", "brown"]
    try:
        idx = models.index(model_str)
        return [linestyles[idx], colors[idx]]
    except ValueError:
        raise ValueError


def tafel_math_expression():
    return r"$\frac{\partial E}{\partial log|i|}$"


def round_r_value(r_value: float) -> float:
    return round(r_value**2, 5)


def corrosion_potential_and_current_density(slope, i0, intercept) -> list[float]:
    E_corr, i_corr = slope * i0 + intercept, 10**i0
    return [E_corr, i_corr]


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def plot_tafel_lines_E_corr_i_corr(
    ax,
    i_applied_log_abs,
    E_applied,
    slope,
    intercept,
    ocp,
    r_value,
    E_corr,
    i_corr,
    color,
    model: str,
    slope_std_error: float,
    intercept_std_error: float,
) -> None:
    # plot tafel line
    x, y = 10**i_applied_log_abs, slope * i_applied_log_abs + intercept
    ax.semilogx(
        x,
        y,
        label=f"{tafel_math_expression()} = {int(slope*1000)} mV/dec, R\u00b2 = {round_r_value(r_value)}",
        color=color,
        linestyle="--",
    )
    # add a zoomed plot of the linreg
    zoom_xlim = [1e-6, 1e-5]
    zoom_ylim = [y.min(), y.max()]

    # Create the zoomed-in plot
    ax_zoom = zoomed_inset_axes(
        ax,
        zoom=10,
        bbox_to_anchor=(0.5, 0.7),
        loc="upper left",
        axes_kwargs={"facecolor": "lightgray"},
    )
    # Set the limits and ticks for the zoomed-in plot
    ax_zoom.set_xlim(zoom_xlim)
    ax_zoom.set_ylim(zoom_ylim)
    ax_zoom.set_xticks([1e-6, 1e-5])
    ax_zoom.set_yticks([round(y.min(), 1), round(y.max(), 1)])

    ax_zoom.semilogx(x, y, color=color)
    plot_confidence_interval(
        confidence_level=0.95,
        i_applied_log_abs=i_applied_log_abs,
        slope=slope,
        slope_std_error=slope_std_error,
        intercept=intercept,
        intercept_std_error=intercept_std_error,
        ax=ax,
        color=color,
        model=model,
    )
    # Set the limits and ticks for the zoomed-in plot
    ax_zoom.set_xlim(zoom_xlim)
    ax_zoom.set_ylim(zoom_ylim)
    ax_zoom.set_xticks([1e-6, 1e-5])
    ax_zoom.set_yticks([round(y.min(), 1), round(y.max(), 1)])

    # Draw a box around the zoomed-in plot in the main plot
    mark_inset(ax, ax_zoom, loc1=3, loc2=4, fc="none", ec="0.5")

    # Hide the ticks and labels for the zoomed-in plot in the main plot
    plt.setp(ax_zoom.get_xticklabels(), visible=False)
    plt.setp(ax_zoom.get_yticklabels(), visible=False)
    plt.setp(ax_zoom.get_xticklines(), visible=False)
    plt.setp(ax_zoom.get_yticklines(), visible=False)

    # plot horizontal line at ocp to calculate i_corr
    ax.axhline(ocp, color="grey", linestyle="--")
    ax.scatter(
        i_corr,
        E_corr,
        color=color,
        label=r"{}: $E_{{corr}}$ = {:.2f} V, $i_{{corr}}$ = {:.2f} $\mu$A/cm$^{{2}}$".format(
            model, E_corr, i_corr * 10**6
        ),
    )


def store_polarization_curve_features(E: np.ndarray, i: np.ndarray, model: str) -> list[Any]:
    ocp = get_ocps_machine_learning_models(E, i)
    (
        E_applied,
        i_applied_log_abs,
        slope,
        intercept,
        rvalue,
        std_error_slope,
        intercept_stderr,
    ) = linreg_tafel_line_ORR_or_HER(ocp, E, i)
    # first value in the tafel line should be where E = ocp, solve for current
    i_applied_log_abs = np.insert(i_applied_log_abs, 0, (ocp - intercept) / slope)
    E_applied = np.insert(E_applied, 0, ocp)
    E_corr, i_corr = corrosion_potential_and_current_density(slope, i_applied_log_abs[0], intercept)

    return [
        E_applied,
        i_applied_log_abs,
        ocp,
        slope,
        intercept,
        rvalue,
        std_error_slope,
        intercept_stderr,
        E_corr,
        i_corr,
    ]


def plot_experimental_testing_data(ph, loc1, loc2, df_features: pd.DataFrame) -> None:
    """
    Plots the experimental filtered curves for all pHs in the test data set
    """

    # potential, abs(current density)
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)

    E, i = X_test_ph[:, 0], 10**y_test_ph
    # store polarization curve features
    (
        [E_applied, i_applied_log_abs, ocp, slope, intercept, rvalue, std_error_slope, intercept_stderr, E_corr, i_corr]
    ) = store_polarization_curve_features(E, i, "exp")
    df_features["Exp. data"] = [ocp, slope, rvalue**2, std_error_slope, intercept_stderr, E_corr, i_corr]

    figures_to_plot_exp_data_in = [
        ax_pred,
        ax_compare_anns,
        ax_individual_model_vs_exp_rf[loc1, loc2],
        ax_individual_model_vs_exp_cb[loc1, loc2],
        ax_individual_model_vs_exp_lgb[loc1, loc2],
        ax_individual_model_vs_exp_xgb[loc1, loc2],
        ax_individual_model_vs_exp_ann[loc1, loc2],
    ]
    for idx, fig in enumerate(figures_to_plot_exp_data_in):
        fig.semilogx(
            10**y_test_ph,
            X_test_ph[:, 0],
            label=f"Exp. data, pH = {ph}" if idx >= 2 else f"Exp., pH = {ph}",
            color="k",
            linestyle="-",
        )

        if fig not in [ax_pred, ax_compare_anns]:
            plot_tafel_lines_E_corr_i_corr(
                fig,
                i_applied_log_abs,
                E_applied,
                slope,
                intercept,
                ocp,
                rvalue,
                E_corr,
                i_corr,
                "blue",
                "Exp_data",
                std_error_slope,
                intercept_stderr,
            )


def random_forest_comparison(ph, loc1, loc2, df_features: pd.DataFrame) -> None:
    """
    The RF data is trained and stored from train_models.py
    """
    X_test_ph, _ = filter_x_y_boolean_mask(ph)
    path = "models_data/random_forest_output"
    # load current density prediction
    # current_density_pred = np.loadtxt(f"{path}/current_density_pred.csv", skiprows=1, usecols=2)
    current_density_pred = pd.read_csv(f"{path}/current_density_pred_ph_{ph}.csv", sep="\t")["Current density [A/cm2]"]

    # store features
    E = X_test_ph[:, 0]
    i = current_density_pred.to_numpy()
    (
        [E_applied, i_applied_log_abs, ocp, slope, intercept, rvalue, std_error_slope, intercept_stderr, E_corr, i_corr]
    ) = store_polarization_curve_features(E, i, "RF")
    df_features["RF"] = [ocp, slope, rvalue**2, std_error_slope, intercept_stderr, E_corr, i_corr]

    linestyle, color = linestyles_and_markers_for_model_comparisons("rf")
    figures_to_plot_pred_in = [ax_pred, ax_individual_model_vs_exp_rf[loc1, loc2]]
    for ax in figures_to_plot_pred_in:
        ax.semilogx(
            current_density_pred,
            X_test_ph[:, 0],
            label="Random Forest" if ax == ax_pred else f"RF, pH = {ph}",
            linestyle=linestyle,
            color=color,
        )
        # plot tafel lines for each pH and have it in appendix
        if ax == ax_individual_model_vs_exp_rf[loc1, loc2]:
            plot_tafel_lines_E_corr_i_corr(
                ax,
                i_applied_log_abs,
                E_applied,
                slope,
                intercept,
                ocp,
                rvalue,
                E_corr,
                i_corr,
                color,
                "RF",
                std_error_slope,
                intercept_stderr,
            )


def catboost_comparison(ph, store_mape: list, loc1, loc2, df_features: pd.DataFrame) -> None:
    cb = CatBoostRegressor()
    cb.load_model("models_saved/catboost_model.cbm")
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)
    y_pred_ph = cb.predict(X_test_ph)
    linestyle, color = linestyles_and_markers_for_model_comparisons("cb")

    # store features
    E, i = X_test_ph[:, 0], 10**y_pred_ph
    (
        [E_applied, i_applied_log_abs, ocp, slope, intercept, rvalue, std_error_slope, intercept_stderr, E_corr, i_corr]
    ) = store_polarization_curve_features(E, i, "CB")
    df_features["CB"] = [ocp, slope, rvalue**2, std_error_slope, intercept_stderr, E_corr, i_corr]

    figures_to_plot_pred_in = [ax_pred, ax_individual_model_vs_exp_cb[loc1, loc2]]
    for ax in figures_to_plot_pred_in:
        ax.semilogx(
            i,
            E,
            label="CatBoost" if ax == ax_pred else f"CB, pH = {ph}",
            linestyle=linestyle,
            color=color,
        )
        # plot tafel lines for each pH and have it in appendix
        if ax == figures_to_plot_pred_in[-1]:
            plot_tafel_lines_E_corr_i_corr(
                ax,
                i_applied_log_abs,
                E_applied,
                slope,
                intercept,
                ocp,
                rvalue,
                E_corr,
                i_corr,
                color,
                "CB",
                std_error_slope,
                intercept_stderr,
            )

    store_mape.append(mape(y_pred_ph, y_test_ph))


def plot_train_val_loss_catboost():
    df_train_loss = pd.read_csv("catboost_info/learn_error.tsv", sep="\t")
    ax_loss_trees.plot(df_train_loss["iter"], df_train_loss["RMSE"], label="CatBoost", color="k", linestyle="-")
    df_val_loss = pd.read_csv("catboost_info/test_error.tsv", sep="\t")
    ax_val_loss_trees.plot(df_val_loss["iter"], df_val_loss["RMSE"], label="Catboost", color="k", linestyle="-")

    plot_scatter_if_early_stopping(
        "cb", df_train_loss["iter"].iloc[-1], df_train_loss["RMSE"].iloc[-1], df_val_loss["RMSE"].iloc[-1]
    )


def xgboost_comparison(ph, store_mape, loc1, loc2, df_features: pd.DataFrame) -> None:
    xgb = XGBRegressor()
    xgb.load_model("models_saved/xgboost.txt")
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)
    # prediction of log10(abs(current density))
    y_pred = xgb.predict(X_test_ph)
    store_mape.append(mape(y_pred, y_test_ph) * 100)
    # store features
    E, i = X_test_ph[:, 0], 10**y_pred
    (
        [E_applied, i_applied_log_abs, ocp, slope, intercept, rvalue, std_error_slope, intercept_stderr, E_corr, i_corr]
    ) = store_polarization_curve_features(E, i, "XGB")
    df_features["XGB"] = [ocp, slope, rvalue**2, std_error_slope, intercept_stderr, E_corr, i_corr]

    linestyle, color = linestyles_and_markers_for_model_comparisons("xgb")
    figures_to_plot_pred_in = [ax_pred, ax_individual_model_vs_exp_xgb[loc1, loc2]]
    for ax in figures_to_plot_pred_in:
        ax.semilogx(
            i,
            E,
            label="XGBoost" if ax == ax_pred else f"XGB, pH = {ph}",
            linestyle=linestyle,
            color=color,
        )
        # plot tafel lines for each pH and have it in appendix
        if ax == figures_to_plot_pred_in[-1]:
            plot_tafel_lines_E_corr_i_corr(
                ax,
                i_applied_log_abs,
                E_applied,
                slope,
                intercept,
                ocp,
                rvalue,
                E_corr,
                i_corr,
                color,
                "XGB",
                std_error_slope,
                intercept_stderr,
            )


def plot_train_val_loss_xgboost():
    # plot rmse for each iter
    df = pd.read_csv("models_data/xgboost_info/train_val_loss.csv", sep="\t")
    ax_loss_trees.plot(df["iter"], df["train_loss_rmse"], label="XGBoost", color="#555555", linestyle="--")
    ax_val_loss_trees.plot(df["iter"], df["val_loss_rmse"], label="XGBoost", color="#555555", linestyle="--")

    plot_scatter_if_early_stopping(
        "xgb", df["iter"].iloc[-1], df["train_loss_rmse"].iloc[-1], df["val_loss_rmse"].iloc[-1]
    )


def lgbm_comparison(ph, store_mape, loc1, loc2, df_features: pd.DataFrame) -> None:
    lgbm = lgb.Booster(model_file="models_saved/lgbm.txt")
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)
    # prediction of log10(abs(current density))
    y_pred = lgbm.predict(X_test_ph)
    store_mape.append(mape(y_pred, y_test_ph) * 100)
    # store features
    E, i = X_test_ph[:, 0], 10**y_pred
    (
        [E_applied, i_applied_log_abs, ocp, slope, intercept, rvalue, std_error_slope, intercept_stderr, E_corr, i_corr]
    ) = store_polarization_curve_features(E, i, "LGBM")
    df_features["LGB"] = [ocp, slope, rvalue**2, std_error_slope, intercept_stderr, E_corr, i_corr]

    linestyle, color = linestyles_and_markers_for_model_comparisons("lgb")

    figures_to_plot_pred_in = [ax_pred, ax_individual_model_vs_exp_lgb[loc1, loc2]]
    for ax in figures_to_plot_pred_in:
        ax.semilogx(
            i,
            E,
            label="LightGBM" if ax == ax_pred else f"LGB, pH = {ph}",
            linestyle=linestyle,
            color=color,
        )
        # plot tafel lines for each pH and have it in appendix
        if ax == figures_to_plot_pred_in[-1]:
            plot_tafel_lines_E_corr_i_corr(
                ax,
                i_applied_log_abs,
                E_applied,
                slope,
                intercept,
                ocp,
                rvalue,
                E_corr,
                i_corr,
                color,
                "LGB",
                std_error_slope,
                intercept_stderr,
            )


def plot_train_val_loss_lgbm():
    # plot rmse for each iter
    # plot rmse for each iter
    df = pd.read_csv("models_data/lgbm_info/train_val_loss.csv", sep="\t")
    ax_loss_trees.plot(df["iter"], df["train_loss_rmse"], label="LightGBM", color="#777777", linestyle=":")
    ax_val_loss_trees.plot(df["iter"], df["val_loss_rmse"], label="LightGBM", color="#777777", linestyle=":")

    plot_scatter_if_early_stopping(
        "lgbm", df["iter"].iloc[-1], df["train_loss_rmse"].iloc[-1], df["val_loss_rmse"].iloc[-1]
    )


def ANN_comparison(ph, store_mape, loc1, loc2, df_features: pd.DataFrame) -> None:
    _, _, _, _, x_scaler, y_scaler = normalize_data_for_ANN()
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)

    # plot the two best models to compare hyperparameters
    files: list[str] = os.listdir("tuning_results")
    for file in files:
        if file.endswith(".h5"):
            model = keras.models.load_model(f"tuning_results/{file}")  # type: ignore
            y_pred_log = y_scaler.inverse_transform(model.predict(x_scaler.fit_transform(X_test_ph)))
            y_pred = 10**y_pred_log
            which_model: str = file.split("_")[0]

            # store error only if best model
            if file == "first_best_model.h5":
                # store features
                E, i = X_test_ph[:, 0], y_pred.reshape(-1)
                try:
                    (
                        [
                            E_applied,
                            i_applied_log_abs,
                            ocp,
                            slope,
                            intercept,
                            rvalue,
                            std_error_slope,
                            intercept_stderr,
                            E_corr,
                            i_corr,
                        ]
                    ) = store_polarization_curve_features(E, i, "ANN")
                except ValueError:
                    (
                        [
                            E_applied,
                            i_applied_log_abs,
                            ocp,
                            slope,
                            intercept,
                            rvalue,
                            std_error_slope,
                            intercept_stderr,
                            E_corr,
                            i_corr,
                        ]
                    ) = store_polarization_curve_features(E, 10**y_test_ph, "ANN")

                df_features["ANN"] = [ocp, slope, rvalue**2, std_error_slope, intercept_stderr, E_corr, i_corr]

                linestyle, color = linestyles_and_markers_for_model_comparisons("ann")

                figures_to_plot_pred_in = [ax_pred, ax_compare_anns, ax_individual_model_vs_exp_ann[loc1, loc2]]
                for idx, ax in enumerate(figures_to_plot_pred_in):
                    ax.semilogx(
                        i,
                        E,
                        label=f"ANN {which_model}" if idx < 2 else f"ANN, pH = {ph}",
                        linestyle=linestyle,
                        color=color,
                    )
                    # plot tafel lines for each pH and have it in appendix
                    if ax == figures_to_plot_pred_in[-1]:
                        plot_tafel_lines_E_corr_i_corr(
                            ax,
                            i_applied_log_abs,
                            E_applied,
                            slope,
                            intercept,
                            ocp,
                            rvalue,
                            E_corr,
                            i_corr,
                            color,
                            "ANN",
                            std_error_slope,
                            intercept_stderr,
                        )
                store_mape.append(mape(y_pred_log, y_test_ph) * 100)
            else:
                ax_compare_anns.semilogx(y_pred, X_test_ph[:, 0], label=f"ANN {which_model}")


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
        values = row.drop("pH").values
        sorting_indices = np.argsort(values)  # type: ignore
        # all rows are sorted in ascending order, rearange all used lists
        sorted_values = [values[i] for i in sorting_indices]
        sorted_labels = [labels[i] for i in sorting_indices]
        for idx, val in enumerate(sorted_values):
            ax.bar(
                locs[i] + idx * 2 * bar_width,
                val,
                bar_width,
                label=sorted_labels[idx] if i == 0 else None,  # only display label once
                color=colors[idx],
                hatch=hatches[idx],
            )

    ax.set_xlim(locs[0] / 2, locs[-1] + 1)

    # Add legend and axis labels
    ax.legend()
    ax.set_ylabel("Mean Absolute Percentage Error of log10(|i|) [%]")
    ax.set_ylim(0, 15)
    ax.set_xticks(
        [(i + 2 * 2 * bar_width) for i in locs], labels=[f"pH = {ph}" for ph in best_scores_mape_log["pH"]], rotation=45
    )
    fig.tight_layout()
    fig.savefig("summarized_data_figures_datafiles/mape_log_%_histogram.png")


if __name__ == "__main__":
    # prediction loss of log(abs(current density)) in Mean Absolute Percentage Errror
    best_scores_mape_log = pd.DataFrame()
    best_scores_mape_log["pH"] = pd.read_csv("testing_pHs.csv", sep="\t")["test_pHs"]
    # store mape_log for RF
    best_scores_mape_log["rf"] = (
        pd.read_csv("models_data/random_forest_output/errors_trees_best_params.csv", sep="\t")["(MAPE of log)"] * 100
    )
    plot_train_val_loss_catboost()
    plot_train_val_loss_xgboost()
    plot_train_val_loss_lgbm()
    plot_train_val_loss_ann_best_model()

    loss_figures = [fig_loss_trees, fig_val_loss_trees, fig_loss_ANN]
    fig_names = ["training_loss_GBDTS", "validation_loss_GBDTS", "train_val_loss_ANN"]
    for loss_fig, fig_name in zip(loss_figures, fig_names):
        loss_fig.legend(loc="upper right")
        loss_fig.savefig(f"summarized_data_figures_datafiles/{fig_name}.png")

    df_features = pd.read_csv("testing_pHs.csv", sep="\t")
    # compare results for all pHs
    store_mape_catboost = []
    store_mape_xgboost = []
    store_mape_lgbm = []
    store_mape_ann = []
    ax_locs_appendix = [[0, 1], [1, 1], [1, 0], [0, 0]]

    # do preds for each pH in the test data set
    for ax_loc, ph in zip(ax_locs_appendix, df_features["test_pHs"]):
        loc1, loc2 = ax_loc[0], ax_loc[1]

        """" Create dataframe to store calculated parameters from the curves. store to tex (latex) format """
        df_features = pd.DataFrame()
        df_features.insert(
            0,
            "",
            value=[
                "OCP [V]",
                "Tafel slope $[$mv/dec$]$",
                "R\\textsuperscript{2} value slope",
                "Std. error slope (1e3)",
                "Std. error intercept (1e3)",
                "E\\textsubscript{corr} [V]",
                "i\\textsubscript{corr} \\text{[$\mu$A cm\\textsuperscript{-2}]}",
            ],
        )

        plot_experimental_testing_data(ph, loc1, loc2, df_features)
        catboost_comparison(ph, store_mape_catboost, loc1, loc2, df_features)
        random_forest_comparison(ph, loc1, loc2, df_features)
        xgboost_comparison(ph, store_mape_xgboost, loc1, loc2, df_features)
        lgbm_comparison(ph, store_mape_lgbm, loc1, loc2, df_features)
        ANN_comparison(ph, store_mape_ann, loc1, loc2, df_features)

        # create dataframe to store important parameters from the curves. Save as tex file too
        df_features.iloc[-1, 1:] *= 10**6  # convert to micro A/cm^2 for i_corr
        df_features.iloc[1, 1:] *= 1000  # convert to mV/dec for tafel slope
        df_features.iloc[3:5, 1:] *= 1000  # multiply with 1000 for convenience
        df_features["Mean error ML"] = df_features.iloc[:, 1:].mean(axis=1)
        df_features["Mean error ML - Exp. data"] = df_features["Mean error ML"] - df_features["Exp. data"]
        df_features["MAPE"] = abs(df_features["Mean error ML - Exp. data"] / df_features["Exp. data"]) * 100
        # Select all columns except for the first column
        cols_to_round = df_features.columns[1:]
        # Round the selected columns to 3 decimal places
        df_features[cols_to_round] = df_features[cols_to_round].applymap(lambda x: f"{x:.3g}")
        df_features.to_csv(f"summarized_data_figures_datafiles/csv_files/df_features{ph}.csv", sep="\t", index=False)

        """Plot figures"""
        ax_list_appendix_plots = [
            ax_individual_model_vs_exp_rf,
            ax_individual_model_vs_exp_cb,
            ax_individual_model_vs_exp_lgb,
            ax_individual_model_vs_exp_xgb,
            ax_individual_model_vs_exp_ann,
        ]
        for ax in ax_list_appendix_plots:
            ax[loc1, loc2].legend(loc="upper left")

        # save figs
        try:
            ax_pred.set_xlabel("|i| [A/cm$^2$]")
            ax_pred.set_ylabel("E [V]")
            ax_pred.legend(loc="upper left")
            fig_pred.savefig(
                f"summarized_data_figures_datafiles/comparison_with_exp_results/comparison_with_exp_data_ph_{ph}.png"
            )
            ax_pred.clear()

            ax_compare_anns.set_xlabel("|i| [A/cm$^2$]")
            ax_compare_anns.set_ylabel("E [V]")
            ax_compare_anns.legend(loc="upper left")
            fig_compare_anns.savefig(f"summarized_data_figures_datafiles/comparison_of_anns_with_exp_ph_{ph}.png")
            ax_compare_anns.clear()

        except Exception as e:
            raise e

    best_scores_mape_log["cb"] = store_mape_catboost
    best_scores_mape_log["xgb"] = store_mape_xgboost
    best_scores_mape_log["lgb"] = store_mape_lgbm
    best_scores_mape_log["ann"] = store_mape_ann
    best_scores_mape_log = best_scores_mape_log.sort_values(by="pH").reset_index(drop=True)
    best_scores_mape_log.to_csv(
        "summarized_data_figures_datafiles/csv_files/mape_of_pred_log%.csv", sep="\t", index=False
    )
    plot_histogram_mape_models(best_scores_mape_log)

    # Appendix scientific paper
    for fig, model in zip([fig_rf, fig_cb, fig_lgb, fig_ann, fig_xgb], ["rf", "cb", "lgb", "ann", "xgb"]):
        fig.tight_layout()
        fig.savefig(f"summarized_data_figures_datafiles/appendix/{model}.png")
