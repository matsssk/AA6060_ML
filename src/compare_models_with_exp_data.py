import matplotlib.pyplot as plt
import matplotlib

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
import os
import functools
import subprocess
import matplotlib
import matplotlib.pyplot as plt

import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt

pdflatex_path = "/usr/bin/pdflatex"
matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

# Create a simple plot
# fig, ax = plt.subplots()
# ax.plot([0, 1], [0, 1], label=r"$\frac{1}{x}$")
# ax.set_xlabel("X axis")
# ax.set_ylabel("Y axis")
# ax.legend()

# # Save the plot as a PDF file
# plt.savefig("test_plot.pgf")


import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import mean_squared_error as mse
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
from src.get_selected_features import plot_confidence_interval, lower_upper_confidence_interval_slope
from typing import Any
from src.get_selected_features import linreg_tafel_line_ORR_or_HER, get_ocps_machine_learning_models
from tensorflow import keras
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

plt.rcParams["axes.labelsize"] = 22  # Font size for x and y axis labels
plt.rcParams["xtick.labelsize"] = 22  # Font size for x-axis tick labels
plt.rcParams["ytick.labelsize"] = 22  # Font size for y-axis tick labels
plt.rcParams["legend.fontsize"] = 22  # Font size for legend

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
    ax_loss_trees.set_ylabel("Loss, RMSE [log|i|]")
    ax_loss_trees.set_yscale("log")
    ax_loss_trees2 = ax_loss_trees.twinx()  # plot loss gradients
    ax_loss_trees2.set_ylabel("Loss Gradient, RMSE [log|i|]")
    ax_loss_trees2.set_yscale("log")

    # # plot the zoomed portion with markers
    # zoom_scale = 0.2
    # zoom_width = 0.5
    # zoom_height = zoom_width * zoom_scale
    # sub_axes = fig_loss_trees.add_axes([0.8, 0.8, 0.25, 0.25])
    # sub_axes.set_xlim([0, 100])
    # sub_axes.set_ylim([1 - zoom_height, 1 + zoom_height])

    # # validation loss GBDTs
    # fig_val_loss_trees = plt.figure(figsize=(5,5))
    # ax_val_loss_trees = fig_val_loss_trees.subplots()
    # ax_val_loss_trees.set_yscale("log")
    # ax_val_loss_trees.set_xlabel("Iterations")
    # ax_val_loss_trees.set_ylabel("Validation loss, RMSE")
    # ax_val_loss_trees2 = ax_val_loss_trees.twinx()  # plot loss gradients
    # ax_val_loss_trees2.set_yscale("log")
    # ax_loss_trees2.set_ylabel("Loss Gradient")

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
        ax_loss_trees.scatter(iterations, val_loss_last_iter, label="Early stopping", color="k", marker="x")


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

    # ax_zoom.semilogx(x, y, color=color)
    plot_confidence_interval(
        i_applied_log_abs=i_applied_log_abs,
        slope=slope,
        slope_std_error=slope_std_error,
        intercept=intercept,
        intercept_std_error=intercept_std_error,
        ax=ax,
        color=color,
        model=model,
    )

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

    _, _, lower_slope, upper_slope = lower_upper_confidence_interval_slope(i_applied_log_abs, slope, std_error_slope, intercept, intercept_stderr)  # type: ignore
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
        lower_slope,
        upper_slope,
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
            lower_ci,
            upper_ci,
        ]
    ) = store_polarization_curve_features(E, i, "exp")
    df_features["Exp. data"] = [
        ocp,
        slope,
        rvalue**2,
        std_error_slope,
        lower_ci,
        upper_ci,
        intercept_stderr,
        E_corr,
        i_corr,
    ]
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
            lower_ci,
            upper_ci,
        ]
    ) = store_polarization_curve_features(E, i, "RF")
    df_features["RF"] = [ocp, slope, rvalue**2, std_error_slope, lower_ci, upper_ci, intercept_stderr, E_corr, i_corr]

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


def catboost_comparison(ph, store_mape: list, rmse_catboost: list, loc1, loc2, df_features: pd.DataFrame) -> None:
    cb = CatBoostRegressor()
    cb.load_model("models_saved/catboost_model.cbm")
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)
    y_pred_ph = cb.predict(X_test_ph)
    store_mape.append(mape(y_test_ph, y_pred_ph) * 100)
    rmse_catboost.append(mse(y_test_ph, y_pred_ph, squared=False))
    linestyle, color = linestyles_and_markers_for_model_comparisons("cb")

    # store features
    E, i = X_test_ph[:, 0], 10**y_pred_ph
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
            lower_ci,
            upper_ci,
        ]
    ) = store_polarization_curve_features(E, i, "CB")
    df_features["CB"] = [ocp, slope, rvalue**2, std_error_slope, lower_ci, upper_ci, intercept_stderr, E_corr, i_corr]

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


def sma_loss_gradients(val_loss: np.ndarray | pd.Series, window: Optional[int] = 3) -> np.ndarray:
    loss_gradients_val_loss = abs(np.diff(val_loss[::100]))
    window = np.ones(3) / 3  # type: ignore
    sma = np.convolve(loss_gradients_val_loss, window, mode="valid")
    return sma


def plot_train_val_loss_catboost():
    df_train_loss = pd.read_csv("catboost_info/learn_error.tsv", sep="\t")
    ax_loss_trees.plot(
        df_train_loss["iter"], df_train_loss["RMSE"], label="CatBoost training loss", color="black", linestyle="-"
    )
    df_val_loss = pd.read_csv("catboost_info/test_error.tsv", sep="\t")
    ax_loss_trees.plot(
        df_val_loss["iter"], df_val_loss["RMSE"], label="Catboost validation loss", color="black", linestyle="--"
    )

    # loss_gradients_val_loss = abs(np.diff(df_val_loss["RMSE"][::100]))
    loss_iter = df_val_loss["iter"][::100][1:]
    # ax_loss_trees2.scatter(loss_iter, loss_gradients_val_loss)

    sma = sma_loss_gradients(df_val_loss["RMSE"])
    ax_loss_trees2.semilogy(
        loss_iter[1:-1], sma, label="SMA CatBoost validation loss gradients", marker="s", color="black"
    )

    plot_scatter_if_early_stopping(
        "cb", df_train_loss["iter"].iloc[-1], df_train_loss["RMSE"].iloc[-1], df_val_loss["RMSE"].iloc[-1]
    )


def xgboost_comparison(ph, store_mape, rmse_xgboost: list, loc1, loc2, df_features: pd.DataFrame) -> None:
    xgb = XGBRegressor()
    xgb.load_model("models_saved/xgboost.txt")
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)
    # prediction of log10(abs(current density))
    y_pred = xgb.predict(X_test_ph)
    store_mape.append(mape(y_test_ph, y_pred) * 100)
    rmse_xgboost.append(mse(y_test_ph, y_pred, squared=False))
    # store features
    E, i = X_test_ph[:, 0], 10**y_pred
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
            lower_ci,
            upper_ci,
        ]
    ) = store_polarization_curve_features(E, i, "XGB")
    df_features["XGB"] = [
        ocp,
        slope,
        rvalue**2,
        std_error_slope,
        lower_ci,
        upper_ci,
        intercept_stderr,
        E_corr,
        i_corr,
    ]

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
    ax_loss_trees.plot(df["iter"], df["train_loss_rmse"], label="XGBoost training loss", color="#555555", linestyle="-")
    ax_loss_trees.plot(
        df["iter"], df["val_loss_rmse"], label="XGBoost validation loss", color="#555555", linestyle="--"
    )

    sma = sma_loss_gradients(df["val_loss_rmse"])
    ax_loss_trees2.semilogy(
        df["iter"][::100][1:][1:-1], sma, label="SMA XGBoost validation loss gradients", marker="^", color="black"
    )
    plot_scatter_if_early_stopping(
        "xgb", df["iter"].iloc[-1], df["train_loss_rmse"].iloc[-1], df["val_loss_rmse"].iloc[-1]
    )


def lgbm_comparison(ph, store_mape, rmse_lgb, loc1, loc2, df_features: pd.DataFrame) -> None:
    lgbm = lgb.Booster(model_file="models_saved/lgbm.txt")
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)
    # prediction of log10(abs(current density))
    y_pred = lgbm.predict(X_test_ph)
    store_mape.append(mape(y_test_ph, y_pred) * 100)
    rmse_lgb.append(mse(y_test_ph, y_pred, squared=False))
    # store features
    E, i = X_test_ph[:, 0], 10**y_pred
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
            lower_ci,
            upper_ci,
        ]
    ) = store_polarization_curve_features(E, i, "LGBM")
    df_features["LGB"] = [
        ocp,
        slope,
        rvalue**2,
        std_error_slope,
        lower_ci,
        upper_ci,
        intercept_stderr,
        E_corr,
        i_corr,
    ]

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
    ax_loss_trees.plot(
        df["iter"], df["train_loss_rmse"], label="LightGBM training loss", color="#777777", linestyle=":"
    )
    ax_loss_trees.plot(
        df["iter"], df["val_loss_rmse"], label="LightGBM validation loss", color="#777777", linestyle=":"
    )

    sma = sma_loss_gradients(df["val_loss_rmse"])
    ax_loss_trees2.semilogy(
        df["iter"][::100][1:][1:-1], sma, label="SMA LightGBM validation loss gradients", marker="o", color="black"
    )

    plot_scatter_if_early_stopping(
        "lgbm", df["iter"].iloc[-1], df["train_loss_rmse"].iloc[-1], df["val_loss_rmse"].iloc[-1]
    )


def ANN_comparison(ph, store_mape, rmse_ann, loc1, loc2, df_features: pd.DataFrame) -> None:
    _, _, _, _, x_scaler, y_scaler = normalize_data_for_ANN()
    X_test_ph, y_test_ph = filter_x_y_boolean_mask(ph)

    # plot the two best models to compare hyperparameters
    files: list[str] = os.listdir("tuning_results_ANN")
    for file in files:
        if file.endswith(".h5"):
            model = keras.models.load_model(f"tuning_results_ANN/{file}")  # type: ignore
            y_pred_log = y_scaler.inverse_transform(model.predict(x_scaler.fit_transform(X_test_ph)))
            y_pred = 10**y_pred_log
            which_model: str = file.split("_")[0]

            # store error only if best model
            if file == "first_best_model.h5":
                store_mape.append(mape(y_test_ph, y_pred_log) * 100)
                rmse_ann.append(mse(y_test_ph, y_pred_log, squared=False))
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
                            lower_ci,
                            upper_ci,
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
                            lower_ci,
                            upper_ci,
                        ]
                    ) = store_polarization_curve_features(E, 10**y_test_ph, "ANN")

                df_features["ANN"] = [
                    ocp,
                    slope,
                    rvalue**2,
                    std_error_slope,
                    lower_ci,
                    upper_ci,
                    intercept_stderr,
                    E_corr,
                    i_corr,
                ]

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

            else:
                ax_compare_anns.semilogx(y_pred, X_test_ph[:, 0], label=f"ANN {which_model}")


def plot_train_val_loss_ann_best_model():
    # plot loss from training for best model
    df_loss = pd.read_csv("models_data/ANN_info/training_val_loss0", sep="\t")
    epochs = [iter for iter in range(1, len(df_loss["val_rmse"]) + 1, 1)]
    # plot epochs vs rmse (root of mse which is the loss given in df) for best model
    ax_loss_ANN.semilogy(epochs, df_loss["rmse"], "s-", label="ANN training loss best model", color="r")
    ax_loss_ANN.semilogy(epochs, df_loss["val_rmse"], "s--", label="ANN validation loss best model", color="r")


# def plot_histogram_mape_rmse_models(best_scores_mape_log: pd.DataFrame, best_scores_rmse_log: pd.DataFrame):
#     fig, ax = plt.subplots(figsize=(5,5))
#     ax2 = ax.twinx()  # Create a second y-axis

#     bar_width = 0.035  # Reduced bar width to fit both MAPE and RMSE values
#     labels = ["RF", "CB", "XGB", "LGB", "ANN"]
#     colors = ["k", "#555555", "#777777", "#999999", "#CCCCCC"]  # grey scales
#     hatches = [None, "/", None, "/", None]
#     locs = [1, 2, 3, 4]

#     # Loop over each row
#     for i, (row_mape, row_rmse) in enumerate(zip(best_scores_mape_log.iterrows(), best_scores_rmse_log.iterrows())):
#         values_mape = row_mape[1].drop("pH").values
#         values_rmse = row_rmse[1].drop("pH").values
#         sorting_indices = np.argsort(values_mape)  # type: ignore

#         sorted_values_mape = [values_mape[i] for i in sorting_indices]
#         sorted_values_rmse = [values_rmse[i] for i in sorting_indices]
#         sorted_labels = [labels[i] for i in sorting_indices]

#         for idx, (val_mape, val_rmse) in enumerate(zip(sorted_values_mape, sorted_values_rmse)):
#             ax.bar(
#                 locs[i] + idx * 2 * bar_width,
#                 val_mape,
#                 bar_width,
#                 label=sorted_labels[idx] if i == 0 else None,  # only display label once
#                 color=colors[idx],
#                 hatch=hatches[idx],
#             )
#             ax2.bar(
#                 locs[i] + idx * 2 * bar_width + bar_width,
#                 val_rmse,
#                 bar_width,
#                 color=colors[idx],
#                 hatch=hatches[idx],
#                 alpha=0.5,  # Added alpha to differentiate between MAPE and RMSE values
#             )

#     ax.set_xlim(locs[0] / 2, locs[-1] + 1)

#     # Add legend and axis labels
#     ax.legend()
#     ax.set_ylabel("Mean Absolute Percentage Error of log10(|i|) [%]")
#     ax.set_ylim(0, 15)

#     ax2.set_ylabel("Root Mean Squared Error of log10(|i|)")
#     ax2.set_ylim(best_scores_rmse_log.stack().min() * 0.9, best_scores_rmse_log.stack().min() * 1.1)  # type: ignore

#     ax.set_xticks(
#         [(i + 2 * 2 * bar_width) for i in locs], labels=[f"pH = {ph}" for ph in best_scores_mape_log["pH"]], rotation=45
#     )
#     fig.tight_layout()
#     fig.savefig("summarized_data_figures_datafiles/mape_rmse_log_%_histogram.pgf")

from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_mape_rmse_pointers(best_scores_mape_log: pd.DataFrame, best_scores_rmse_log: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax2 = ax.twinx()  # Create a second y-axis
    bar_width = 0.35
    labels = ["RF", "CB", "XGB", "LGB", "ANN"]
    locs = np.arange(len(best_scores_mape_log))
    colors = {"RF": "green", "CB": "red", "XGB": "blue", "LGB": "pink", "ANN": "brown"}

    for i, (mape_row, rmse_row) in enumerate(zip(best_scores_mape_log.iterrows(), best_scores_rmse_log.iterrows())):
        mape_values = mape_row[1].drop("pH").values
        rmse_values = rmse_row[1].drop("pH").values

        # Plot highest MAPE bar
        ax.bar(locs[i] - bar_width / 2, np.max(mape_values), bar_width, color="dimgray", alpha=0.5)
        # Plot lines for individual MAPE values
        for _, (val, label) in enumerate(zip(mape_values, labels)):
            ax.hlines(
                val,
                locs[i] - bar_width / 2 - bar_width / 2,
                locs[i] - bar_width / 2 + bar_width / 2,
                color=colors[label],
            )

        # Plot highest RMSE bar
        ax2.bar(locs[i] + bar_width / 2, np.max(rmse_values), bar_width, color="darkgray", alpha=0.5)
        # Plot lines for individual RMSE values
        for _, (val, label) in enumerate(zip(rmse_values, labels)):
            ax2.hlines(
                val,
                locs[i] + bar_width / 2 - bar_width / 2,
                locs[i] + bar_width / 2 + bar_width / 2,
                color=colors[label],
            )

    # Add horizontal lines for average MAPE and RMSE values
    avg_mape = best_scores_mape_log.drop("pH", axis=1).mean().mean()
    avg_rmse = best_scores_rmse_log.drop("pH", axis=1).mean().mean()
    ax.axhline(avg_mape, linestyle="--", color="dimgray", linewidth=1)
    ax2.axhline(avg_rmse, linestyle="--", color="darkgray", linewidth=1)
    ylower, yupper = (
        np.min(best_scores_rmse_log.drop("pH", axis=1).values) * 0.9,
        np.max(best_scores_rmse_log.drop("pH", axis=1).values) * 1.1,
    )

    ax2.set_ylim(ylower, yupper)

    ax.set_xticks(locs)
    ax.set_xticklabels(
        [f"MAPE      RMSE\n\npH = {ph}" for ph in best_scores_mape_log["pH"]], rotation=0, ha="center", fontsize=10
    )

    ax.set_ylabel("Mean Absolute Percentage Error of log10(|i|) [%]")
    ax.set_ylim(0, 15)

    ax2.set_ylabel("Root Mean Squared Error of log10(|i|)")

    # Create legend
    legend_elements = [Line2D([0], [0], color=colors[label], lw=2, label=label) for label in labels]
    ax.legend(handles=legend_elements, loc="upper left")

    fig.tight_layout()
    fig.savefig("summarized_data_figures_datafiles/mape_rmse_pointers.pgf")


def plot_losses():
    plot_train_val_loss_catboost()
    plot_train_val_loss_xgboost()
    plot_train_val_loss_lgbm()
    plot_train_val_loss_ann_best_model()

    handles1, labels1 = ax_loss_trees.get_legend_handles_labels()
    handles2, labels2 = ax_loss_trees2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    # Add the merged legend to the first subplot to obtain merged legends
    ax_loss_trees.legend(handles, labels)
    fig_loss_trees.tight_layout()
    fig_loss_ANN.legend()
    fig_loss_ANN.tight_layout()
    for ftype in ["pgf", "pdf"]:
        fig_loss_trees.savefig(f"summarized_data_figures_datafiles/learning_curves_GBDTS.{ftype}")
        fig_loss_ANN.savefig(f"summarized_data_figures_datafiles/train_val_loss_ANN.{ftype}")


if __name__ == "__main__":
    plot_losses()

    best_scores_mape_log, best_scores_rmse_log = pd.DataFrame(), pd.DataFrame()
    best_scores_mape_log["pH"] = pd.read_csv("testing_pHs.csv", sep="\t")["test_pHs"]
    best_scores_rmse_log["pH"] = pd.read_csv("testing_pHs.csv", sep="\t")["test_pHs"]
    # compare results for all pHs
    mape_catboost, mape_xgboost, mape_lgbm, mape_ann = [], [], [], []
    rmse_catboost, rmse_xgboost, rmse_lgbm, rmse_ann = [], [], [], []

    df_features = pd.read_csv("testing_pHs.csv", sep="\t")
    ax_locs_appendix = [[0, 1], [1, 1], [1, 0], [0, 0]]

    # do preds for each pH in the test data set
    for ax_loc, ph in zip(ax_locs_appendix, df_features["test_pHs"]):
        loc1, loc2 = ax_loc[0], ax_loc[1]

        # Create dataframe to store calculated parameters from the curves. store to tex (latex) format
        df_features = pd.DataFrame()
        df_features.insert(
            0,
            "",
            value=[
                "OCP [V]",
                "Tafel slope $[$mv/dec$]$",
                "R\\textsuperscript{2} value slope",
                "Std. error slope (10$^3$)",
                "Lower limit 99\\% confidence interval",
                "Upper limit 99\\% confidence interval",
                "Std. error intercept (10$^3$)",
                "E\\textsubscript{corr} [V]",
                "i\\textsubscript{corr} \\text{[$\mu$A cm\\textsuperscript{-2}]}",
            ],
        )

        plot_experimental_testing_data(ph, loc1, loc2, df_features)
        catboost_comparison(ph, mape_catboost, rmse_catboost, loc1, loc2, df_features)
        random_forest_comparison(ph, loc1, loc2, df_features)
        xgboost_comparison(ph, mape_xgboost, rmse_xgboost, loc1, loc2, df_features)
        lgbm_comparison(ph, mape_lgbm, rmse_lgbm, loc1, loc2, df_features)
        ANN_comparison(ph, mape_ann, rmse_ann, loc1, loc2, df_features)

        # create dataframe to store important parameters from the curves. Save as tex file too
        df_features.iloc[-1, 1:] *= 10**6  # convert to micro A/cm^2 for i_corr
        df_features.iloc[[1, 3, 4, 5, 6], 1:] *= 1000
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
                f"summarized_data_figures_datafiles/comparison_with_exp_results/comparison_with_exp_data_ph_{ph}.pgf"
            )
            ax_pred.clear()

            ax_compare_anns.set_xlabel("|i| [A/cm$^2$]")
            ax_compare_anns.set_ylabel("E [V]")
            ax_compare_anns.legend(loc="upper left")
            fig_compare_anns.savefig(f"summarized_data_figures_datafiles/comparison_of_anns_with_exp_ph_{ph}.pgf")
            ax_compare_anns.clear()

        except Exception as e:
            raise e

    # store mape_log for RF
    best_scores_mape_log["rf"], best_scores_rmse_log["rf"] = (
        pd.read_csv("models_data/random_forest_output/errors_trees_best_params.csv", sep="\t")["(MAPE of log)"] * 100
    ), (pd.read_csv("models_data/random_forest_output/errors_trees_best_params.csv", sep="\t")["rmse"])
    best_scores_mape_log["cb"], best_scores_rmse_log["cb"] = mape_catboost, rmse_catboost
    best_scores_mape_log["xgb"], best_scores_rmse_log["xgb"] = mape_xgboost, rmse_xgboost
    best_scores_mape_log["lgb"], best_scores_rmse_log["lgb"] = mape_lgbm, rmse_lgbm
    best_scores_mape_log["ann"], best_scores_rmse_log["ann"] = mape_ann, rmse_ann
    best_scores_mape_log, best_scores_rmse_log = best_scores_mape_log.sort_values(by="pH").reset_index(
        drop=True
    ), best_scores_rmse_log.sort_values(by="pH").reset_index(drop=True)

    plot_mape_rmse_pointers(best_scores_mape_log, best_scores_rmse_log)
    best_scores_mape_log.to_csv(
        "summarized_data_figures_datafiles/csv_files/mape_of_pred_log%.csv", sep="\t", index=False
    )
    best_scores_rmse_log.to_csv(
        "summarized_data_figures_datafiles/csv_files/rmse_of_pred_log.csv", sep="\t", index=False
    )

    # Appendix scientific paper
    for fig, model in zip([fig_rf, fig_cb, fig_lgb, fig_ann, fig_xgb], ["rf", "cb", "lgb", "ann", "xgb"]):
        fig.tight_layout()
        fig.savefig(f"summarized_data_figures_datafiles/appendix/{model}.pgf")
