from src.load_data import load_raw_data
from src.filter_raw_data import remove_first_cath_branch
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import norm
from src.train_models import load_ANN_runtime

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
# change matplotlib to store pgf files. this to make matplotlib more compatible
# with latex
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


def plot_E_pit_ph10_2(folder: str = "raw_data"):
    plt.figure(figsize=(4, 3.5))

    plt.xlabel(r"Absolute value of current density ($|\mathit{i}|$) [A/cm$^2$]")
    plt.ylabel("Potential ($E$) vs Ref. [V]")

    file_path = os.path.join(folder, "ph10,2.DTA")
    potential, current_density = load_raw_data(file_path)
    i_filtered, E_filtered = remove_first_cath_branch(current_density, potential)
    # crop cathodic branch
    mask = (E_filtered > -1.0) & (E_filtered < -0.6)
    E_filtered, i_filtered = E_filtered[mask], i_filtered[mask]

    plt.semilogx(abs(i_filtered), E_filtered, color="k")
    plt.text(2e-7, -0.63, "$E_{{\mathrm{{pit}}}}$")
    plt.tight_layout()
    for ftype in ["pgf", "pdf"]:
        plt.savefig(f"sketches_for_report/E_pit.{ftype}")


def overfit_underfit_good_fit():
    fig_u, ax_u = plt.subplots(figsize=(2.5, 2.5))
    fig_o, ax_o = plt.subplots(figsize=(2.5, 2.5))
    fig_gf, ax_gf = plt.subplots(figsize=(2.5, 2.5))

    # fig_gf, ax_gf = plt.subplots(figsize=(5 / 2, 5 / 2))

    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    noise = 0.2 * np.random.normal(size=len(x))
    y_with_noise = y + noise
    ax_u.plot(x, np.array([0] * len(x)), color="black", label="Underfit")  # underfit
    ax_o.plot(x, y_with_noise, color="black", label="Overfit")
    ax_gf.plot(x, y, color="black", label="Good fit")  # good fit

    # Plot the scatter plot
    for ax in [ax_gf, ax_u, ax_o]:
        ax.scatter(x, y_with_noise, color="dimgray", s=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.legend(loc="upper right")

    for ftype in ["pgf", "pdf"]:
        fig_u.savefig(f"sketches_for_report/underfit.{ftype}")
        fig_o.savefig(f"sketches_for_report/overfit.{ftype}")
        fig_gf.savefig(f"sketches_for_report/good_fit.{ftype}")


def tafel_plot():
    i0 = 10**-5  # A/cm^2
    alpha_c = 0.5
    alpha_a = 1 - alpha_c
    n = 2
    F = 96485
    R = 8.314
    T = 298
    Erev = -0.6
    E = np.linspace(-0.2, -1.0, 100)
    eta = E - Erev
    constant = 1 / np.log10(np.e)
    ian = 10 ** (alpha_a * n * F * eta / (constant * R * T))
    icat = -(10 ** (-alpha_c * n * F * eta / (constant * R * T)))

    inet = i0 * (ian + icat)

    plt.figure(figsize=(4, 3.5))
    plt.xlabel(r"Absolute value of current density ($|\mathit{i}|$) [A/cm$^2$]")
    plt.ylabel("Potential ($E$) vs Ref. [V]")

    slope_an, intercept_an, r_an, p_an, se_an = stats.linregress(np.log10(ian), E, alternative="two-sided")
    slope_cat, intercept_cat, r_cat, p_cat, se_cat = stats.linregress(np.log10(abs(icat)), E, alternative="two-sided")
    print(p_cat)
    plt.semilogx(
        ian * i0, intercept_an + slope_an * np.log10(ian), label="Anodic tafel slope", linestyle="dashed", color="k"
    )
    plt.semilogx(
        abs(icat * i0),
        intercept_cat + slope_cat * np.log10(abs(icat)),
        label="Cathodic tafel slope",
        linestyle="dashed",
        color="k",
    )

    plt.semilogx(abs(inet), E, label="E(log|i|), BV", color="k")

    plt.xlim(1 * 10**-6, 1)

    plt.axhline(-0.6, color="grey", linestyle="dashed")
    plt.axvline(10**-5, color="grey", linestyle="dashed")

    plt.text(1.1 * 10**-5, -1.02, "$i_{corr}$")
    plt.text(0.22, -0.65, "$E_{corr}$")
    plt.text(5 * 10**-5, -0.4, "Anodic Tafel line")
    plt.text(5 * 10**-5, -0.81, "Cathodic Tafel line")
    # plt.annotate(
    #     "Cathodic branch with extrapolated Tafel line",
    #     xytext=(5 * 10**-5, -0.9 - 0.05),
    #     #xy=(5.45 * 10**-3, -0.78),
    #     #arrowprops=dict(arrowstyle="->"),
    # )
    # plt.annotate(
    #     "Anodic branch with extrapolated Tafel line",
    #     xy=(1.47 * 10**-3, -0.45),
    #     xytext=(5 * 10**-5, -0.32 + 0.05),
    #     arrowprops=dict(arrowstyle="->"),
    # )

    plt.tight_layout()
    for ftype in ["pdf", "pgf"]:
        plt.savefig(f"sketches_for_report/polarisation_curve_illustration.{ftype}")


def diffusion():
    i0 = 10**-5  # A/cm^2
    ilim_a = 10**-4
    ilim_c = 10**-6
    alpha_c = 0.5
    alpha_a = 1 - alpha_c

    n = 2
    F = 96485
    R = 8.314
    T = 298
    Erev = -0.6
    E = np.linspace(-0.2, -1.0, 100)
    eta = E - Erev
    ian = (i0 * np.e ** (alpha_a * n * F * eta / (R * T))) / (
        1 + i0 / ilim_a * np.e ** (alpha_a * n * F * eta / (R * T))
    )
    icat = -(i0 * np.e ** (-alpha_c * n * F * eta / (R * T))) / (
        1 + i0 / ilim_c * np.e ** (-alpha_c * n * F * eta / (R * T))
    )

    inet = ian + icat

    plt.figure(figsize=(4, 3.5))
    # plt.title('Polarisation curve fictive reaction, Butler-Volmer')
    plt.xlabel(r"Absolute value of current density ($|\mathit{i}|$) [A/cm$^2$]")
    plt.ylabel("Potential ($E$) vs Ref. [V]")

    # slope_an, intercept_an, r_an, p_an, se_an = stats.linregress(np.log10(ian), E, alternative='two-sided')
    # slope_cat, intercept_cat, r_cat, p_cat, se_cat = stats.linregress(np.log10(abs(icat)), E, alternative='two-sided')

    # plt.semilogx(ian*i0, intercept_an + slope_an*np.log10(ian), label = 'Anodic tafel slope', linestyle = 'dashed', color = 'r')#, label = 'Tafel slope anodic, R= {}'.format(round(r_an,5)), linestyle = 'dashed')
    # plt.semilogx(abs(icat*i0), intercept_cat + slope_cat*np.log10(abs(icat)), label = 'Cathodic tafel slope', linestyle = 'dashed', color = 'b')#, label = 'Tafel slope cathodic , R = {}'.format(round(-r_cat,5)), linestyle = 'dashed')

    plt.semilogx(abs(inet), E, label="E(log|i|), BV", color="k")

    # plt.xlim(2*10**-6, 9*10**-5)

    plt.axvline(ilim_a, color="grey", linestyle="dashed")
    plt.axvline(ilim_c, color="grey", linestyle="dashed")
    plt.text(2.8 * 10**-5, -1, "$i_{lim, anodic}$")
    plt.text(1.1 * 10**-6, -1, "$i_{lim, cathodic}$")
    plt.tick_params(axis="x")
    plt.tick_params(axis="y")
    plt.tight_layout()
    for ftype in ["pdf", "pgf"]:
        plt.savefig(f"sketches_for_report/diffusion.{ftype}")


def pourbaix_diagram():
    # pourbaix diagram for Aluminium

    plt.figure(figsize=(4, 3.5))
    plt.xlabel("pH")
    plt.ylabel("Potential ($E$) vs Ref. [V]")
    x = np.linspace(-2, 16, 1000000)
    y = np.linspace(-3.5, 1.5, 1000000)
    plt.xlim(0, 13)
    plt.ylim(-3, 1.5)

    ph4_99 = np.where(x >= 4.90)[0][0]
    ph7_78 = np.where(x >= 7.78)[0][0]

    # horisontal line. OK
    plt.hlines(-1.794, xmin=0, xmax=4.9, color="black")

    # line between Al and Al2O3. OK
    plt.plot(x[(ph4_99 - 1) : (ph7_78 + 1)], -1.504 - 0.0591 * x[(ph4_99 - 1) : (ph7_78 + 1)], color="black")

    # right vline between Al2O3 and AlO2-, OK

    plt.vlines(7.78, ymin=-1.9634, ymax=1.5, color="black")

    # left vline
    plt.vlines(4.90, ymin=-1.7937, ymax=1.5, color="black")

    # slope between Al and AlO2-
    plt.plot(x[(ph7_78 + 1) :], -1.350 - 0.0788 * x[(ph7_78 + 1) :], color="black")

    # HER
    plt.plot(x, -0.0591 * x, linestyle="dashed", color="grey")

    # ORR
    plt.plot(x, 1.23 - 0.0591 * x, linestyle="dashed", color="grey")

    # inserting text in plot

    plt.text(2, -1, "Al$^{3+}$")
    plt.text(1.8, -1.3, "(Active)")

    plt.text(6, -2.5, "Al")
    plt.text(5.5, -2.8, "(Immune)")

    plt.text(5.5, -1, "$\\alpha$-Al$_2$O$_3$")
    plt.text(5.4, -1.3, "(Passive)")
    plt.text(10, 0.2, "AlO$_2^-$")
    plt.text(9.7, -0.1, "(Active)")
    plt.text(2, 0, "HER")
    plt.text(2, 1.2, "ORR")

    plt.tick_params(axis="x")
    plt.tick_params(axis="y")
    plt.tight_layout()
    for ftype in ["pdf", "pgf"]:
        plt.savefig(f"sketches_for_report/pourbaix.{ftype}")


def plot_feature_imp_as_func_of_iter():
    df = pd.read_csv("models_data/random_forest_output/results_from_tuning/feature_importances.csv", sep="\t")
    df_1 = df[df["max_features"] == 1.0]
    df_03 = df[df["max_features"] == 0.3]

    fig, ax1 = plt.subplots(figsize=(3.5, 3))

    ax1.set_xlabel("Number of estimators/trees (n\\_estimators)")
    ax1.set_ylabel("Potential ($E$) / pH")
    ax1.scatter(
        df_03["n_estimators"],
        df_03["potential (E)"] / df_03["pH"],
        color="gray",
        marker="o",
        label="max\\_features = 0.3",
    )

    # ax2.set_ylabel("Potential / pH, max\\_features = 1.0, marker = square", color="black")
    ax1.scatter(
        df_1["n_estimators"],
        df_1["potential (E)"] / df_1["pH"],
        color="black",
        marker="s",
        label="max\\_features = 1.0",
    )
    ax1.legend()

    fig.tight_layout()
    fig.savefig("summarized_data_figures_datafiles/pdf_plots/rf_feature_imp_plot.pdf")
    fig.savefig("summarized_data_figures_datafiles/pgf_plots/rf_feature_imp_plot.pgf")


def lgbm_tuning_last_iterations_before_termination_rmse():
    df = pd.read_excel("models_data/lgbm_info/tuning_last_iterations_before_termination_excel.xlsx")
    fig, ax1 = plt.subplots(figsize=(3.5, 3))

    iter = df["iter"]
    train = df["train_rmse"]
    val = df["val_rmse"]
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("RMSE [log($i$)]")
    ax1.plot(iter, train, label="Training loss", color="k")
    ax1.plot(iter, val, label="Validation loss", color="gray", linestyle="--")
    ax1.legend()
    fig.tight_layout()
    fig.savefig("summarized_data_figures_datafiles/pdf_plots/lgbm_last_iter_tuning_loss.pdf")
    fig.savefig("summarized_data_figures_datafiles/pgf_plots/lgbm_last_iter_tuning_loss.pgf")


def plot_training_times_per_DT():
    plt.figure(figsize=(2.7, 2.7))
    # plt.xlabel("ML algorithm")
    plt.ylabel("Training time per tree [s/tree]")
    df = pd.read_csv("summarized_data_figures_datafiles/csv_files/training_times_per_tree_DTs.csv", sep="\t")

    pos = [0, 1, 2, 3]
    plt.bar(df["Model"], df["Time"], width=0.25, color="gray")
    plt.xticks(
        pos, labels=[f" {k.upper()}: {round(v,3)} s" for k, v in zip(df["Model"], df["Time"])], rotation=45, ha="center"
    )

    plt.tight_layout()
    plt.savefig("summarized_data_figures_datafiles/pgf_plots/models_training_time_per_tree_DTs.pgf")
    plt.savefig("summarized_data_figures_datafiles/pdf_plots/models_training_time_per_tree_DTs.pdf")


def convert_seconds_ANN(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"ANN: {hours}h, {minutes}m"


def plot_training_times_tot_all_models():
    fig, ax1 = plt.subplots(figsize=(2.7, 2.7))

    # ax1.set_xlabel("ML algorithm")
    ax1.set_ylabel("Training time [s]", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Training time [hrs]", color="tab:red")

    df = pd.read_csv("summarized_data_figures_datafiles/csv_files/training_times_all_models.csv", sep="\t")

    df1 = df[df["Model"] != "ANN"]
    df2 = df[df["Model"] == "ANN"]

    ax1.bar(df1["Model"], df1["Time"], width=0.25, color="tab:blue", label="DTs")
    ax2.bar(df2["Model"], df2["Time"] / 3600, width=0.25, color="tab:red", label="ANN")

    ax1.set_ylim(0, max(df1["Time"]) + 5)
    ax2.set_ylim(0, df2["Time"].values[0] / 3600 + 0.1)
    label_ANN = convert_seconds_ANN(df2["Time"].values[0])

    labels = [f"{k.upper()}: {round(v, 2)} s" for k, v in zip(df1["Model"], df1["Time"])] + [label_ANN]
    ax1.set_xticks(range(len(labels)))  # ensure that the number of ticks matches the number of labels
    ax1.set_xticklabels(labels, rotation=45, ha="center")

    plt.tight_layout()
    plt.savefig("summarized_data_figures_datafiles/pgf_plots/models_training_time_all_models.pgf")
    plt.savefig("summarized_data_figures_datafiles/pdf_plots/models_training_time_all_models.pdf")


def plot_residuals_linreg_tafel():
    residuals_txt_files = os.listdir("linreg_residuals")
    length_residuals = sum([len(np.loadtxt("linreg_residuals/" + file)) for file in residuals_txt_files])

    standardized_residuals = np.zeros(length_residuals)
    residuals = np.zeros_like(standardized_residuals)
    for file in residuals_txt_files:
        res = np.loadtxt("linreg_residuals/" + file)

        mean_residuals = np.mean(res)

        std_residuals = np.std(res)

        # standardize the residuals
        std_res = (res - mean_residuals) / std_residuals
        # find the first occurence of 0 in standardized_residuals
        idx_where_0 = np.argwhere(standardized_residuals == 0)[0][0]

        standardized_residuals[idx_where_0 : (idx_where_0 + len(std_res))] = std_res
        residuals[idx_where_0 : (idx_where_0 + len(res))] = res

    fig_normality_test, ax_normality_test = plt.subplots(figsize=(3, 3))
    hist, bins = np.histogram(standardized_residuals, bins="auto")
    ax_normality_test.bar(bins[:-1], hist, align="edge", width=np.diff(bins), color="dimgray")
    ax_normality_test.set_xlabel("Standardized residuals of linear \n regression of Tafel line (Z-score)")
    ax_normality_test.set_ylabel(f"Samples, N$_{{\\mathrm{{tot}}}}$ = {len(standardized_residuals)}")
    ax_normality_test.set_xlim(min(standardized_residuals), max(standardized_residuals))

    # plot normal dist curve on 2nd y axis
    ax2 = ax_normality_test.twinx()
    x = np.linspace(-3, 3, 100)
    p = norm.pdf(x, 0, 1)  # mean=0, std=1
    ax2.plot(x, p, color="k")
    ax2.set_ylabel("Probability density")
    # plot 99\% quantiles
    mu = 0
    sigma = 1
    # q = norm.ppf(0.99, mu, sigma)
    q = 2.575829
    # Get the y-value at the 99% quantile
    y_q = norm.pdf(q, mu, sigma)
    ax2.plot([q, q], [0, y_q], color="r", linestyle=":", label="99% quantile")
    ax2.plot([-q, -q], [0, y_q], color="r", linestyle=":", label="99% quantile")
    ax2.set_ylim(0, 0.41)

    #### plot NOT standardized residuals
    fig_res, ax_res = plt.subplots(figsize=(3, 3))
    hist_res_res, bins_res = np.histogram(residuals, bins="auto")
    ax_res.bar(bins_res[:-1], hist_res_res, align="edge", width=np.diff(bins_res), color="dimgray")
    ax_res.set_xlabel("Residuals of linear \n regression of Tafel line")
    ax_res.set_ylabel(f"Samples, N$_{{\\mathrm{{tot}}}}$ = {len(residuals)}")
    ax_res.set_xlim(min(residuals), max(residuals))

    # save standardized
    # text box quantile
    ax_normality_test.text(-6.5, 350, "{} = 0.005".format(r"$\alpha$"), color="r")
    fig_normality_test.tight_layout()
    fig_normality_test.savefig("summarized_data_figures_datafiles/appendix/standardized_residuals_linreg_tafel.pgf")
    fig_normality_test.savefig("summarized_data_figures_datafiles/appendix/standardized_residuals_linreg_tafel.pdf")

    # save NOT standardized
    fig_res.tight_layout()
    fig_res.savefig("summarized_data_figures_datafiles/appendix/linreg_residuals_normality_test.pgf")
    fig_res.savefig("summarized_data_figures_datafiles/appendix/linreg_residuals_normality_test.pdf")


def plot_histogram_feature_importances_DTs():
    df = pd.read_csv("summarized_data_figures_datafiles/csv_files/feature_importances.csv", sep="\t")

    # Model colors
    # colors = {"rf": "tab:green", "cb": "tab:red", "xgb": "tab:blue", "lgbm": "tab:purple"}
    colors_pot = {"rf": "dimgray", "cb": "dimgray", "xgb": "dimgray", "lgbm": "dimgray"}
    colors_ph = {"rf": "k", "cb": "k", "xgb": "k", "lgbm": "k"}
    # Setup plot
    barWidth = 0.1
    r1 = np.arange(len(df["Potential"]))
    r2 = [x + barWidth for x in r1]

    fig, ax = plt.subplots(figsize=(3.7, 3.7))
    ax.bar(
        r1,
        df["Potential"],
        color=[colors_pot[model] for model in df["Model"]],
        width=barWidth,
        label="Potential",
    )

    ax.bar(r2, df["pH"], color=[colors_ph[model] for model in df["Model"]], width=barWidth, label="pH")

    ax.set_xticks([r + barWidth / 2 for r in range(len(df["Potential"]))], [model.upper() for model in df["Model"]])

    ax.set_ylim([0, 1.0])
    ax.set_ylabel("Importance")

    # plot averages
    potential_mean = df["Potential"].mean()
    ax.hlines(
        potential_mean, xmin=-1, xmax=4, label=f"Potential mean = {potential_mean:.2f}", linestyle="--", color="dimgray"
    )
    ph_mean = df["pH"].mean()
    ax.hlines(ph_mean, xmin=-1, xmax=4, label=f"pH mean = {ph_mean:.2f}", linestyle=":", color="k")
    ax.set_xlim(-0.5, 3.5)
    ax.legend()
    fig.savefig("summarized_data_figures_datafiles/pgf_plots/histogram_feat_imp_DTs.pgf")
    fig.savefig("summarized_data_figures_datafiles/pdf_plots/histogram_feat_imp_DTs.pdf")


df_E_pit_E_corr_ = pd.read_csv("csv_files_E_corr_E_pit/ocp_t0_vs_ph.csv", sep="\t")


def plot_Ecorr_as_function_of_ph():
    df = pd.read_csv("selected_features.csv", sep="\t")
    phs = df["pH"]
    Ecorrs = df["OCP_t_half"]
    deltaE = df["delta_OCP"]

    fig_Ecorr, ax_Ecorr = plt.subplots(figsize=(2.5, 2.5))
    ax_Ecorr.set_xlabel("pH")
    ax_Ecorr.set_ylabel("$E$\\textsubscript{corr\\textsubscript{th}} vs SCE [V]")
    phs = list(np.arange(2.0, 12.2, 0.2))
    ax_Ecorr.set_xticks(np.arange(2, 13, 2))
    ax_Ecorr.scatter(phs, Ecorrs, s=4, color="k")

    fig_delta_E, ax_delta_E = plt.subplots(figsize=(3, 2.5))
    ax_delta_E.set_xlabel("pH")
    ax_delta_E.set_ylabel(
        "$\\Delta E$, $E$\\textsubscript{corr\\textsubscript{t0}} - $E$\\textsubscript{corr\\textsubscript{th}} [V]"
    )
    ax_delta_E.set_xticks(np.arange(2, 13, 2))
    ax_delta_E.scatter(phs, deltaE, s=4, color="k")
    ax_delta_E_percentage = ax_delta_E.twinx()
    ax_delta_E_percentage.scatter(phs, deltaE / df["OCP_t0"] * 100, color="tab:red", s=4, marker="x")
    ax_delta_E_percentage.set_ylabel(
        "$E$\\textsubscript{corr\\textsubscript{t0}} / $E$\\textsubscript{corr\\textsubscript{th}} $\cdot$ 100 [%] ",
        color="tab:red",
    )

    mask = np.array(phs) < 6.1
    phs_2_6 = np.array(phs)[mask]
    ocps_ph_2_6 = np.array(Ecorrs)[mask]
    avg_2_6 = np.mean(ocps_ph_2_6)

    avg_residual_ph_2_6 = np.mean(abs((ocps_ph_2_6 - avg_2_6)))
    ax_Ecorr.hlines(avg_2_6, xmin=2, xmax=6, linestyle="--", color="dimgray")
    ax_Ecorr.text(
        2, -1.2, "Avg. $E$\\textsubscript{corr\\textsubscript{th}}" + f"\n $\in [2.0, 6.0]$ \n = {round(avg_2_6,3)} V"
    )

    ax_Ecorr.annotate("", xy=(3, -0.74), xytext=(3, -0.94), arrowprops=dict(facecolor="dimgray", shrink=0.002))

    df_E_pit_E_corr_["OCP_th"] = Ecorrs
    df_E_pit_E_corr_["delta"] = deltaE

    fig_Ecorr.tight_layout()

    fig_Ecorr.savefig("summarized_data_figures_datafiles/appendix/Ecorr_vs_ph.pgf")
    fig_Ecorr.savefig("summarized_data_figures_datafiles/appendix/E_corr_vs_ph.pdf")

    fig_delta_E.tight_layout()
    fig_delta_E.savefig("summarized_data_figures_datafiles/appendix/deltaE_vs_ph.pgf")
    fig_delta_E.savefig("summarized_data_figures_datafiles/appendix/deltaE_vs_ph.pdf")


def plot_E_pit_manually_written():
    df = pd.read_excel("summarized_data_figures_datafiles/csv_files/E_pit_visible.xlsx")
    ph, E_pit = df["pH"], df["E_pit"]
    df_E_pit_E_corr_["pH_Epit"] = ph
    df_E_pit_E_corr_["Epit"] = E_pit

    plt.figure(figsize=(2.5, 2.5))
    plt.xlabel("pH")
    plt.ylabel("$E$\\textsubscript{pit} vs SCE [V]")
    plt.scatter(ph, E_pit, color="k", s=6)
    plt.axhline(E_pit.mean(), color="dimgray", linestyle="--")
    plt.text(6, -0.69, "Avg. $E$\\textsubscript{pit}" + f"\n = {round(E_pit.mean(),2)} V")
    plt.xticks(np.arange(2, 11, 2))
    plt.annotate(
        "", xy=(6.5, E_pit.mean() - 0.002), xytext=(6.5, -0.68), arrowprops=dict(facecolor="dimgray", shrink=0.002)
    )
    plt.tight_layout()

    plt.savefig("summarized_data_figures_datafiles/appendix/Epit_vs_ph.pgf")
    plt.savefig("summarized_data_figures_datafiles/appendix/Epit_vs_ph.pdf")


from src.load_data import list_of_filenames, load_raw_data
from src.filter_raw_data import remove_first_cath_branch


def plot_E_pit_sketch():
    file_path = "raw_data_without_gamry_noise/ph2,8.DTA"
    E, i = load_raw_data(file_path)
    i, E = remove_first_cath_branch(i, E)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.set_xlabel("Absolute value of current density $|i|$ [A/cm$^2$]")
    ax.set_ylabel("Potential $E$ vs SCE [V]")
    ax.semilogx(abs(i), E, color="k")
    ax.text(8 * 10**-7, -0.55, "$E$\\textsubscript{pit}", color="k")
    ax.annotate(
        "",
        xy=(6 * 10**-6, -0.625),
        xytext=(2 * 10**-6, -0.57),
        arrowprops=dict(facecolor="dimgray", shrink=0.15, width=0.5, headwidth=5, headlength=7),
    )
    ax.set_ylim(np.min(E) + 0.05, np.max(E) - 0.05)
    ax.set_xlim(10**-8, 10**-3)
    fig.tight_layout()

    fig.savefig("sketches_for_report/E_pit.pgf")
    fig.savefig("sketches_for_report/E_pit.pdf")


if __name__ == "__main__":
    # plot_histogram_feature_importances_DTs()
    # plot_E_pit_ph10_2()
    # overfit_underfit_good_fit()
    # plot_residuals_linreg_tafel()
    # tafel_plot()
    # diffusion()
    pourbaix_diagram()
    # plot_feature_imp_as_func_of_iter()
    # lgbm_tuning_last_iterations_before_termination_rmse()
    # plot_training_times_per_DT()
    # plot_training_times_tot_all_models()
    # plot_Ecorr_as_function_of_ph()
    # plot_E_pit_manually_written()
    # df_E_pit_E_corr_.to_csv("summarized_data_figures_datafiles/E_corr_E_pit_data.csv", sep="\t", index=False)
    # plot_E_pit_sketch()
