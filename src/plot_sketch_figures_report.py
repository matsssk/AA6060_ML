from src.load_data import load_raw_data
from src.filter_raw_data import remove_first_cath_branch
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats

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
    plt.figure(figsize=(5, 4))

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

    plt.figure(figsize=(5, 4))
    plt.xlabel(r"Absolute value of current density ($|\mathit{i}|$) [A/cm$^2$]")
    plt.ylabel("Potential ($E$) vs Ref. [V]")

    slope_an, intercept_an, r_an, p_an, se_an = stats.linregress(np.log10(ian), E, alternative="two-sided")
    slope_cat, intercept_cat, r_cat, p_cat, se_cat = stats.linregress(np.log10(abs(icat)), E, alternative="two-sided")

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
    plt.text(0.32, -0.65, "$E_{corr}$")
    plt.text(5 * 10**-5, -0.9 - 0.05, "Cathodic branch with extrapolated Tafel line")
    plt.text(5 * 10**-5, -0.32 + 0.05, "Anodic branch with extrapolated Tafel line")
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

    plt.figure(figsize=(5, 4))
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

    plt.figure(figsize=(5, 4))
    plt.xlabel("pH")
    "Potential ($E$) vs Ref. [V]"
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

    plt.text(6, -1, "Al$_2$O$_3$")
    plt.text(5.7, -1.3, "(Passive)")
    plt.text(10, 0.2, "AlO$_2^-$")
    plt.text(9.7, -0.1, "(Active)")
    plt.text(2, 0, "HER")
    plt.text(2, 1.2, "ORR")

    plt.tick_params(axis="x")
    plt.tick_params(axis="y")

    for ftype in ["pdf", "pgf"]:
        plt.savefig(f"sketches_for_report/pourbaix.{ftype}")


if __name__ == "__main__":
    plot_E_pit_ph10_2()
    overfit_underfit_good_fit()
    tafel_plot()
    diffusion()
    pourbaix_diagram()
