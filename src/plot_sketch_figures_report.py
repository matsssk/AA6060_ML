from src.load_data import load_raw_data
from src.filter_raw_data import remove_first_cath_branch
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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
    plt.figure()

    plt.xlabel("Current density ($i$) [A/cm$^2$]")
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


if __name__ == "__main__":
    plot_E_pit_ph10_2()
    overfit_underfit_good_fit()
