import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import tstd
from io import StringIO
from src.load_data import list_of_filenames
from src.plot_raw_data import get_grid_for_axs
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

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


def folder_with_ocps() -> str:
    return "ocp_raw_data"


def load_ocp_data(file_path: str) -> list[np.ndarray]:
    """
    Returns arrays with potential [V] and time [s] from file path

    :param file_path: file path of the folder to load data from
    """
    with open(file_path, "r", encoding="ISO-8859-1") as f:
        data = f.read().replace(",", ".")
    data = StringIO(data)
    try:
        matrix = np.loadtxt(data, skiprows=51, usecols=(1, 2))
    except ValueError:
        raise ValueError
    time, potential = matrix[:, 0], matrix[:, 1]

    return [time, potential]


def sort_files_based_on_ph(filename: str) -> float:
    return float(filename.split("h")[1].split(",")[0] + "." + filename.split(",")[1].split(".")[0])


def one_third_of_data(X: np.ndarray):
    return X[(int(len(X) / 3 * 2)) :]


def standard_deviation(X: np.ndarray) -> float:
    """
    Calculates the sample standard deviation for last 1/3 * N data points using Bessel's correction
    This accounts for the bias that the mean is calculated by the sample mean and
    therefor underestimates the true variance. Bessel's correction uses N-1 instead of N
    as denominator, resulting in a higher variance

    : param X: 1 dim array with potential as a function of time, E(t)
    """
    return tstd(X, limits=None)


# def normalize_data(ocp_vals: np.ndarray):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     return scaler.fit_transform(ocp_vals.reshape(-1, 1))


# def sma_ocps(ocps: np.ndarray):
#     """Calculates the 3 window SMA for the 30 seconds change in potential"""
#     delta_ocp = abs(np.diff(ocps[::30]))
#     window = np.ones(3) / 3  # type: ignore
#     sma = np.convolve(delta_ocp, window, mode="valid")
#     return sma


def deltaocps(ocps: np.ndarray):
    """Calculates the 3 window SMA for the 30 seconds change in potential"""
    delta_ocp = abs(np.diff(ocps[::30]))
    window = np.ones(3) / 3  # type: ignore
    sma = np.convolve(delta_ocp, window, mode="valid")
    return delta_ocp


if __name__ == "__main__":
    #  define figure to plot raw data in
    fig, ax = plt.subplots(2, 2)
    fig.supxlabel("Time [s]")
    fig.supylabel("E vs SCE [V]")


def plot_ocp_files():
    files: list[str] = sorted(list_of_filenames(folder=folder_with_ocps()), key=sort_files_based_on_ph)

    # the coordinates of the different subplots
    positions = [[0, 0], [0, 1], [1, 0], [1, 1]] * int(len(files) / 4 + 1)
    std_list, mean_ocps_for_phs, phs, residuals = [], [], [], []

    # SMA figure
    # fig_smas, ax_smas = plt.subplots()
    # ax_smas.set_ylabel("20 window SMA of gradient of OCP per time")
    # ax_smas.set_xlabel("Time")

    sma_list = []  # List to store the SMA arrays for each file

    for idx, file in enumerate(files):
        file_path = os.path.join(folder_with_ocps(), file)
        time, potential = load_ocp_data(file_path)
        one_third_of_ocps = one_third_of_data(potential)

        deltaocp_30s = deltaocps(one_third_of_ocps)
        x_values = one_third_of_data(time)[::30][: len(deltaocp_30s)]  # Ensure same length as sma_values
        # print(len(sma_values))
        # ax_smas.plot(x_values, sma_values)
        mean = np.mean(one_third_of_ocps)
        mean_ocps_for_phs.append(mean)
        std_list.append(standard_deviation(one_third_of_ocps))
        for residual in one_third_of_ocps - mean:
            residuals.append(residual)

        phs.append(float(file.split("h")[1].split(",")[0] + "." + file.split(",")[1].split(".")[0]))

        loc = positions[idx][0], positions[idx][1]
        ax[loc].plot(time, potential, label=f"pH: {sort_files_based_on_ph(file)}", color="black")
        ax[loc].set_ylim(np.min(potential) - 0.1, np.max(potential) + 0.1)
        ax[loc].legend()
        if (idx + 1) % 4 == 0 or (idx + 1) == len(files):
            get_grid_for_axs(ax)
            if idx + 1 == len(files) and (idx + 1) % 4 != 0:
                ax[1, 1].remove()

            fig.tight_layout()
            for ftype in ["pgf", "pdf"]:
                fig.savefig(f"ocp_plots/ocp_plots_{idx+1}.{ftype}")
            for subplot_ax in ax.flat:
                if (idx + 1) != len(files):
                    subplot_ax.clear()

        # Calculate SMA at all time positions
        if len(deltaocp_30s) == 39:
            sma_list.append(deltaocp_30s)

    pd.DataFrame({"pH": phs, "Corrected standard dev.": std_list}).to_csv(
        "summarized_data_figures_datafiles/csv_files/standard_dev_ocps.csv", index=False, sep="\t"
    )

    # Calculate the average SMA at each time position
    avg_delta_ocp30s = np.mean(sma_list, axis=0)

    # Plot the array of average SMA values
    fig_deltaocp, ax_deltaocp = plt.subplots(figsize=(3, 3))
    ax_deltaocp.plot(x_values, avg_delta_ocp30s * 1000, color="dimgray", linestyle="-", marker="o")

    ax_deltaocp.axhline(np.mean(avg_delta_ocp30s * 1000), label="Mean", linestyle="--", color="dimgray")
    ax_deltaocp.set_xlabel("Time [s]")
    # ax_avg_sma.set_ylabel(
    #     r"SMA (window = 3) of the 30 seconds change in potential $\left[\frac{\partial E}{\partial t}\right]$"
    # )
    ax_deltaocp.set_ylabel("Average $\Delta E_{{\\mathrm{{t}}}}(\Delta t = 30$ s) [mV]")
    fig_deltaocp.tight_layout()

    # Calculate and plot the normality test
    fig_normality_test, ax_normality_test = plt.subplots(figsize=(3, 3))

    hist, bins = np.histogram(residuals, bins="auto")
    ax_normality_test.bar(bins[:-1], hist, align="edge", width=np.diff(bins), color="dimgray")
    ax_normality_test.set_ylabel(f"Samples, N$_{{\\mathrm{{tot}}}}$ = {len(residuals)}")
    ax_normality_test.set_xlabel("OCP residuals from the mean")

    # create new figure for standardization
    fig_std, ax_std = plt.subplots(figsize=(3, 3))
    residuals_array = np.array(residuals)
    # the figure implies residual normality -> standardize data
    mean_residuals = np.mean(residuals_array)
    std_residuals = np.std(residuals_array)
    # standardize the residuals
    std_res = (residuals_array - mean_residuals) / std_residuals
    hist, bins = np.histogram(std_res, bins="auto")
    ax_std.bar(bins[:-1], hist, align="edge", width=np.diff(bins), color="dimgray")
    ax_std.set_xlabel("Standardized OCP residuals")
    ax_std.set_ylabel(f"Samples, N$_{{\\mathrm{{tot}}}}$ = {len(std_res)}")
    ax_std.set_xlim(-3, 3)

    # plot normal dist curve on 2nd y axis
    ax2 = ax_std.twinx()
    x = np.linspace(-3, 3, 100)
    p = norm.pdf(x, 0, 1)  # mean=0, std=1
    ax2.plot(x, p, color="k")
    ax2.set_ylabel("Probability density")

    fig_normality_test.tight_layout()
    fig_std.tight_layout()

    for ftype in ["pgf", "pdf"]:
        fig_normality_test.savefig(f"summarized_data_figures_datafiles/appendix/normality_test_ocp.{ftype}")

        fig_deltaocp.savefig(f"summarized_data_figures_datafiles/appendix/average_deltaocp_30s.{ftype}")

        fig_std.savefig(f"summarized_data_figures_datafiles/appendix/ocp_residuals_standardized_normal_dist.{ftype}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    plot_ocp_files()
    print(time.perf_counter() - t0)
