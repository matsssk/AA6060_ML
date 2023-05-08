import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import tstd
from io import StringIO
from src.load_data import list_of_filenames


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


def standard_deviation(X: np.ndarray) -> float:
    """
    Calculates the sample standard deviation for last 1/3 * N data points using Bessel's correction
    This accounts for the bias that the mean is calculated by the sample mean and
    therefor underestimates the true variance. Bessel's correction uses N-1 instead of N
    as denominator, resulting in a higher variance

    : param X: 1 dim array with potential as a function of time, E(t)
    """
    n = int(len(X) / 3 * 2)
    X_frac = X[n:]
    return tstd(X_frac, limits=None)


if __name__ == "__main__":
    #  define figure to plot raw data in
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    fig.supxlabel("Time [s]")
    fig.supylabel("E vs SCE [V]")
    fig.tight_layout()


def plot_ocp_files():
    files: list[str] = sorted(list_of_filenames(folder=folder_with_ocps()), key=sort_files_based_on_ph)

    # the coordinates of the different subplots
    positions = [[0, 0], [0, 1], [1, 0], [1, 1]] * int(len(files) / 4 + 1)
    std_list = []
    for idx, file in enumerate(files):
        file_path = os.path.join(folder_with_ocps(), file)
        time, potential = load_ocp_data(file_path)
        std_list.append(standard_deviation(potential))

        loc = positions[idx][0], positions[idx][1]
        ax[loc].plot(time, potential, label=f"pH: {sort_files_based_on_ph(file)}", color="black")
        ax[loc].set_ylim(np.min(potential) - 0.1, np.max(potential) + 0.1)
        ax[loc].legend()
        if (idx + 1) % 4 == 0 or (idx + 1) == len(files):
            if idx + 1 == len(files) and (idx + 1) % 4 != 0:
                ax[1, 1].remove()

            fig.tight_layout()
            fig.savefig(f"ocp_plots/ocp_plots_{idx+1}")
            for subplot_ax in ax.flat:
                if (idx + 1) != len(files):
                    subplot_ax.clear()
    try:
        pH_list = list(np.arange(2.0, 12.2, 0.2))
        pd.DataFrame({"pH": pH_list, "Corrected standard dev.": std_list}).to_csv(
            "model_figures/standard_dev_ocps.csv", index=False, sep="\t"
        )
    except ValueError:
        print("Exception occured, missing some files")
        pH_list = [sort_files_based_on_ph(s) for s in files]
        pd.DataFrame({"pH": pH_list, "Corrected standard dev.": std_list}).to_csv(
            "model_figures/standard_dev_ocps.csv", index=False, sep="\t"
        )


if __name__ == "__main__":
    plot_ocp_files()
