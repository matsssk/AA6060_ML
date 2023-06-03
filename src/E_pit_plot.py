import matplotlib.pyplot as plt
from io import StringIO
import numpy as np
from src.load_data import list_of_filenames, load_raw_data
import os


files = list_of_filenames("raw_data")
phs_visible_pit = [
    2.8,
    3.0,
    3.6,
    4.2,
    4.8,
    5.2,
    5.6,
    5.8,
    6.4,
    6.6,
    6.8,
    7.0,
    7.2,
    7.6,
    7.8,
    8.0,
    8.8,
    9.2,
    9.6,
    10.2,
]


def plot_and_return_dataframe_with_filtered_data(folder_raw: str = "raw_data"):
    for idx, file in enumerate(files):
        file_path_raw: str = os.path.join(folder_raw, file)
        pH: float = float(file.split("h")[1].split(",")[0] + "." + file.split(",")[1].split(".")[0])

        if pH in phs_visible_pit:
            potential_raw, current_density_raw = load_raw_data(file_path_raw)
            plt.figure()
            plt.semilogx(abs(current_density_raw), potential_raw, label=f" pH : {pH}")
            plt.legend()
            plt.show()


plot_and_return_dataframe_with_filtered_data()
