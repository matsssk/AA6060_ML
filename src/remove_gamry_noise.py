from src.plot_raw_data import list_of_filenames, load_raw_data
from src.filter_raw_data import remove_first_cath_branch
from src.get_selected_features import get_ocps
import matplotlib.pyplot as plt

# get Times New Roman font for matlotlib plots
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
import os
import numpy as np

plt.figure(figsize=(5, 5))

folder = "raw_data_without_gamry_noise"


for i in [0, 1, 2, 3, 4, 5, 6]:
    files = list_of_filenames(folder)

    file_path: str = os.path.join(folder, files[i])
    potential, current_density = load_raw_data(file_path)
    plt.semilogx(abs(current_density), potential, color="k", label=f"pH = {files[i]}")
    plt.legend()
    plt.show()
