from src.load_data import load_raw_data
from src.filter_raw_data import remove_first_cath_branch
import os
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def plot_E_pit_ph10_2(folder: str = "raw_data"):
    plt.figure(figsize=(10, 10))
    s = 16
    plt.xlabel("|i| [A/cm$^2$]", size=s)
    plt.ylabel("E [V]", size=s)

    file_path = os.path.join(folder, "ph10,2.DTA")
    potential, current_density = load_raw_data(file_path)
    i_filtered, E_filtered = remove_first_cath_branch(current_density, potential)
    # crop cathodic branch
    mask = (E_filtered > -1.0) & (E_filtered < -0.6)
    E_filtered, i_filtered = E_filtered[mask], i_filtered[mask]

    plt.semilogx(abs(i_filtered), E_filtered, color="k")
    plt.text(2e-7, -0.63, "$E_{{\mathrm{{pit}}}}$", size=s + 7)
    plt.xticks(fontsize=s)
    plt.yticks(fontsize=s)
    plt.savefig("sketches_for_report/E_pit.png")


plot_E_pit_ph10_2()
