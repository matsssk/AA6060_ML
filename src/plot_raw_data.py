import os
import time
import matplotlib.pyplot as plt
from src.load_data import list_of_filenames, load_raw_data

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

import pandas as pd
from src.filter_raw_data import remove_first_cath_branch
from src.get_selected_features import _get_ocps, linreg_tafel_line_ORR_or_HER


def plot_and_return_dataframe_with_filtered_data(
    folder: str = "raw_data",
    save_figs_raw_data: str = "raw_data_plots",
    save_figs_filtered_data: str = "filtered_raw_data_plots",
) -> pd.DataFrame:
    """
    Returns pd.DataFrame with all the filtered data into a dataframe of Potential, current density and pH
    The current density array consist of negative values as well

    Also plots all raw data and filtered data into separate folders

    :param folder: folder with raw data
    :param save_figs_raw_data: the folder to save raw data in
    :param save_figs_filtered_data: the folder to save filtered data in
    :return: returns plots of raw data and filtered dara in separate folders

    Note: pandas do not support this version (69) of stata, so numpy.loadtxt is applied
    """

    if __name__ == "__main__":
        #  define figure to plot raw data in
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        fig.supxlabel("|i| [A/cm$^2$]")
        fig.supylabel("E [V]")
        fig.tight_layout()

        # define figure to plot filtered data in
        fig2, ax2 = plt.subplots(2, 2, figsize=(15, 15))
        fig2.supxlabel("|i| [A/cm$^2$]")
        fig2.supylabel("E [V]")
        fig2.tight_layout()

        # figure to plot tafel slopes
        fig_tafel, ax_tafel = plt.subplots(2, 2, figsize=(15, 15))
        fig_tafel.supxlabel("|i| [A/cm$^2$]")
        fig_tafel.supylabel("E [V]")
        fig_tafel.tight_layout()

    # create empty dataframe and add data to it from all files
    df_merged_filtered = pd.DataFrame()
    files = list_of_filenames()
    # dataframe to store selected features
    selected_features_df = pd.DataFrame()

    for idx, file in enumerate(files):
        file_path = os.path.join(folder, file)
        potential, current_density = load_raw_data(file_path)
        current_density_filtered, potential_filtered = remove_first_cath_branch(current_density, potential)

        ocp_t0, ocp_t_half, delta_ocp = _get_ocps(potential, potential_filtered, current_density_filtered)
        try:
            # ORR or HER tafel slopes
            _, i_tafel_abs_log, tafel_slope, intercept, r_value, std_err = linreg_tafel_line_ORR_or_HER(
                ocp_t_half, potential_filtered, current_density_filtered
            )

        except ValueError:
            print(f"Something wrong with file {file}")
            raise ValueError

        df_add_to_selected_features = pd.DataFrame(
            {
                "pH": [float(file[2:5].replace(",", ".")) if file[4] != "," else float(file[2:6].replace(",", "."))],
                "OCP_t0": ocp_t0,
                "OCP_t_half": ocp_t_half,
                "delta_OCP": delta_ocp,
                "tafel_slope [mV/dec]": tafel_slope * 1000,
                "intercept [V]": intercept,
                "r_value_squared": r_value**2,
                "standard error": std_err,
            }
        )
        selected_features_df = pd.concat([selected_features_df, df_add_to_selected_features])
        # store selected features, E_corr, i_corr, tafel, etc...

        #  create a pandas DataFrame with the output from the filtering process
        #  for pH, the name format "ph2,0" etc. is necessary, otherwise the indices will
        #  be incorrect, [2:5] means e.g. the letters "2,0"
        df_to_add_to_df_merged = pd.DataFrame(
            {
                "Potential [V]": potential_filtered,
                "pH": [float(file[2:5].replace(",", ".")) if file[4] != "," else float(file[2:6].replace(",", "."))]
                * len(potential_filtered),
                "Current density [A/cm$^2$]": current_density_filtered,
            }
        )
        #  update the existing dataframe, adding the rows
        df_merged_filtered = pd.concat(
            [
                df_merged_filtered,
                df_to_add_to_df_merged,
            ]
        )
        # the coordinates of the different subplots
        positions = [[0, 0], [0, 1], [1, 0], [1, 1]] * int(len(files) / 4 + 1)
        if __name__ == "__main__":
            loc = positions[idx][0], positions[idx][1]
            ax[loc].semilogx(abs(current_density), potential, color="k", label=f"pH : {file[2:6]}")
            for _ax in [ax2, ax_tafel]:
                _ax[loc].semilogx(
                    abs(current_density_filtered),
                    potential_filtered,
                    color="k",
                    label=f"pH : {file[2:6]}",
                )

            ax_tafel[loc].semilogx(
                10**i_tafel_abs_log,
                tafel_slope * 10**i_tafel_abs_log + intercept,
                color="k",
                label=f"pH : {file[2:6]}, slope = {tafel_slope}, R2 = {r_value**2}",
            )

            ax[loc].legend()
            ax2[loc].legend()
            ax_tafel[loc].legend()

            if (idx + 1) % 4 == 0:
                fig.savefig(f"{save_figs_raw_data}/plots_of_raw_data_{idx+1}")
                fig2.savefig(f"{save_figs_filtered_data}/plots_of_filtered_data_{idx+1}")
                fig_tafel.savefig(f"tafel_slopes_figures/{idx+1}")
                for subplot_ax, subplot_ax2 in zip(ax.flat, ax2.flat):
                    subplot_ax.clear()
                    subplot_ax2.clear()

    # df_merged_filtered.to_csv("df_merged_not_normalized.txt", sep="\t", index=False)

    # save selected featurs
    selected_features_df.to_csv("selected_features.csv", sep="\t", index=False)
    return df_merged_filtered


if __name__ == "__main__":
    t0 = time.perf_counter()
    plot_and_return_dataframe_with_filtered_data()
    runtime = time.perf_counter() - t0
    print(f"Runtime: {runtime} s")
