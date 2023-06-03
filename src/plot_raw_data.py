import os
import time
import matplotlib.pyplot as plt
from src.load_data import list_of_filenames, load_raw_data
import matplotlib

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
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

import pandas as pd
from src.filter_raw_data import remove_first_cath_branch
from src.get_selected_features import get_ocps, linreg_tafel_line_ORR_or_HER


def get_grid_for_axs(ax):
    for i in range(2):
        for j in range(2):
            ax[i, j].grid()


def plot_and_return_dataframe_with_filtered_data(
    folder_raw: str = "raw_data",
    save_figs_raw_data: str = "raw_data_plots",
    save_figs_filtered_data: str = "filtered_raw_data_plots",
):
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
        fig_raw, ax_raw = plt.subplots(2, 2)
        fig_raw.supxlabel(r"Absolute value of current density ($|\mathit{i}|$) [A/cm$^2$]")
        fig_raw.supylabel("Potential ($E$) vs SCE [V]")
        get_grid_for_axs(ax_raw)

        # define figure to plot filtered data in
        fig2_filt, ax2_filt = plt.subplots(2, 2)
        fig2_filt.supxlabel(r"Absolute value of current density ($|\mathit{i}|$) [A/cm$^2$]")
        fig2_filt.supylabel("Potential ($E$) vs SCE [V]")
        get_grid_for_axs(ax2_filt)

        # figure to plot raw data and filtered data in
        fig_compare_raw_filt, ax_compare = plt.subplots(2, 2)
        fig_compare_raw_filt.supxlabel(r"Absolute value of current density ($|\mathit{i}|$) [A/cm$^2$]")
        fig_compare_raw_filt.supylabel("Potential ($E$) vs SCE [V]")
        get_grid_for_axs(ax_compare)

        # figure to plot tafel slopes
        fig_tafel, ax_tafel = plt.subplots(2, 2)
        fig_tafel.supxlabel(r"Absolute value of current density ($|\mathit{i}|$) [A/cm$^2$]")
        fig_tafel.supylabel("Potential ($E$) vs SCE [V]")
        get_grid_for_axs(ax_tafel)

    # create empty dataframe and add data to it from all files
    df_all_filtered_data = pd.DataFrame()
    files = list_of_filenames("raw_data")
    files_without_gamry_errors = list_of_filenames("raw_data_without_gamry_noise")
    # print(files_without_gamry_errors)
    # dataframe to store selected features
    selected_features_df = pd.DataFrame()
    # loop over all files and plot necessary plots for report, while storing data to dataframe

    for idx, (file_raw, file_no_gamry_noise) in enumerate(zip(files, files_without_gamry_errors)):
        file_path_raw: str = os.path.join(folder_raw, file_raw)
        pH: float = float(file_raw.split("h")[1].split(",")[0] + "." + file_raw.split(",")[1].split(".")[0])

        potential_raw, current_density_raw = load_raw_data(file_path_raw)

        # filtered data without gamry noise
        potential_no_gamry_noise, current_density_no_gamry_noise = load_raw_data(
            os.path.join("raw_data_without_gamry_noise", file_no_gamry_noise)
        )
        current_density_filtered, potential_filtered = remove_first_cath_branch(
            current_density_no_gamry_noise, potential_no_gamry_noise
        )
        # if abs(len(potential_raw) - len(potential_filtered)) < 1000:
        #     print(pH)
        # if pH == 8.4:
        #     print(len(potential_filtered))
        #     print(len(potential_raw))

        ocp_t0, ocp_t_half, delta_ocp = get_ocps(potential_raw, potential_filtered, current_density_filtered)
        # plot pH vs OCP

        try:
            # ORR or HER tafel slopes
            (
                E_applied,
                i_applied_log_abs,
                tafel_slope,
                intercept,
                r_value,
                std_err_slope,
                intercept_stderr,
                _,
            ) = linreg_tafel_line_ORR_or_HER(ocp_t_half, potential_filtered, current_density_filtered)
            # slope is delta E / delta abs(i), we need delta E / delta log10(r"$\lvert i \rvert$")
        except Exception as e:
            raise e

        df_add_to_selected_features = pd.DataFrame(
            {
                "pH": [pH],  # pandas does not accept float, only numpy float (e.g. ocp_t0)
                "OCP_t0": ocp_t0,
                "OCP_t_half": ocp_t_half,
                "delta_OCP": delta_ocp,
                "tafel_slope [mV/dec]": tafel_slope * 1000,
                "intercept [V]": intercept,
                "r_value_squared": r_value**2,
                "standard error tafel": std_err_slope,
            }
        )
        selected_features_df = pd.concat([selected_features_df, df_add_to_selected_features])
        # store selected features, E_corr, i_corr, tafel, etc...

        #  create a pandas DataFrame with the output from the filtering process
        #  for pH, the name format "ph2,0" etc. is necessary, otherwise the indices will
        #  be incorrect, [2:5] means e.g. the letters "2,0"
        df_to_add_to_df_all_filtered_data = pd.DataFrame(
            {
                "Potential [V]": potential_filtered,
                "pH": [pH] * len(potential_filtered),
                "Current density [A/cm$^2$]": current_density_filtered,
            }
        )
        #  update the existing dataframe, adding the rows
        df_all_filtered_data = pd.concat(
            [
                df_all_filtered_data,
                df_to_add_to_df_all_filtered_data,
            ]
        )
        # the coordinates of the different subplots
        positions = [[0, 0], [0, 1], [1, 0], [1, 1]] * int(len(files) / 4) * 2
        if __name__ == "__main__":
            loc = positions[idx][0], positions[idx][1]

            # plot raw data
            ax_raw[loc].semilogx(abs(current_density_raw), potential_raw, color="k", label=f"pH = {pH}")

            # plot filtered data in filtered data figure and tafel figure
            for _ax in [ax2_filt, ax_tafel]:
                _ax[loc].semilogx(
                    abs(current_density_filtered),
                    potential_filtered,
                    color="k",
                    label=f"pH = {pH}",
                )

            # plot Tafel lines in Tafel figure
            # partial_derivative = r"$\frac{\partial E}{\partial \log r}$|$i|$ [A/cm$^2$]"
            # rounded_r_squared = round(r_value**2, 5)  # type: ignore
            # ax_tafel[loc].semilogx(
            #     10**i_applied_log_abs,
            #     tafel_slope * i_applied_log_abs + intercept,
            #     color="r",
            #     linestyle="--",
            #     label=f"pH = {pH}, {partial_derivative} = {int(tafel_slope*1000)} mV/dec, R\u00b2 = {rounded_r_squared}",
            # )

            # plot both raw data and filtered data in figure for comparing raw data and filtered data
            # raw data is first column (positions 0,0 and 1,0)
            # filtered data is to the left

            if idx % 2 == 0 and (idx + 1) != len(files):
                ax_compare[0, 0].semilogx(abs(current_density_raw), potential_raw, color="k", label=f"pH = {pH}")
                ax_compare[0, 1].semilogx(
                    abs(current_density_filtered), potential_filtered, color="k", label=f"pH = {pH}, filtered"
                )
                ax_compare[0, 1].legend()
                ax_compare[0, 0].legend()
            if idx % 2 == 1 and (idx + 1) != len(files):
                ax_compare[1, 0].semilogx(abs(current_density_raw), potential_raw, color="k", label=f"pH = {pH}")
                ax_compare[1, 1].semilogx(
                    abs(current_density_filtered), potential_filtered, color="k", label=f"pH = {pH}, filtered"
                )
                ax_compare[1, 1].legend()
                ax_compare[1, 0].legend()

                # save end clear fig

                fig_compare_raw_filt.tight_layout()
                for ftype in ["pgf", "pdf"]:
                    fig_compare_raw_filt.savefig(f"raw_data_vs_filtered_data/{idx+1}.{ftype}")
                for subplot_ax in ax_compare.flat:
                    subplot_ax.clear()
                get_grid_for_axs(ax_compare)

            if (idx + 1) == len(files):
                # figure to plot ph 12.0 as we do not have len(files) % 4 = 0
                fig_ph12, ax_ph12 = plt.subplots(1, 2, figsize=(5 * 1.11317 / 0.85, 2.306 / 0.85))  # fitted to latex
                ax_ph12[0].grid()
                ax_ph12[1].grid()
                fig_ph12.supxlabel(r"Absolute value of current density ($|\mathit{i}|$) [A/cm$^2$]")
                fig_ph12.supylabel("Potential ($E$) vs SCE [V]")
                ax_ph12[0].semilogx(abs(current_density_raw), potential_raw, color="k", label=f"pH = {pH}")
                ax_ph12[1].semilogx(
                    abs(current_density_filtered), potential_filtered, color="k", label=f"pH = {pH}, filtered"
                )
                ax_ph12[0].legend()
                ax_ph12[1].legend()
                fig_ph12.tight_layout()
                for ftype in ["pgf", "pdf"]:
                    fig_ph12.savefig(f"raw_data_vs_filtered_data/{idx+2}.{ftype}")

            ax_raw[loc].legend()
            ax2_filt[loc].legend()
            # ax_tafel[loc].legend()

            if (idx + 1) % 4 == 0 or (idx + 1) == len(files):
                # delete the empty slot in the last figure
                if idx + 1 == len(files) and (idx + 1) % 4 != 0:
                    for axs in [ax_raw, ax2_filt, ax_tafel]:
                        axs[1, 1].remove()

                fig_raw.tight_layout()
                fig2_filt.tight_layout()
                # fig_tafel.tight_layout()

                for ftype in ["pgf", "pdf"]:
                    fig_raw.savefig(f"{save_figs_raw_data}/plots_of_raw_data_{idx+1}.{ftype}")
                    fig2_filt.savefig(f"{save_figs_filtered_data}/plots_of_filtered_data_{idx+1}.{ftype}")
                    # fig_tafel.savefig(f"tafel_slopes_figures/{idx+1}.{ftype}")

                for subplot_ax, subplot_ax2, subplot_ax_tafel in zip(ax_raw.flat, ax2_filt.flat, ax_tafel.flat):
                    if (idx + 1) != len(files):
                        subplot_ax.clear()
                        subplot_ax2.clear()
                        subplot_ax_tafel.clear()

    df_all_filtered_data.to_csv("df_training_data.csv", sep="\t", index=False)
    # save selected featurs
    selected_features_df.to_csv("selected_features.csv", sep="\t", index=False)


if __name__ == "__main__":
    t0 = time.perf_counter()
    plot_and_return_dataframe_with_filtered_data()
    runtime = time.perf_counter() - t0
    print(f"Runtime: {runtime} s")
