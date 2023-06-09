import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy import stats
from typing import Optional


def get_ocps(E: np.ndarray, E_filtered: np.ndarray, i_filtered: np.ndarray) -> list[float]:
    """
    Returns: ocp at t= 0, ocp at t = 2/3*t_tot (before anodic scan) and delta ocp

    param E: np.ndarray, potentials. raw data
    param E_filtered: np.ndarray, filtered curves
    param i_filtered: np.ndarray, filtered curves
    """
    E_ocp_t0 = E[0]
    E_ocp_t_half = E_filtered[np.argmin(abs(i_filtered))]
    delta_ocp = E_ocp_t0 - E_ocp_t_half
    return [E_ocp_t0, E_ocp_t_half, delta_ocp]


def check_ORR_dominance(ocp_t_half: float) -> bool:
    return ocp_t_half > -1.0500


def linreg_tafel_line_ORR_or_HER(
    ocp_t_half: float, E_filtered: np.ndarray, i_filtered: np.ndarray, ph
) -> list[np.ndarray | float]:
    """
    Apply linear regression on ORR or HER dominant region at the cathodic branch
    ORR is dominant for higher potentials than roughly -1.05, depending on ph
    The upper potential limit is set to - 0.15 V vs OCP (not at t= 0s)

    For high pH, HER is dominant at low potentials, seen by less steep curve
    """
    if ph == 6.6:
        if check_ORR_dominance(ocp_t_half):
            mask = (E_filtered < (ocp_t_half - 0.15)) & (E_filtered > -1.1)
        else:
            mask = E_filtered < (ocp_t_half - 0.15)
        E_applied, i_applied = np.flip(E_filtered[mask]), np.flip(i_filtered[mask])
        # apply linear regression on this region
        # remember that we want a linear region only for logarithmic x axis
        i_applied_log_abs = np.log10(abs(i_applied))
        result = stats.linregress(i_applied_log_abs, E_applied)
        slope, intercept, rvalue, pvalue, std_error_slope, intercept_stderr = (
            result.slope,
            result.intercept,
            result.rvalue,
            result.pvalue,
            result.stderr,
            result.intercept_stderr,
        )
        residual = E_applied - (intercept + slope * i_applied_log_abs)
        # check if we have ORR or HER
    # create a boolean mask to filter potential and current to the desired region
    # where kinetics are defined by the tafel equation
    # take not that for lower potentials on ORR, we have diffusion limitations
    else:
        if check_ORR_dominance(ocp_t_half):
            mask = (E_filtered < (ocp_t_half - 0.15)) & (E_filtered > -1.05)
        else:
            mask = E_filtered < (ocp_t_half - 0.15)
        E_applied, i_applied = np.flip(E_filtered[mask]), np.flip(i_filtered[mask])
        # apply linear regression on this region
        # remember that we want a linear region only for logarithmic x axis
        i_applied_log_abs = np.log10(abs(i_applied))
        result = stats.linregress(i_applied_log_abs, E_applied)
        slope, intercept, rvalue, pvalue, std_error_slope, intercept_stderr = (
            result.slope,
            result.intercept,
            result.rvalue,
            result.pvalue,
            result.stderr,
            result.intercept_stderr,
        )
        residual = E_applied - (intercept + slope * i_applied_log_abs)

    return [E_applied, i_applied_log_abs, slope, intercept, rvalue, std_error_slope, intercept_stderr, residual]


def lower_upper_confidence_interval_slope(
    i_applied_log_abs,
    slope,
    slope_std_error: float,
    intercept,
    intercept_std_error: float,
    confidence_level: Optional[float] = 0.99,
) -> list[float]:
    pred = slope * i_applied_log_abs + intercept
    z = norm.ppf((1 + confidence_level) / 2)
    # Calculate the confidence interval for the slope and intercept
    slope_ci = z * slope_std_error
    lower_slope = slope - z * slope_std_error
    upper_slope = slope + z * slope_std_error

    intercept_ci = z * intercept_std_error
    lower_intercept = intercept - z * intercept_std_error
    upper_intercept = intercept + z * intercept_std_error

    lower_ci = pred - slope_ci - intercept_ci
    upper_ci = pred + slope_ci + intercept_ci

    return [lower_ci, upper_ci, lower_slope, upper_slope, lower_intercept, upper_intercept]


def plot_confidence_interval(
    i_applied_log_abs,
    slope,
    slope_std_error: float,
    intercept,
    intercept_std_error: float,
    ax,
    color: str,
    model: str,
    confidence_level: Optional[float] = 0.99,
):
    """
    Returns the plot for a 95% confidence interval

    Args:
        confidence level should be given as frac of 1
    """
    lower_ci, upper_ci, _, _, _, _ = lower_upper_confidence_interval_slope(
        i_applied_log_abs, slope, slope_std_error, intercept, intercept_std_error
    )

    ax.fill_between(
        10**i_applied_log_abs,
        upper_ci,
        lower_ci,
        alpha=0.2,
        color=color,
        label=f"{model}: {int(confidence_level*100)}% CI Tafel",
    )


def get_ocps_machine_learning_models(E, i):
    return E[np.argmin(abs(i))]


import os

# def _confidence_interval_i_corr():
#     # read csv file to get data
#     csv_files = os.listdir("summarized_data_figures_datafiles/csv_files")
#     csv_files_df_features = [feature_file for feature_file in csv_files if feature_file.startswith("df_features")]

#     for file in csv_files_df_features:
#         # E = slope * i0 + intercept   , where i0 is log(|i|)
#         # -> i0 = (E - intercept) / slope
#         # -> i_corr = (E_corr - intercept) / slope
#         pass


def lower_upper_i_corr(intercept_lower, intercept_upper, slope_lower, slope_upper, E_corr) -> list[float]:
    """Calculated the 99% confidence interval of i_corr
    return i_corr on normal form"""

    i_corr_lower_log = (E_corr - intercept_lower) / slope_lower
    i_corr_upper_log = (E_corr - intercept_upper) / slope_upper
    return [10**i_corr_lower_log, 10**i_corr_upper_log]


if __name__ == "__main__":
    # _confidence_interval_i_corr()
    pass
