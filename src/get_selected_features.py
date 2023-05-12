import numpy as np
from scipy import stats


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
    ocp_t_half: float, E_filtered: np.ndarray, i_filtered: np.ndarray
) -> list[np.ndarray | float]:
    """
    Apply linear regression on ORR or HER dominant region at the cathodic branch
    ORR is dominant for higher potentials than roughly -1.05, depending on ph
    The upper potential limit is set to - 0.15 V vs OCP (not at t= 0s)

    For high pH, HER is dominant at low potentials, seen by less steep curve
    """

    # check if we have ORR or HER
    # create a boolean mask to filter potential and current to the desired region
    # where kinetics are defined by the tafel equation
    # take not that for lower potentials on ORR, we have diffusion limitations

    if check_ORR_dominance(ocp_t_half):
        mask = (E_filtered < (ocp_t_half - 0.15)) & (E_filtered > -1.05)
    else:
        mask = E_filtered < (ocp_t_half - 0.15)
    E_applied, i_applied = np.flip(E_filtered[mask]), np.flip(i_filtered[mask])
    # apply linear regression on this region
    # remember that we want a linear region only for logarithmic x axis
    i_applied_log_abs = np.log10(abs(i_applied))
    slope, intercept, r_value, _, std_err_slope = stats.linregress(i_applied_log_abs, E_applied)
    return [E_applied, i_applied_log_abs, slope, intercept, r_value, std_err_slope]


def get_ocps_machine_learning_models(E, i):
    return E[np.argmin(abs(i))]
