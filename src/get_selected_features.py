import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats


def _get_ocps(E: np.ndarray, E_filtered: np.ndarray, i_filtered: np.ndarray) -> list[float]:
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


def slope(i0, i1, E0, E1):
    return (E1 - E0) / (np.log(abs(i1)) - np.log(abs(i0)))


# import warnings

# warnings.filterwarnings("ignore", category=RuntimeWarning)


# def cathodic_tafel_slope(ocp_t_half: float, E: np.ndarray, i: np.ndarray) -> list[float]:
#     # create a new array with the relevant area for a tafel region
#     # remove all potentials that are above ocp and below a treshold defined as 0.2 lower than ocp
#     mask = (E > (ocp_t_half - 0.02)) & (E <= ocp_t_half)
#     E, i = E[mask], i[mask]
#     # highest E is first in the array
#     index_sort = np.argsort(E)[::-1]
#     # use these indices to create new arrays, get every 100 value and use these as slope points
#     E_sort, i_sort = E[index_sort], i[index_sort]
#     # tafel slope is change in potental for each unit log(abs(i))
#     tafel_slopes = []
#     w = 50
#     # for idx, (E, i) in enumerate(zip(E_sort, i_sort), start=1):
#     #     if idx == len(E_sort):
#     #         break
#     # perform linreg on every n points
#     # tafel_slopes.append(slope(i, i_sort[idx], E, E_sort[idx]))

#     for n in range(0, int(len(E_sort) / w)):
#         # perform linreg on w points
#         try:
#             i = np.log(abs(i_sort[(w * n) : (w + w * n)]))
#             tafel_slope, _, _, _, _ = stats.linregress(i, E_sort[(w * n) : (w + w * n)])
#         except IndexError:
#             continue
#         tafel_slopes.append(tafel_slope)

#     # boolean mask that gives true for non-infinite slopes
#     mask = [(ts != math.inf) and (ts != -math.inf) for ts in tafel_slopes]
#     tafel_slopes = np.array(tafel_slopes)[mask]  # to use the mask
#     avg_slope = np.mean(tafel_slopes)

#     slope_diffs = np.abs(tafel_slopes - avg_slope)
#     # get the index of the minimum difference
#     min_idx = np.argmin(slope_diffs)
#     # get the potentials corresponding to the two points used to make the closest tafel slope
#     E1, E2 = E_sort[min_idx], E_sort[min_idx + 1]
#     i1, i2 = i_sort[min_idx], i_sort[min_idx + 1]
#     b = E1 - avg_slope * np.log(abs(i1))

#     return [avg_slope, E1, E2, i1, i2, b]
