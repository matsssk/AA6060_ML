import numpy as np


# def remove_outliers(
#     current_density_removed_first_branch: np.ndarray,
#     potential_removed_first_branch: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray]:
#     for idx, current in enumerate(current_density_removed_first_branch):
#         if abs(current - current_density_removed_first_branch[idx - 1]) > 20 * 10**-6:
#             current = np.nan
#             potential_removed_first_branch[idx] = np.nan

#     valid_mask = ~(np.isnan(current_density_removed_first_branch) | np.isnan(potential_removed_first_branch))

#     resulting_current = current_density_removed_first_branch[valid_mask]
#     resulting_potential = potential_removed_first_branch[valid_mask]
#     return resulting_potential, resulting_current


def remove_first_cath_branch(current_density: np.ndarray, potential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns tuple of filtered np.ndarray, potential and current density
    where the first cathodic scan is removed, from t = 0 until it reverses
    at the potential scan limit

    :param potential: np.ndarray of the potential
    :param current_density: np.ndarray of current density
    :return: tuple of potential and current density
    """
    # remove first part until minima is reached
    minima_index = np.argmin(potential)
    potential, current_density = (
        potential[minima_index:],
        current_density[minima_index:],
    )
    return current_density, potential


# def final_filtering(potential: np.ndarray, current_density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     potential_removed_first, current_density_removed_first = remove_first_cath_branch(potential, current_density)
#     # potential_outliers_removed, current_density_outliers_removed = remove_outliers(
#     #     current_density_removed_first, potential_removed_first
#     # )
#     return potential_outliers_removed, current_density_outliers_removed
