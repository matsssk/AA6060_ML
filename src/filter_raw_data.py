import numpy as np


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
