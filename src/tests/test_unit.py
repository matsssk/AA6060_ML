import numpy as np

from filter_raw_data import remove_first_cath_branch


def test_remove_first_cath_branch(load_data: np.ndarray):
    """
    This tests the filtering process where the cathodic part before reaching
    potential minima is removed correctly. Checks if the first index in the
    potential array after applied filter is the same as the minima potential
    in the originl potential array. It tests for ph2,0.DTA in raw_data folder
    where load_data is a fixture given in conftest.py
    """

    current_density, potential = load_data[:, 0], load_data[:, 1]

    # checks if first potential index in remove_first_cath_branch is the minima in test_data
    assert remove_first_cath_branch(potential, current_density)[0][0] == np.min(potential)
