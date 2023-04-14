import numpy as np
import os
from io import StringIO


def _diameter():
    return 1.3  # cm


def _area_WE():
    return np.pi * (_diameter() / 2) ** 2  # cm^2


def list_of_filenames(folder: str = "raw_data") -> list[str]:
    """
    Returns a list with all filenames (str) in directory name

    :param folder: the folder with raw data (default is 'raw_data')
    :raises: FileNotFoundError if the directory cannot be found
    """
    try:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    except FileNotFoundError:
        raise FileNotFoundError(f"Folder {folder} not found")
    return files


def load_raw_data(file_path: str, area=_area_WE()) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns arrays with potential and current density from file path

    :param file_path: file path of the folder to load data from
    :param AREA_WORKING_ELECTRODE: area of the exposed aluminium alloy
    """
    with open(file_path, "r", encoding="ISO-8859-1") as f:
        data = f.read().replace(",", ".")
    data = StringIO(data)
    try:
        df = np.loadtxt(data, skiprows=51, usecols=(2, 3))
    #  different format because of different potentiostat
    except ValueError:
        df = np.loadtxt(data, skiprows=60, usecols=(2, 3))

    potential, current_density = df[:, 0], df[:, 1] / area
    return (potential, current_density)
