import os
from src.load_data import load_raw_data


def folder_with_ocps() -> str:
    return "ocp_raw_data"


def plot_ocps():
    files: list[str] = load_raw_data(folder_with_ocps())
    print(files)


plot_ocps()
