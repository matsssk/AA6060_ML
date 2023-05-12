import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from plot_raw_data import plot_and_return_dataframe_with_filtered_data
from nptyping import NDArray, Float, Shape
from typing import TypeVar
from sklearn.model_selection import train_test_split
import random

# N_ROWS can be any length
N_ROWS = TypeVar("N_ROWS")


# define func that loads all NOT normalized data (for DT models)
# shape is 3 columns, N_ROWS rows
def all_filtered_experimental_data_not_normalized() -> NDArray[Shape["N_ROWS, 3"], Float]:
    df = pd.read_csv("df_training_data.csv", sep="\t")
    return df.values  # not normalized


def convert_current_to_log(current: NDArray[Shape["1, N_ROWS"], Float]) -> NDArray[Shape["1, N_ROWS"], Float]:
    return np.log10(abs(current))


def ph_for_testing() -> None:
    """
    Saves dataframe with randomly drawn pHs
    PS: Uniform probabilties
    """
    samples = np.array_split(np.arange(2.0, 12.2, 0.2), 4)
    test_phs = []
    for sample in samples:
        test_phs.append(round((random.choice(sample)), 3))

    # sample1, sample2, sample3, sample4 = np.array_split(samples, 4)
    # print(sample1, sample2, sample3, sample4)
    # test_ph = []
    # for
    # while len(test_ph) < 4:
    #     ph = round(random.choice(samples), 3)
    #     if ph not in test_ph:
    #         test_ph.append(ph)
    df = pd.DataFrame(test_phs, columns=["test_pHs"])
    print(df)
    df.to_csv("testing_pHs.csv", sep="\t", index=False)


def split_data_into_training_and_testing(
    all_data: NDArray[Shape["N_ROWS, 3"], Float]
) -> tuple[NDArray[Shape["N_ROWS, 3"], Float], NDArray[Shape["N_ROWS, 3"], Float]]:
    """ "
    Splits data into training and testing

    returns: (n,3) shaped numpy arrays with training and testing
    :param all_data: all available data from laboratory, sorted into a (n,3) numpy array
    """

    # create boolean mask with the pH for testing
    # pH is second column
    ph_for_testing = pd.read_csv("testing_pHs.csv", sep="\t")
    mask = np.isin(all_data[:, 1], ph_for_testing["test_pHs"])

    # Split data into training and testing arrays
    testing_data = all_data[mask]
    training_data = all_data[~mask]
    return (training_data, testing_data)


def return_training_data_X_y() -> tuple[NDArray[Shape["N_ROWS, 2"], Float], NDArray[Shape["1, N_ROWS"], Float]]:
    """
    Returns training data X with shape (N,2) and y with shape (N,1)
    """

    all_data = all_filtered_experimental_data_not_normalized()
    training_data = split_data_into_training_and_testing(all_data)[0]

    X = training_data[:, :2]
    # convert current to log10, [:, -1] automatically reshapes from (N,1) to (N,) which
    # is what we want to be the shape of our output as models only accept (N,) for y
    y = convert_current_to_log(training_data[:, -1])

    return (X, y)


def split_into_training_and_validation(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """
    Takes in training data and returns X_train, X_val, y_train, y_val
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def normalize_data_for_ANN():
    X, y = return_training_data_X_y()
    y = y.reshape(-1, 1)
    # Normalize the data using MinMaxScaler
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = x_scaler.fit_transform(X)
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = y_scaler.fit_transform(y)
    X_train, X_val, y_train, y_val = split_into_training_and_validation(X_scaled, y_scaled)

    return X_train, X_val, y_train, y_val, x_scaler, y_scaler


if __name__ == "__main__":
    # ph_for_testing()
    pass
