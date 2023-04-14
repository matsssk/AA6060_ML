from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from src.data_preprocessing import return_training_data_X_y
import pickle
import keras_tuner
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers


def extract_hyperparams_from_json(model: str) -> dict:
    """
    Returns a dictionairy with all the hyperparams optimized through RandomSearchCV
    in hyperparameter_tuning.py
    """
    with open(f"{PATH_HYPERPARAM_FOLDER}/{model}.json", "rb") as f:
        # Use pickle.load to load the dictionary from the file
        best_params: dict = pickle.load(f)
    return best_params


PATH_HYPERPARAM_FOLDER = "hyperparams_random_search_results"


# time expensive. will only tune ANN
def catboost_tuning():
    # search space. Ranges completed
    params = {
        # # is Default
        "iterations": np.arange(200, 1200, 50),  # 1000
        "depth": [5, 6, 7, 15, 16, 17],  # 6 for not LossGuide, 16 for LossGuide
        "l2_leaf_reg": [2.5, 3.0, 3.5],  # 3.0
        "random_strength": [0.8, 1.0, 1.2],  # 1.0
        "bagging_temperature": [0, 1.0, 2.0],  # 1.0
        "border_count": [128, 254],
        "grow_policy": ["SymmetricTree", "Lossguide", "Depthwise"],  # SymmetricTree
        "min_data_in_leaf": [1, 2, 3],  # 1
        "max_leaves": [26, 31, 36],  # 31
    }
    # test 50% of possible hyperparam in search space
    n_iter = 0.5 * np.prod([len(v) for k, v in params.items()])
    early_stopping = 10

    # initialize model
    model = CatBoostRegressor()

    # Initialize RSCV
    random_search = RandomizedSearchCV(model, param_distributions=params, cv=3, n_iter=n_iter, n_jobs=-1)

    X, y = return_training_data_X_y()
    # fit model
    random_search.fit(X, y, verbose=0, early_stopping_rounds=early_stopping)
    best_params = random_search.best_params_

    with open(f"{PATH_HYPERPARAM_FOLDER}/catboost.json", "wb") as f:
        pickle.dump(best_params, f)


def ANN_model(hp):
    # define search space for hyperparams
    neurons = hp.Int("neurons", min_value=20, max_value=150, step=10)
    optimizers = hp.Choice("optimizer", values=["adam", "sgd", "adagrad", "adamw"])

    model = Sequential()
    # input has two neurons since we have two features (E and pH), unknown data points

    model.add(Dense(neurons, input_shape=(2,), activation=activation, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(neurons, activation=activation, kernel_regularizer=regularizers.l2(0.01)))
    # 1 output, current density
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=optimizers, loss=loss, metrics=loss)


def ANN_tuning():
    batch_size = hp.Choice("batch", values=[16, 32, 64])
    a = 3


if __name__ == "__main__":
    # catboost_tuning()
    # print(extract_hyperparams_from_json("catboost"))
    pass
