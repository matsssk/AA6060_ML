from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from src.data_preprocessing import return_training_data_X_y, normalize_data_for_ANN
import pickle
import keras_tuner
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, Callback
from keras_tuner.tuners import RandomSearch
from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers
import pandas as pd
import time
import os
from keras_tuner.engine.hyperparameters import HyperParameters

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


PATH_HYPERPARAM_FOLDER = "hyperparams_random_search_results"


def search_spaces() -> list[list | float]:
    neurons_space = [i for i in range(20, 160)[::10]]
    num_layers_space = [i for i in range(1, 5)]
    l2s_space = [1e-3, 1e-2, 1e-1]
    optimizers_space = ["adam", "sgd", "adagrad"]
    loss_funcs_space = ["mse", "mae", "msle"]
    activation_func_space = ["relu", "tanh", "sigmoid"]
    output_act_func_space = ["softmax", "linear"]
    batch_size_space = [16, 32, 64, 128]

    combinations = (
        len(neurons_space)
        * len(num_layers_space)
        * len(l2s_space)
        * len(optimizers_space)
        * len(loss_funcs_space)
        * len(activation_func_space)
        * len(output_act_func_space)
    )

    return [
        neurons_space,
        num_layers_space,
        l2s_space,
        optimizers_space,
        loss_funcs_space,
        activation_func_space,
        output_act_func_space,
        batch_size_space,
        combinations,
    ]


def ANN_model(hp: HyperParameters):
    # call search spaces
    (
        neurons_space,
        num_layers_space,
        l2s_space,
        optimizers_space,
        loss_funcs_space,
        activation_func_space,
        output_act_func_space,
        batch_size_space,
        combinations,
    ) = search_spaces()

    neurons = hp.Choice("neurons", values=neurons_space)
    num_layers = hp.Choice("num_layers", values=num_layers_space)
    l2s = hp.Choice("l2", values=l2s_space)
    optimizers = hp.Choice("optimizer", values=optimizers_space)
    loss_funcs = hp.Choice("loss", values=loss_funcs_space)
    activation_func = hp.Choice("activation", values=activation_func_space)
    output_act_func = hp.Choice("output_activation", values=output_act_func_space)
    batch_size = hp.Choice("batch_size", values=batch_size_space)
    print(f"The number of hyperparameter combinations: {combinations}")

    # Define the loss function as I want loss function and metrics to use same
    # otherwise the randomizer will draw from loss_func twice for each variables (loss and metrics)
    if loss_funcs == "mse":
        loss = "mse"
    elif loss_funcs == "mae":
        loss = "mae"
    else:
        loss = "msle"

    model = Sequential()
    # depending on num_layers, build
    for n_randomly_drawn_layers in range(num_layers):
        if n_randomly_drawn_layers == 0:
            # first hidden layer
            # input shape = 2 as we have two variables (potential, pH)
            model.add(
                Dense(
                    neurons,
                    input_shape=(2,),
                    activation=activation_func,
                    kernel_regularizer=regularizers.l2(l2=l2s),
                )
            )
        else:
            # additional hidden layers
            model.add(
                Dense(
                    neurons,
                    activation=activation_func,
                    kernel_regularizer=regularizers.l2(l2=l2s),
                )
            )
    # 1 output, current density
    model.add(Dense(1, activation=output_act_func))
    model.compile(optimizer=optimizers, loss=loss, metrics=loss)

    return model


def create_tuner_and_return_results() -> list[list]:
    X_train, X_val, y_train, y_val, _, _ = normalize_data_for_ANN()
    max_trials = 2
    tuner = keras_tuner.RandomSearch(
        ANN_model,
        objective="val_loss",
        max_trials=max_trials,
        overwrite=True,
        directory="tuning_results",
        project_name="ANN",
    )

    tuner.search(
        X_train, y_train, validation_data=(X_val, y_val), epochs=2, callbacks=[EarlyStopping("val_loss", patience=3)]
    )
    tuner.results_summary()

    best_hps: list = tuner.get_best_hyperparameters(num_trials=max_trials)
    best_models: list = tuner.get_best_models(num_models=max_trials)

    return [best_hps, best_models]


def store_tuning_results() -> pd.DataFrame:
    """
    best_hps and best_models are lists of keras objects
    <class 'keras_tuner.engine.hyperparameters.hyperparameters.HyperParameters'>
    <class 'keras.engine.sequential.Sequential'>
    """
    best_hps, best_models = create_tuner_and_return_results()
    _, X_val, _, y_val, _, _ = normalize_data_for_ANN()
    results = []
    print(best_hps[0].values)
    for hp_element, best_model in zip(best_hps, best_models):
        results_dict = {}
        val_loss, _ = best_model.evaluate(X_val, y_val)  # tuple of loss and metric (equal)
        results_dict["val_loss"] = val_loss

        for hp_name in hp_element.values:
            results_dict[hp_name] = hp_element.get(hp_name)

        results.append(results_dict)

    results = pd.DataFrame(results)
    results.insert(0, "best_models_sorted", [i for i in range(1, len(best_models) + 1)])
    results.to_csv("tuning_results/df_with_results", sep=",", index=False)
    return results


def format_time(seconds):
    # Calculate the number of days, hours, and minutes
    days = seconds // (24 * 3600)
    seconds %= 24 * 3600
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60

    # Build the formatted time string
    return f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes"


if __name__ == "__main__":
    t0 = time.perf_counter()
    store_tuning_results()
    elapsed_time = time.perf_counter() - t0
    time_str = format_time(elapsed_time)
    print(time_str)

# # time expensive. will only tune ANN
# def catboost_tuning():
#     # search space. Ranges completed
#     params = {
#         # # is Default
#         "iterations": np.arange(200, 1200, 50),  # 1000
#         "depth": [5, 6, 7, 15, 16, 17],  # 6 for not LossGuide, 16 for LossGuide
#         "l2_leaf_reg": [2.5, 3.0, 3.5],  # 3.0
#         "random_strength": [0.8, 1.0, 1.2],  # 1.0
#         "bagging_temperature": [0, 1.0, 2.0],  # 1.0
#         "border_count": [128, 254],
#         "grow_policy": ["SymmetricTree", "Lossguide", "Depthwise"],  # SymmetricTree
#         "min_data_in_leaf": [1, 2, 3],  # 1
#         "max_leaves": [26, 31, 36],  # 31
#     }
#     # test 50% of possible hyperparam in search space
#     n_iter = 0.5 * np.prod([len(v) for k, v in params.items()])
#     early_stopping = 10

#     # initialize model
#     model = CatBoostRegressor()

#     # Initialize RSCV
#     random_search = RandomizedSearchCV(model, param_distributions=params, cv=3, n_iter=n_iter, n_jobs=-1)

#     X, y = return_training_data_X_y()
#     # fit model
#     random_search.fit(X, y, verbose=0, early_stopping_rounds=early_stopping)
#     best_params = random_search.best_params_

#     with open(f"{PATH_HYPERPARAM_FOLDER}/catboost.json", "wb") as f:
#         pickle.dump(best_params, f)

# def extract_hyperparams_from_json(model: str) -> dict:
#     """
#     Returns a dictionairy with all the hyperparams optimized through RandomSearchCV
#     in hyperparameter_tuning.py
#     """
#     with open(f"{PATH_HYPERPARAM_FOLDER}/{model}.json", "rb") as f:
#         # Use pickle.load to load the dictionary from the file
#         best_params: dict = pickle.load(f)
#     return best_params
