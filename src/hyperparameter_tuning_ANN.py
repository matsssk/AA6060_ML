from src.data_preprocessing import normalize_data_for_ANN
import keras_tuner
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers
import pandas as pd
import time
import os
from keras_tuner.engine.hyperparameters import HyperParameters

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def path_hyperparam_folder():
    return "hyperparams_random_search_results"


def directory_for_tuning_results():
    return "tuning_results_ANN"


def name_df_hyperparams_results():
    return "df_with_results_from_tunersearch.csv"


def epochs_for_search_and_train():
    return 80


def trials():
    return 25


def early_stopping_callback() -> list[EarlyStopping]:
    return [EarlyStopping("val_loss", patience=5)]


def search_spaces() -> list[list | float]:
    # neurons_space = [i for i in range(20, 160)[::10]]
    neurons_space = [i for i in range(30, 170)[::20]]
    num_layers_space = [i for i in range(2, 5)]
    l2s_space = [1e-4, 1e-3, 1e-2]
    # optimizers_space = ["adam", "sgd", "adagrad"]
    # optimizers_space = [
    #     tf.keras.optimizers.Adam(learning_rate=1e-3),
    #     tf.keras.optimizers.Adam(learning_rate=1e-2),
    #     tf.keras.optimizers.Adam(learning_rate=1e-1),
    # ]
    learning_rate_space = [1e-3, 1e-2, 1e-1]
    # loss_funcs_space = ["mse", "mae", "msle"]
    loss_funcs_space = ["mse", "mae"]
    activation_func_space = ["relu", "tanh", "sigmoid"]
    # activation_func_space = ["relu"]
    output_act_func_space = ["softmax", "linear"]
    batch_size_space = [16, 32, 64, 128]
    # batch_size_space = [32]

    combinations = (
        len(neurons_space)
        * len(num_layers_space)
        * len(l2s_space)
        * len(learning_rate_space)
        * len(loss_funcs_space)
        * len(activation_func_space)
        * len(output_act_func_space)
    )

    return [
        neurons_space,
        num_layers_space,
        l2s_space,
        learning_rate_space,
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
        learning_rates_space,
        loss_funcs_space,
        activation_func_space,
        output_act_func_space,
        batch_size_space,
        combinations,
    ) = search_spaces()

    neurons = hp.Choice("neurons", values=neurons_space)
    num_layers = hp.Choice("num_hidden_layers", values=num_layers_space)
    l2s = hp.Choice("l2", values=l2s_space)
    learning_rate = hp.Choice("learning_rate", values=learning_rates_space)
    loss_funcs = hp.Choice("loss", values=loss_funcs_space)
    activation_func = hp.Choice("activation", values=activation_func_space)
    output_act_func = hp.Choice("output_activation", values=output_act_func_space)
    batch_size = hp.Choice("batch_size", values=batch_size_space)
    print(f"The number of hyperparameter combinations: {combinations}")

    model = Sequential()

    # add the first hidden layer
    # input shape = 2 as we have two variables (potential, pH)
    model.add(
        Dense(
            neurons,
            input_shape=(2,),
            activation=activation_func,
            kernel_regularizer=regularizers.l2(l2=l2s),  # type: ignore
        )
    )

    # add additional layers if num_layers is greater than 1
    # for loop will not trigger if num_layers = 1
    for _ in range(1, num_layers):  # type: ignore
        model.add(
            Dense(
                neurons,
                activation=activation_func,
                kernel_regularizer=regularizers.l2(l2=l2s),  # type: ignore
            )
        )

    # 1 output, current density
    model.add(Dense(1, activation=output_act_func))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_funcs, metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])  # type: ignore

    return model


def create_tuner_and_return_results() -> list[list]:
    X_train, X_val, y_train, y_val, _, _ = normalize_data_for_ANN()

    tuner = keras_tuner.RandomSearch(
        ANN_model,
        objective="val_loss",
        max_trials=trials(),
        overwrite=True,
        directory=directory_for_tuning_results(),
        project_name="ANN",
    )

    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_for_search_and_train(),
        callbacks=early_stopping_callback(),
    )
    tuner.results_summary()
    best_hps: list = tuner.get_best_hyperparameters(num_trials=trials())
    best_models: list = tuner.get_best_models(num_models=trials())

    # save the two best models
    for i in [0, 1]:
        model = best_models[i]
        if i == 0:
            model.save(f"{directory_for_tuning_results()}/first_best_model.h5")
            # copy it to directory with other models
            model.save("models_saved/ANN_tuned_best.h5")
        else:
            model.save(f"{directory_for_tuning_results()}/second_best_model.h5")
            # copy it to directory with other models
            model.save("models_saved/ANN_tuned_second_best.h5")

    return [best_hps, best_models]


def store_tuning_results() -> pd.DataFrame:
    """
    best_hps and best_models are lists of keras objects
    best_hps: <class 'keras_tuner.engine.hyperparameters.hyperparameters.HyperParameters'>
    best_models: <class 'keras.engine.sequential.Sequential'>
    """

    best_hps, best_models = create_tuner_and_return_results()
    _, X_val, _, y_val, _, _ = normalize_data_for_ANN()
    results = []  # append hyperparams for each trial to this list
    for hp_element, best_model in zip(best_hps, best_models):
        results_dict = {}
        _, rmse_loss = best_model.evaluate(X_val, y_val)  # tuple of loss and metric (equal)
        results_dict["val_loss_rmse"] = rmse_loss

        for hp_name in hp_element.values:
            results_dict[hp_name] = hp_element.get(hp_name)

        results.append(results_dict)

    results = pd.DataFrame(results).sort_values(by="val_loss_rmse", ascending=True)  # lowest loss : index 1
    results.insert(0, "best_models_sorted", [i for i in range(1, len(best_models) + 1)])
    results.to_csv(f"{directory_for_tuning_results()}/{name_df_hyperparams_results()}", sep=",", index=False)
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
