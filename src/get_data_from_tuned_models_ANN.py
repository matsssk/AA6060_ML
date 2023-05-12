from src.data_preprocessing import normalize_data_for_ANN
from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers
import pandas as pd
import time
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from hyperparameter_tuning_ANN import (
    directory_for_tuning_results,
    name_df_hyperparams_results,
    epochs_for_search_and_train,
    early_stopping_callback,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def plot_learning_curves_best_n_models():
    """
    It would be benefitial to access the learning curves directly
    from the tuner.search. This is however not possible yet with
    KerasTuner and therefor the models that performed best (lowest val loss)
    will be trained again with the same hyperparams to obtain learning curves
    """
    # create figure to plot learning curves for each trained model
    plt.figure(figsize=(10, 10))
    plt.xlabel("Epochs")
    plt.ylabel("Error, RMSE")

    hyperparams_df: pd.DataFrame = pd.read_csv(
        f"{directory_for_tuning_results()}/{name_df_hyperparams_results()}", sep=","
    )
    # create model and train it, as well as plot the learning curve for each model
    X_train, X_val, y_train, y_val, _, _ = normalize_data_for_ANN()
    # train best n models to obtain learning curves
    # store training time
    runtimes = []
    epochs = []
    colors = ["r", "b", "k", "y", "o"]
    # for each set of hyperparameters, construct and train model to obtain learning curves
    for idx, row in hyperparams_df.iterrows():
        model = Sequential()

        # first hidden layer
        # input shape = 2 as we have two variables (potential, pH)
        model.add(
            Dense(
                row["neurons"],
                input_shape=(2,),
                activation=row["activation"],
                kernel_regularizer=regularizers.l2(l2=row["l2"]),
            )
        )
        # add additional layers if num_layers is greater than 1
        # for loop will not trigger if num_hidden_layers = 1
        for _ in range(1, row["num_hidden_layers"]):
            model.add(
                Dense(
                    row["neurons"],
                    activation=row["activation"],
                    kernel_regularizer=regularizers.l2(l2=row["l2"]),
                )
            )

        # add output layer. 1 output: current density
        model.add(Dense(1, activation=row["output_activation"]))
        model.compile(
            optimizer=row["optimizer"], loss=row["loss"], metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
        )

        t0 = time.perf_counter()
        history = model.fit(
            X_train,
            y_train,
            batch_size=row["batch_size"],
            epochs=epochs_for_search_and_train(),
            callbacks=[early_stopping_callback()],
            verbose=1,
            validation_data=(X_val, y_val),
        )
        runtimes.append(time.perf_counter() - t0)
        df_loss = pd.DataFrame(history.history)
        df_loss.insert(0, "epochs", df_loss.shape[0])
        epochs.append(df_loss.shape[0])

        # store training and validation loss
        df_loss.to_csv(f"models_data/ANN_info/training_val_loss{idx}", sep="\t", index=False)

        # plot learning curves for each model
        epochs_list = [iter for iter in range(1, df_loss.shape[0] + 1, 1)]

        model_index = idx + 1  # type: ignore
        plt.semilogy(epochs_list, df_loss["rmse"], "s-", label=f"Training loss, model {model_index}", color=colors[idx])  # type: ignore
        plt.semilogy(
            epochs_list, df_loss["val_rmse"], "s--", label=f"Validation loss, model {model_index}", color=colors[idx]  # type: ignore
        )

    hyperparams_df["runtime"] = runtimes
    hyperparams_df["epochs"] = epochs
    hyperparams_df = hyperparams_df.drop("val_loss_rmse", axis=1)  #  remove old val_loss from tuner.search
    hyperparams_df.to_csv("models_data/ANN_info/data_for_n_best_models.csv", sep=",", index=False)

    plt.legend()
    plt.savefig("summarized_data_figures_datafiles/learning_curves_best_models_tuned_ann.png")


if __name__ == "__main__":
    plot_learning_curves_best_n_models()