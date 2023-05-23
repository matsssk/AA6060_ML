import time
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from xgboost import XGBRegressor
from src.data_preprocessing import return_training_data_X_y, split_into_training_and_validation
from src.train_models_func_helpers import (
    train_random_forest_for_some_hyperparams,
    create_df_average_error_for_each_trial_across_phs,
    _training_time_per_tree,
)
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# dict with training time for DTs
training_time_per_tree = {}
# dict with all training times
training_times_all_models = {}
# dict to plot feature importances in models
feature_imp = {}

n_iterations_GBTS = {}


def random_forest_model(tune: bool = True) -> None:
    """
    Train Random Forest model using RandomForestRegressor from scikit-learn
    Model is not saved as the size of the file is too large for standard GitHub membership
    Prediction data is stored in models_data/random_forest_output

    This should preferably be done with a RandomSearchCV technique, however it was not considered enough time to rewrite all code
    """
    if tune:
        # tune based on scikit-learn docs suggestions
        n_trees_list = [1, 5, 10, 100, 200]
        max_features_list = [0.3, 1.0]

        feat_imp_list = []
        # train model based on given hyperparams
        for n_trees in n_trees_list:
            for max_features in max_features_list:
                dir = f"models_data/random_forest_output/results_from_tuning/errors_trees_{n_trees}_max_feat{max_features}.csv"
                # tune
                train_random_forest_for_some_hyperparams(
                    n_trees,
                    max_features,
                    dir,
                    False,
                    feat_imp_list,
                    training_times_all_models,
                    training_time_per_tree,
                    feature_imp,
                )
        # store feature importances to csv
        df = pd.DataFrame(feat_imp_list, columns=["n_estimators", "max_features", "potential (E)", "pH"])
        df.to_csv("models_data/random_forest_output/results_from_tuning/feature_importances.csv", sep="\t", index=False)
        df.style.hide(axis="index").to_latex(
            "models_data/random_forest_output/results_from_tuning/feature_importances.tex",
            hrules=True,
            position="H",
            position_float="centering",
            label="feature_imp_RF_tuning",
            caption="Feature importances for the trials for tuning RF",
        )
        create_df_average_error_for_each_trial_across_phs(
            tuning_files_dir="models_data/random_forest_output/results_from_tuning/"
        )

    elif not tune:
        # default values in RandomForestRegressor
        # n_trees_list = [100]
        # max_features_list = [1.0]

        # collect best hyperparams
        # need to find the combination with lowest error (any column)
        # the best combo has lowest average error (or sum, equivalent)
        lowest_sum = np.inf
        lowest_df = None
        n_trees_best, max_feat_best = 0, 0
        # Loop through all files in the folder and find the file with best hyperparams
        for file_name in os.listdir("models_data/random_forest_output/results_from_tuning"):
            if file_name.endswith(".csv") and file_name.startswith("errors"):
                file_path = os.path.join("models_data/random_forest_output/results_from_tuning/", file_name)
                df = pd.read_csv(file_path, sep="\t")
                ph_sum = df["rmse"].sum()
                # Check if this sum is the lowest so far
                if ph_sum < lowest_sum:
                    lowest_sum = ph_sum
                    lowest_df = df
                    try:
                        n_trees_best, max_feat_best = int(file_name.split("errors_trees_")[1].split("_")[0]), float(
                            file_name.split(".csv")[0][-3:]
                        )
                    except FileNotFoundError:
                        raise FileNotFoundError

        # train the model with the optimized hyperparameters
        train_random_forest_for_some_hyperparams(
            n_trees_best,
            max_feat_best,
            "models_data/random_forest_output/errors_trees_best_params.csv",
            True,
            None,
            training_times_all_models,
            training_time_per_tree,
            feature_imp,
        )

        create_df_average_error_for_each_trial_across_phs(
            tuning_files_dir="models_data/random_forest_output/results_from_tuning/"
        )

    else:
        raise ValueError


def catboost_model() -> None:
    """
    Train CatBoost model and load it later in compare_models_with_exp_data.py
    Earlystopping of 50 rounds are applied, i.e. model will stop if no new loss minima
    are found within 50 iterations after the previous minima


    Notes:
        For default learning rate(adjusted) the model could do 10000 iterations without converging

    """
    n_iterations = 100000
    n_iterations_GBTS["cb"] = n_iterations

    # load hyperparams from RandomSearchCV in hyperparameter_tuning.py

    # y needs shape (1, N_ROWS)
    X, y = return_training_data_X_y()
    X_train, X_val, y_train, y_val = split_into_training_and_validation(X, y)

    # extract key-value pairs, use default values besides the ones in RS
    cb = CatBoostRegressor(n_estimators=n_iterations, loss_function="RMSE", learning_rate=0.35)
    t0 = time.perf_counter()
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100, early_stopping_rounds=50)

    runtime = time.perf_counter() - t0
    training_times_all_models["cb"] = runtime
    training_time_per_tree["cb"] = _training_time_per_tree(runtime, n_iterations)

    # get feature importances and divide by 100 to become fractions of 1
    feature_imp["cb"] = [v / 100 for v in cb.feature_importances_]

    cb.save_model(f"models_saved/catboost_model.cbm", format="cbm")


def xgboost_model() -> None:
    """
    Train XGBoost model and load it later in compare_models_with_exp_data.py
    Earlystopping of 50 rounds are applied, i.e. model will stop if no new loss minima
    are found within 50 iterations after the previous minima
    """
    n_iterations = 100000
    n_iterations_GBTS["xgb"] = n_iterations

    X, y = return_training_data_X_y()
    X_train, X_val, y_train, y_val = split_into_training_and_validation(X, y)

    xgb = XGBRegressor(eval_metric=["rmse"], n_estimators=n_iterations, early_stopping_rounds=50)
    t0 = time.perf_counter()
    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=100)
    runtime = time.perf_counter() - t0
    training_times_all_models["xgb"] = runtime
    training_time_per_tree["xgb"] = _training_time_per_tree(runtime, n_iterations)

    # get feature importances and plot in df_feature_imp
    feature_imp["xgb"] = xgb.feature_importances_

    # get loss for each iteration
    evals_result = xgb.evals_result()
    train_loss, val_loss = evals_result["validation_0"]["rmse"], evals_result["validation_1"]["rmse"]

    # convert training loss to df and save df
    pd.DataFrame(
        {
            "iter": [iter for iter in range(1, len(train_loss) + 1, 1)],
            "train_loss_rmse": train_loss,
            "val_loss_rmse": val_loss,
        }
    ).to_csv("models_data/xgboost_info/train_val_loss.csv", sep="\t", index=False)

    xgb.save_model("models_saved/xgboost.txt")


def lightgbm_model() -> None:
    n_iterations = 10**5
    n_iterations_GBTS["lgbm"] = n_iterations

    X, y = return_training_data_X_y()
    X_train, X_val, y_train, y_val = split_into_training_and_validation(X, y)

    lgbm = LGBMRegressor(n_estimators=n_iterations)
    t0 = time.perf_counter()
    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=["rmse"],
        callbacks=[lgb.log_evaluation(True)],
        early_stopping_rounds=50,
    )
    runtime = time.perf_counter() - t0
    training_times_all_models["lgbm"] = runtime
    training_time_per_tree["lgbm"] = _training_time_per_tree(runtime, n_iterations)

    # get feature importances, defaults on split = how many times the feature is used in the model
    # convert the number to fraction of total splits
    feature_imp["lgbm"] = [v / sum(lgbm.feature_importances_) for v in lgbm.feature_importances_]

    evals_result = lgbm.evals_result_
    train_loss, val_loss = evals_result["training"]["rmse"], evals_result["valid_1"]["rmse"]

    # convert training loss to df and save df
    pd.DataFrame(
        {
            "iter": [iter for iter in range(1, len(train_loss) + 1, 1)],
            "train_loss_rmse": train_loss,
            "val_loss_rmse": val_loss,
        }
    ).to_csv("models_data/lgbm_info/train_val_loss.csv", sep="\t", index=False)

    lgbm.booster_.save_model("models_saved/lgbm.txt")


def load_ANN_runtime() -> None:
    # load best model's runtime
    # ANN is already trained through tuner.search in hyperparameter_tunig.py

    training_times_all_models["ANN"] = pd.read_csv(
        "models_data/ANN_info/data_for_n_best_models.csv", sep="\t", usecols=["runtime"]
    )["runtime"][0]


def plot_histogram_training_time_all_models() -> None:
    plt.figure()
    plt.xlabel("Algorithm")
    plt.ylabel("Training time [s]")
    df = pd.DataFrame(list(training_times_all_models.items()), columns=["Model", "Time"]).sort_values("Time")

    # ajust xticks locations
    pos = [0, 1, 2, 3, 4]
    plt.bar(df["Model"], df["Time"], width=0.25, color="dimgray")
    plt.xticks(
        pos, labels=[f" {k.upper()}: {round(v,2)} s" for k, v in zip(df["Model"], df["Time"])], rotation=45, ha="center"
    )

    plt.savefig("summarized_data_figures_datafiles/models_training_time_all_models.pgf")


def plot_histogram_training_time_per_tree_DTs():
    plt.figure()
    plt.xlabel("Algorithm")
    plt.ylabel("Training time per tree [s]")
    df = pd.DataFrame(list(training_time_per_tree.items()), columns=["Model", "Time"]).sort_values("Time")

    # ajust xticks locations
    pos = [0, 1, 2, 3]
    plt.bar(df["Model"], df["Time"], width=0.25, color="dimgray")
    plt.xticks(
        pos, labels=[f" {k.upper()}: {round(v,2)} s" for k, v in zip(df["Model"], df["Time"])], rotation=45, ha="center"
    )

    plt.savefig("summarized_data_figures_datafiles/models_training_time_per_tree_DTs.pgf")


def feature_importances_to_pd():
    models: list = list(feature_imp.keys())
    potential = [feature_imp[key][0] for key in feature_imp]
    pH = [feature_imp[key][1] for key in feature_imp]
    df = pd.DataFrame({"Model": models, "Potential": potential, "pH": pH})
    df.to_csv("summarized_data_figures_datafiles/csv_files/feature_importances.csv", sep="\t", index=False)


def save_iterations_GBDTs_into_df():
    df = pd.DataFrame(columns=["model", "max_iterations"])
    ks, vs = [k for k, v in n_iterations_GBTS.items()], [v for k, v in n_iterations_GBTS.items()]
    df["model"], df["max_iterations"] = ks, vs
    df.to_csv("summarized_data_figures_datafiles/csv_files/max_iterations_GBDTs.csv", sep="\t", index=False)


if __name__ == "__main__":
    random_forest_model(tune=False)
    catboost_model()
    xgboost_model()
    lightgbm_model()
    load_ANN_runtime()
    plot_histogram_training_time_all_models()
    plot_histogram_training_time_per_tree_DTs()
    feature_importances_to_pd()
    save_iterations_GBDTs_into_df()
