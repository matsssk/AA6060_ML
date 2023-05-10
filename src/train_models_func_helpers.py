import time
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from src.compare_models_with_exp_data import return_test_data
from src.data_preprocessing import return_training_data_X_y, split_into_training_and_validation
import numpy as np


def _training_time_per_tree(training_time: float, trees: int) -> float:
    """
    returns average training time per tree in seconds for DTs
    """
    return training_time / trees


def train_random_forest_for_some_hyperparams(
    n_trees,
    max_features,
    directory_to_save_df: str,
    best_model: bool,
    feat_imp_list: list | None,
    training_times_all_models: dict,
    training_time_per_tree: dict,
    feature_imp: dict,
):
    """
    This function is training RF on a combination of number of trees and max features
    It is training and storing results for all test data (test pH)
    It is also comparing feature importances for all training sessions, as well as returning
    the error between exacted and predicted data (current density) for all pHs
    Store feature_importances in its own dictionairy if best_model = False
    """
    X, y = return_training_data_X_y()  # y is log(abs(i))
    rf = RandomForestRegressor(n_estimators=n_trees, max_features=max_features)  # MSE default loss criterion
    t0 = time.perf_counter()
    rf.fit(X, y)
    runtime = time.perf_counter() - t0
    # get feature importances and plot in df_feature_imp

    # store values to compare with other models if we have already tuned the model
    if best_model and feat_imp_list is None:
        training_times_all_models["rf"] = runtime
        training_time_per_tree["rf"] = _training_time_per_tree(runtime, n_trees)
        feature_imp["rf"] = rf.feature_importances_
        print("executed")

    # evaluate the feature importances against each other to compare
    elif not best_model and feat_imp_list is not None:
        feat_imp_list.append([n_trees, max_features, rf.feature_importances_[0], rf.feature_importances_[1]])
    else:
        raise ValueError
    # X_test and y_test containt all pHs
    X_test, y_test = return_test_data()
    testing_phs = pd.read_csv("testing_pHs.csv", sep="\t")["test_pHs"]
    df_errors = pd.DataFrame()
    df_errors["pH"] = testing_phs
    mape_log_list = []
    mape_list = []
    rmse_list = []
    for ph in testing_phs:
        # create mask to get data with only that pH
        ph_mask = X_test[:, 1] == ph
        X_test_ph, y_test_ph = X_test[ph_mask], y_test[ph_mask]
        # predict and store results to pandas dataframe
        pred_ph = rf.predict(X_test_ph)
        if best_model:
            df_pred = pd.DataFrame({"E [V]": X_test_ph[:, 0], "Current density [A/cm2]": 10**pred_ph})
            df_pred.to_csv(f"models_data/random_forest_output/current_density_pred_ph_{ph}.csv", sep="\t", index=False)

        # get errors from training data vs test data
        mape_log_list.append(mape(pred_ph, y_test_ph))  # must be array-like
        mape_list.append(10 ** mape_log_list[-1])
        rmse_list.append(mse(pred_ph, y_test_ph, squared=False))

    df_errors["(MAPE of log)"] = mape_log_list
    df_errors["mape"] = mape_list
    df_errors["rmse"] = rmse_list
    df_errors.to_csv(
        directory_to_save_df,
        sep="\t",
        index=False,
    )


def average_error_for_each_trial(tuning_files_dir: str):
    """
    Loops through all csv files with errors for the various trials in results_from_tuning
    and stores a dataframe with the average error across all pH for that trial

    param: tuning_files_dir, the directory of where the tuning results are stored for the machine learning
           algorithm under consideration
    """

    # sort on all files that starts with "error"
    files = [file for file in os.listdir(tuning_files_dir) if file.startswith("error")]
    n_trees, max_features, avg_error_across_ph = [], [], []
    for file in files:
        if file.startswith("errors"):
            avg_error_across_ph.append(
                format((pd.read_csv(os.path.join(tuning_files_dir, file), sep="\t")["rmse"].sum() / 4), ".4f")
            )
            n_trees.append(int(file.split("errors_trees_")[1].split("_")[0]))
            max_features.append(float(file.split(".csv")[0][-3:]))
        else:
            continue
    pd.DataFrame({"Trees": n_trees, "Max features": max_features, "Average RMSE": avg_error_across_ph}).sort_values(
        by="Average RMSE"
    ).to_csv(f"{tuning_files_dir}average_rmse_for_each_trial.csv", index=False, sep="\t")


if __name__ == "__main__":
    average_error_for_each_trial(tuning_files_dir="models_data/random_forest_output/results_from_tuning/")
