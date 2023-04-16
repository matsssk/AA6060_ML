# MUST DO
# må fikse earlystopping for alle GBDTS
# må fikse så df_with_results_from_tunersearch.csv er sortert på val_loss


import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse

from src.hyperparameter_tuning import path_hyperparam_folder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from src.compare_models_with_exp_data import return_test_data
from src.data_preprocessing import return_training_data_X_y, split_into_training_and_validation

import pickle
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# dict with training time for models
training_time = {}
# dict to plot feature importances in models
feature_imp = {}


# def extract_hyperparams_from_json(model: str) -> dict:
#     """
#     Returns a dictionairy with all the hyperparams optimized through RandomSearchCV
#     in hyperparameter_tuning.py
#     """
#     with open(f"{path_hyperparam_folder()}/{model}.json", "rb") as f:
#         # Use pickle.load to load the dictionary from the file
#         best_params: dict = pickle.load(f)
#     return best_params


def catboost_model() -> None:
    """
    Train CatBoost model and load it later in compare_models_with_exp_data.py
    """
    # load hyperparams from RandomSearchCV in hyperparameter_tuning.py

    # y needs shape (1, N_ROWS)
    X, y = return_training_data_X_y()
    X_train, X_val, y_train, y_val = split_into_training_and_validation(X, y)

    # extract key-value pairs, use default values besides the ones in RS
    cb = CatBoostRegressor(n_estimators=2000, loss_function="RMSE", early_stopping_rounds=50)
    t0 = time.perf_counter()
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    runtime = time.perf_counter() - t0
    training_time["cb"] = runtime

    # get feature importances and divide by 100 to become fractions of 1
    feature_imp["cb"] = [v / 100 for v in cb.feature_importances_]

    cb.save_model("models_saved/catboost_model.cbm", format="cbm")


def random_forest_model() -> None:
    """
    Train RandomForest model and load it later in compare_models_with_exp_data.py
    Model size is incredibly large, so the model will not be saved, but important outputs
    will be stored txt files and loaded in compare_models...
    """

    X, y = return_training_data_X_y()
    n_estimators = 500
    rf = RandomForestRegressor(n_estimators=n_estimators)
    t0 = time.perf_counter()
    rf.fit(X, y)
    runtime = time.perf_counter() - t0
    # get feature importances and plot in df_feature_imp
    feature_imp["rf"] = rf.feature_importances_

    training_time["rf"] = runtime

    X_test, y_test = return_test_data()
    # apply prediction for all stages to evaluate loss as function of iterations
    pred_each_iter = [est.predict(X_test) for est in rf.estimators_]

    # store last pred to csv file
    df_pred = pd.DataFrame()
    df_pred["E [V]"] = X_test[:, 0]
    df_pred["Current density [A/cm2]"] = 10 ** pred_each_iter[-1]
    df_pred.to_csv("models_data/random_forest_output/current_density_pred.csv", sep="\t")

    # rmse for each pred
    rmse_rf = [mse(pred, y_test, squared=False) for pred in pred_each_iter]
    # store a df with iterations and rmse as columns
    df_rmse = pd.DataFrame()
    df_rmse["Trees"] = [iter for iter in range(1, n_estimators + 1, 1)]
    df_rmse["rmse of log"] = rmse_rf
    df_rmse.to_csv("models_data/random_forest_output/rmse.csv", sep="\t")

    # last pred is the used pred
    df_mape = pd.DataFrame()
    df_mape["(MAPE of log)"] = [mape(pred_each_iter[-1], y_test)]  # must be array-like
    df_mape["MAPE"] = 10 ** df_mape["(MAPE of log)"]
    df_mape.to_csv("models_data/random_forest_output/mape.csv", sep="\t")


def xgboost_model() -> None:
    X, y = return_training_data_X_y()
    # assign test data to evaluate against
    X_test, y_test = return_test_data()
    # number of estimators/iterations
    n_iter = 500

    xgb = XGBRegressor(eval_metric=["rmse"], n_estimators=n_iter)
    t0 = time.perf_counter()
    xgb.fit(X, y, eval_set=[(X, y), (X_test, y_test)], verbose=False)
    runtime = time.perf_counter() - t0
    training_time["xgb"] = runtime

    # get feature importances and plot in df_feature_imp
    feature_imp["xgb"] = xgb.feature_importances_

    # get loss for each iteration
    evals_result = xgb.evals_result()
    # convert training loss to df
    df = pd.DataFrame()
    df["iterations"] = [iter for iter in range(1, n_iter + 1, 1)]
    df["rmse"] = evals_result["validation_0"]["rmse"]
    # store df in csv file
    df.to_csv("models_data/xgboost_info/training_loss.csv", sep="\t", index=False)

    xgb.save_model("models_saved/xgboost.txt")


def lightgbm_model() -> None:
    X, y = return_training_data_X_y()
    X_test, y_test = return_test_data()
    n_iter = 500
    lgbm = LGBMRegressor(n_estimators=n_iter)
    t0 = time.perf_counter()
    lgbm.fit(X, y, eval_set=[(X, y), (X_test, y_test)], eval_metric=["rmse"], callbacks=[lgb.log_evaluation(False)])
    runtime = time.perf_counter() - t0
    training_time["lgbm"] = runtime

    # get loss for each iteration
    evals_result = lgbm.evals_result_

    # get feature importances, defaults on split = how many times the feature is used in the model
    # convert the number to fraction of total splits
    feature_imp["lgbm"] = [v / sum(lgbm.feature_importances_) for v in lgbm.feature_importances_]

    df = pd.DataFrame()
    df["iterations"] = [iter for iter in range(1, n_iter + 1, 1)]
    df["rmse"] = evals_result["training"]["rmse"]

    # store df to csv file
    df.to_csv("models_data/lgbm_info/training_loss.csv", sep="\t", index=False)

    lgbm.booster_.save_model("models_saved/lgbm.txt")


def load_ANN_runtime() -> None:
    # load best model's runtime
    # ANN is already trained through tuner.search in hyperparameter_tunig.py

    training_time["ANN"] = pd.read_csv("models_data/ANN_info/data_for_n_best_models.csv", sep=",", usecols=["runtime"])[
        "runtime"
    ][0]


def plot_histogram_training_time() -> None:
    plt.figure(figsize=(10, 10))
    plt.xlabel("Model")
    plt.ylabel("Training time [s]")
    df = pd.DataFrame(list(training_time.items()), columns=["Model", "Time"]).sort_values("Time")

    # ajust xticks locations
    pos = [0, 1, 2, 3, 4]
    plt.bar(df["Model"], df["Time"], width=0.25)
    plt.xticks(
        pos, labels=[f" {k.upper()}: {round(v,2)} s" for k, v in zip(df["Model"], df["Time"])], rotation=45, ha="center"
    )

    plt.savefig("model_figures/models_training_time.png")


def feature_importances_to_pd():
    models: list = list(feature_imp.keys())
    potential = [feature_imp[key][0] for key in feature_imp]
    pH = [feature_imp[key][1] for key in feature_imp]
    df = pd.DataFrame({"Model": models, "Potential": potential, "pH": pH})
    df.to_csv("model_figures/feature_importances.csv", sep="\t", index=False)


if __name__ == "__main__":
    catboost_model()
    random_forest_model()
    xgboost_model()
    lightgbm_model()
    load_ANN_runtime()
    plot_histogram_training_time()
    feature_importances_to_pd()
