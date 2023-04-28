import time

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

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# dict with training time for DTs
training_time_per_tree = {}
# dict with all training times
training_times_all_models = {}
# dict to plot feature importances in models
feature_imp = {}

n_iterations_GBTS = {}


def _training_time_per_tree(training_time: float, trees: int) -> float:
    """
    returns average training time per tree in seconds for DTs
    """
    return training_time / trees


def random_forest_model() -> None:
    """
    Train Random Forest model using RandomForestRegressor from scikit-learn
    Model is not saved as the size of the file is too large for standard GitHub membership
    Prediction data is stored in models_data/random_forest_output
    """
    n_trees = 100
    X, y = return_training_data_X_y()
    rf = RandomForestRegressor(n_estimators=n_trees)  # MSE default loss criterion
    t0 = time.perf_counter()
    rf.fit(X, y)
    runtime = time.perf_counter() - t0
    # get feature importances and plot in df_feature_imp
    training_times_all_models["rf"] = runtime
    training_time_per_tree["rf"] = _training_time_per_tree(runtime, n_trees)
    feature_imp["rf"] = rf.feature_importances_

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
        df_pred = pd.DataFrame({"E [V]": X_test_ph[:, 0], "Current density [A/cm2]": 10**pred_ph})
        df_pred.to_csv(f"models_data/random_forest_output/current_density_pred_ph_{ph}.csv", sep="\t", index=False)

        # get errors from training data vs test data
        mape_log_list.append(mape(pred_ph, y_test_ph))  # must be array-like
        mape_list.append(10 ** mape_log_list[-1])
        rmse_list.append(mse(pred_ph, y_test_ph, squared=False))

    df_errors["(MAPE of log)"] = mape_log_list
    df_errors["mape"] = mape_list
    df_errors["rmse"] = rmse_list
    df_errors.to_csv("models_data/random_forest_output/errors.csv", sep="\t", index=False)


def catboost_model() -> None:
    """
    Train CatBoost model and load it later in compare_models_with_exp_data.py
    Earlystopping of 50 rounds are applied, i.e. model will stop if no new loss minima
    are found within 50 iterations after the previous minima


    Notes:
        For default learning rate(adjusted) the model could do 10000 iterations without converging

    """
    n_iterations = 100
    n_iterations_GBTS["cb"] = n_iterations

    # load hyperparams from RandomSearchCV in hyperparameter_tuning.py

    # y needs shape (1, N_ROWS)
    X, y = return_training_data_X_y()
    X_train, X_val, y_train, y_val = split_into_training_and_validation(X, y)

    # extract key-value pairs, use default values besides the ones in RS
    cb = CatBoostRegressor(n_estimators=n_iterations, loss_function="RMSE", learning_rate=0.5)
    t0 = time.perf_counter()
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100, early_stopping_rounds=5)

    runtime = time.perf_counter() - t0
    training_times_all_models["cb"] = runtime
    training_time_per_tree["cb"] = _training_time_per_tree(runtime, n_iterations)

    # get feature importances and divide by 100 to become fractions of 1
    feature_imp["cb"] = [v / 100 for v in cb.feature_importances_]

    cb.save_model("models_saved/catboost_model.cbm", format="cbm")


def xgboost_model() -> None:
    """
    Train XGBoost model and load it later in compare_models_with_exp_data.py
    Earlystopping of 50 rounds are applied, i.e. model will stop if no new loss minima
    are found within 50 iterations after the previous minima
    """
    n_iterations = 100
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
    n_iterations = 100
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
        "models_data/ANN_info/data_for_n_best_models.csv", sep=",", usecols=["runtime"]
    )["runtime"][0]


def plot_histogram_training_time_all_models() -> None:
    plt.figure(figsize=(10, 10))
    plt.xlabel("Algorithm")
    plt.ylabel("Training time [s]")
    df = pd.DataFrame(list(training_times_all_models.items()), columns=["Model", "Time"]).sort_values("Time")

    # ajust xticks locations
    pos = [0, 1, 2, 3, 4]
    plt.bar(df["Model"], df["Time"], width=0.25, color="#777777")
    plt.xticks(
        pos, labels=[f" {k.upper()}: {round(v,2)} s" for k, v in zip(df["Model"], df["Time"])], rotation=45, ha="center"
    )

    plt.savefig("model_figures/models_training_time_all_models.png")


def plot_histogram_training_time_per_tree_DTs():
    plt.figure(figsize=(10, 10))
    plt.xlabel("Algorithm")
    plt.ylabel("Training time per tree [s]")
    df = pd.DataFrame(list(training_time_per_tree.items()), columns=["Model", "Time"]).sort_values("Time")

    # ajust xticks locations
    pos = [0, 1, 2, 3]
    plt.bar(df["Model"], df["Time"], width=0.25, color="#777777")
    plt.xticks(
        pos, labels=[f" {k.upper()}: {round(v,2)} s" for k, v in zip(df["Model"], df["Time"])], rotation=45, ha="center"
    )

    plt.savefig("model_figures/models_training_time_per_tree_DTs.png")


def feature_importances_to_pd():
    models: list = list(feature_imp.keys())
    potential = [feature_imp[key][0] for key in feature_imp]
    pH = [feature_imp[key][1] for key in feature_imp]
    df = pd.DataFrame({"Model": models, "Potential": potential, "pH": pH})
    df.to_csv("model_figures/feature_importances.csv", sep="\t", index=False)


def save_iterations_GBDTs_into_df():
    df = pd.DataFrame(columns=["model", "max_iterations"])
    ks, vs = [k for k, v in n_iterations_GBTS.items()], [v for k, v in n_iterations_GBTS.items()]
    df["model"], df["max_iterations"] = ks, vs
    df.to_csv("model_figures/max_iterations_GBDTs.csv", sep="\t", index=False)


if __name__ == "__main__":
    random_forest_model()
    catboost_model()
    xgboost_model()
    lightgbm_model()
    load_ANN_runtime()
    plot_histogram_training_time_all_models()
    plot_histogram_training_time_per_tree_DTs()
    feature_importances_to_pd()
    save_iterations_GBDTs_into_df()
