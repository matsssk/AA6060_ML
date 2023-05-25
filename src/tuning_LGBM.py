import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from lightgbm import LGBMRegressor
from src.data_preprocessing import return_training_data_X_y
import time


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def tune_lgbm():
    X, y = return_training_data_X_y()
    # search space
    param_grid = {
        "num_leaves": [21, 31, 41, 51, 61],
        "learning_rate": [0.001, 0.01, 0.1, 0.5],
        "n_estimators": [1000],
        "boosting_type": ["gbdt", "dart"],
    }

    # Instantiate the model
    lgbm = LGBMRegressor()
    t0 = time.perf_counter()
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        scoring=make_scorer(rmse, greater_is_better=False),
        cv=5,
        verbose=3,
        n_jobs=None,
    )

    # Fit the model
    grid_search.fit(X, y)

    # Extract results to a pandas DataFrame and sort by rmse
    results = pd.DataFrame(grid_search.cv_results_)
    print(f"time spent: {time.perf_counter() - t0} s")
    # Select only the desired columns
    results = results[
        ["param_num_leaves", "param_learning_rate", "param_n_estimators", "param_boosting_type", "mean_test_score"]
    ]

    # Sort the results by test score
    results = results.sort_values(by="mean_test_score", ascending=True)

    # Save results to csv
    results.to_csv("models_data/lgbm_info/tuning_results1.csv", sep="\t", index=False)


if __name__ == "__main__":
    tune_lgbm()
