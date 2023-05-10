from catboost import CatBoostRegressor
from src.data_preprocessing import return_training_data_X_y, split_into_training_and_validation


def tune_catboost():
    X, y = return_training_data_X_y()
