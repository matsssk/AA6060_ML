# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_squared_error
# from src.data_preprocessing import return_training_data_X_y
# import pandas as pd
# import time


# def tune_RF():
#     param_grid = {"n_estimators": [1, 5, 10, 100, 200], "max_features": [0.3, 1.0]}
#     rf = RandomForestRegressor()
#     grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5)
#     X_train, y_train = return_training_data_X_y()

#     grid_search.fit(X_train, y_train)

#     best_params = grid_search.best_params_
#     best_score = -grid_search.best_score_

#     t0 = time.perf_counter()
#     best_rf = RandomForestRegressor(**best_params)
#     best_rf.fit(X_train, y_train)  # X_train and y_train are your training data
#     runtime = time.perf_counter() - t0
#     runtime_per_tree = runtime / best_params["n_estimators"]

#     # Evaluate the best model on the test data
#     y_pred = best_rf.predict(X_test)  # X_test is your test data
#     mse = mean_squared_error(y_test, y_pred)

#     df = pd.DataFrame(feat_imp_list, columns=["n_estimators", "max_features", "potential (E)", "pH"])
#     df.to_csv("models_data/random_forest_output/results_from_tuning/feature_importances.csv", sep="\t", index=False)
#     print("Best Parameters:", best_params)
#     print("Best MSE Score:", best_score)
#     print("MSE on Test Data:", mse)
