from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor)
from xgboost import XGBRegressor

def get_model_grids():
    return {
        "XGBoost": {
            "model": XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
                tree_method="hist",
            ),
            "param_grid": {
                "n_estimators": [200, 500, 800],
                "max_depth": [2, 3, 5],
                "learning_rate": [0.01, 0.03, 0.05],
                "min_child_weight": [1, 5, 10],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
        },

        "HistGradientBoosting": {
            "model": HistGradientBoostingRegressor(
                random_state=42
            ),
            "param_grid": {
                "learning_rate": [0.03, 0.05, 0.1],
                "max_iter": [100, 300, 600],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [10, 20, 50],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
        },

        "RandomForest": {
            "model": RandomForestRegressor(
                random_state=42,
                n_jobs=1,
            ),
            "param_grid": {
                "n_estimators": [200, 500, 800],
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5],
                "max_features": [0.5, 0.8, 1.0],
                "max_samples": [None, 0.7, 0.9],
            },
        },

        "ExtraTrees": {
            "model": ExtraTreesRegressor(
                random_state=42,
                n_jobs=1,
            ),
            "param_grid": {
                "n_estimators": [200, 500, 800],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "max_features": [0.5, 0.8, 1.0],
                "bootstrap": [False],
            },
        },

        "GradientBoosting": {
            "model": GradientBoostingRegressor(
                random_state=42
            ),
            "param_grid": {
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 5],
                "min_samples_leaf": [5, 10, 20],
                "subsample": [0.7, 0.9, 1.0],
                "loss": ["squared_error"],
            },
        },

        "Ridge": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]),
            "param_grid": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__fit_intercept": [True, False],
                "model__solver": ["auto", "lsqr"],
                "model__tol": [1e-4, 1e-3, 1e-2],
            },
        },

        "ElasticNet": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(
                    max_iter=20_000,
                    random_state=42,
                )),
            ]),
            "param_grid": {
                "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95],
                "model__fit_intercept": [True, False],
                "model__tol": [1e-5, 1e-4, 1e-3],
                "model__positive": [False, True],
                "model__max_iter": [10_000, 20_000, 50_000],
            },
        },
    }