from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, BaggingRegressor, 
    GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np


def get_model_grids():
    return {
        'Ridge': {
            'model': Ridge(),
            'param_grid': {
                'alpha': [0.01, 0.1, 1, 10],
                'fit_intercept': [True, False],
                'solver': ['auto', 'cholesky', 'lsqr'],
            },
        },
        'Lasso': {
            'model': Lasso(),
            'param_grid': {
                'alpha': [0.001, 0.01, 0.1, 1],
                'fit_intercept': [True, False],
                'max_iter': [1000, 5000],
            },
        },
        'ElasticNet': {
            'model': ElasticNet(),
            'param_grid': {
                'alpha': [0.001, 0.01, 0.1],
                'l1_ratio': [0.3, 0.5, 0.7],
                'max_iter': [1000, 5000],
            },
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_leaf': [1, 3, 5],
            },
        },
        'AdaBoost': {
            'model': AdaBoostRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5],
            },
        },
        'ExtraTrees': {
            'model': ExtraTreesRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_leaf': [1, 3, 5],
            },
        },
        'Bagging': {
            'model': BaggingRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [10, 50, 100],
                'max_samples': [0.5, 0.8, 1.0],
            },
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
            },
        },
        'HistGBR': {
            'model': HistGradientBoostingRegressor(random_state=42),
            'param_grid': {
                'max_iter': [100, 200, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
            },
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
            },
        },
    }


def run_grid_search(X_train, y_train, X_val, y_val, n_splits=5):
    grid_results = {}
    for name, config in get_model_grids().items():
        tscv = TimeSeriesSplit(n_splits)
        grid = GridSearchCV(config['model'], config['param_grid'], verbose=1, cv=tscv, scoring='r2')
        grid.fit(X_train, y_train)
        train_pred = grid.predict(X_train)
        val_pred = grid.predict(X_val)
        grid_results[name] = {
            'best_model': grid.best_estimator_,
            'best_params': grid.best_params_,
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
        }
        print(f"{name:20s} | Train R2: {grid_results[name]['train_r2']:.4f} | Val R2: {grid_results[name]['val_r2']:.4f}")
    best_name = max(grid_results, key=lambda x: grid_results[x]['val_r2'])
    best = grid_results[best_name]
    print(f"\nBest model: {best_name}")
    print(f"  Val R2: {best['val_r2']:.4f}")
    print(f"  Params: {best['best_params']}")
    return grid_results


def plot_permutation_importance(model, X_data, y_data, feature_cols, n_top=6):
    result = permutation_importance(model, X_data, y_data, n_repeats=10, )
    importance = result.importances_mean
    sorted_idx = importance.argsort()
    top_features = [feature_cols[i] for i in sorted_idx[-n_top:]]
    plt.barh([feature_cols[i] for i in sorted_idx], importance[sorted_idx])
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    return top_features


def evaluate_top_model(model, X_train, y_train, X_val, y_val, X_test, y_test, top_features):
    """Re-fit model on top features and calculate R2 scores."""
    X_train_top = X_train[top_features]
    X_val_top = X_val[top_features]
    X_test_top = X_test[top_features]

    model.fit(X_train_top, y_train)

    train_pred = model.predict(X_train_top)
    val_pred = model.predict(X_val_top)
    test_pred = model.predict(X_test_top)

    results = {
        'train_r2': r2_score(y_train, train_pred),
        'val_r2': r2_score(y_val, val_pred),
        'test_r2': r2_score(y_test, test_pred)
    }
    print(f"Train R2: {results['train_r2']:.4f}")
    print(f"Val R2: {results['val_r2']:.4f}")
    print(f"Test R2: {results['test_r2']:.4f}")
    return results


def evaluate_direction(model, X_train, y_train, X_test, y_test):
    """Evaluate if predictions match actual direction."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_dir = np.sign(y_train) == np.sign(train_pred)
    test_dir = np.sign(y_test) == np.sign(test_pred)

    results = {
        'train_direction_accuracy': train_dir.mean(),
        'test_direction_accuracy': test_dir.mean(),
    }

    print(f"Train direction accuracy: {results['train_direction_accuracy']:.4f}")
    print(f"Test direction accuracy: {results['test_direction_accuracy']:.4f}")

    return results