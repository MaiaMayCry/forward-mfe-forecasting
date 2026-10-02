from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
import pandas as pd

from model_configs import get_model_grids

def run_randomized_search(X_train, y_train, gap, n_splits=5, n_iter=50, random_state=42):
    """do a randomized search through the param_grids from the models"""
    search_results = {}
    tscv = TimeSeriesSplit(n_splits, gap=gap)
    
    # get list of models and param grid from the model_grid.py 
    for name, config in get_model_grids().items():
        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["param_grid"],
            n_iter=n_iter,
            verbose=1,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=random_state,
            refit=True,
        )
        search.fit(X_train, y_train)
        train_pred = search.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        search_results[name] = {
            "best_model": search.best_estimator_,
            "best_params": search.best_params_,
            "best_cv_score": search.best_score_,
            "train_mae": train_mae,
            "cv_results": search.cv_results_,
        }

        print(
            f"{name:20s} | "
            f"Best CV score: {search.best_score_:.6f} | "
            f"Train MAE: {train_mae:.6f} | "
        )

    # Sort all models by CV score (descending) and take top 4
    sorted_results = sorted(
        search_results.items(),
        key=lambda item: item[1]["best_cv_score"],
        reverse=True
    )
    top_4 = dict(sorted_results[:4])

    # Access them:
    for name, result in top_4.items():
        print(f"{name}: CV={result['best_cv_score']:.6f}, Params={result['best_params']}")
    # return search results dictionary and the best models found
    return top_4


def evaluate_top_model(search_results, top_features, X_train, y_train, X_test, y_test, X_val, y_val, gap=5, suite='test'):
    """train the models in the search_results with the given top_features on either the validation or test data"""
    results = {}

    for name, result in search_results.items():
        model = result["best_model"]
        features = (top_features or {}).get(name) or None

        # select features if provided
        if features is None:
            X_tr, X_te, X_va = X_train, X_test, X_val
        else:
            X_tr = X_train.loc[:, features]
            X_te = X_test.loc[:, features]
            X_va = X_val.loc[:, features]

        
        if suite == 'val':
            # fit train data and predict validation data
            validation_model = clone(model)
            validation_model.fit(X_tr, y_train)
            result_pred = validation_model.predict(X_va)
            y_true = y_val

        elif suite == 'test':
            # combine training and validation data
            X_train_val = pd.concat([X_tr, X_va],axis=0)
            y_train_val = pd.concat([y_train, y_val],axis=0)

            # take train + validation then predict test data
            test_model = clone(model)
            test_model.fit(X_train_val, y_train_val)
            result_pred = test_model.predict(X_te)
            y_true = y_test
        else:
            raise ValueError(f"suite must be 'val' or 'test', got {suite}")

        mae = mean_absolute_error(y_true, result_pred)
        results[name] = {"pred": result_pred, "mae": mae}

        print('-' * 40)
        print(f'{name} | MAE: {mae:.6f} | n_features: {X_tr.shape[1]}')
        print('-' * 40)

    return results

def get_best_by_mae(results):
    """Return the (name, result) pair with the lowest MAE."""
    if not results:
        raise ValueError("results is empty, nothing to select from")
    best_name = min(results, key=lambda name: results[name]["mae"])
    return best_name, results[best_name]