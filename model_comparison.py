from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
import pandas as pd

from model_grid import get_model_grids


def run_randomized_search(X_train, y_train, gap, n_splits=5, n_iter=50, random_state=42):
    search_results = {}
    tscv = TimeSeriesSplit(n_splits, gap=gap)
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
        train_mae = -mean_absolute_error(y_train, train_pred)
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

    best_name = max(
        search_results,
        key=lambda name: search_results[name]["best_cv_score"]
    )
    best = search_results[best_name]

    print(
        f"\nSelected model: {best_name}"
        f"  Best CV score: {best['best_cv_score']:.6f}"
        f"  Params: {best['best_params']}"
    )
    # return search results dictionary and the best model found
    return search_results, best['best_model']


def evaluate_top_model(model, X_train, y_train, X_predict, y_predict, top_features, predict):
    # select features if provided
    if top_features is not None:
        X_train_selected = X_train.loc[:, top_features]
        X_predict_selected = X_predict.loc[:, top_features]
    else:
        X_train_selected = X_train
        X_predict_selected = X_predict

    
    if predict == 'val':
        # fit train data and predict validation data
        validation_model = clone(model)
        validation_model.fit(X_train_selected, y_train)
        result_pred = validation_model.predict(X_predict_selected)

    if predict == 'test':
        # combine training and validation data
        X_train_val = pd.concat([X_train_selected, X_predict_selected],axis=0)
        y_train_val = pd.concat([y_train, y_predict],axis=0)

        # take train + validation then predict test data
        final_model = clone(model)
        final_model.fit(X_train_val, y_train_val)
        result_pred = final_model.predict(X_test_top)

    print('-' * 40)
    print(f'NEG MAE: {-mean_absolute_error(y_predict, result_pred)}')
    print('-' * 40)
    return result_pred



'''
def evaluate_direction(model, X_train, y_train, X_test, y_test):
    """Evaluate if predictions match actual direction."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Convert log returns to real returns
    y_train_real = np.exp(y_train) - 1
    y_test_real = np.exp(y_test) - 1
    train_pred_real = np.exp(train_pred) - 1
    test_pred_real = np.exp(test_pred) - 1


    train_dir = np.sign(y_train) == np.sign(train_pred)
    test_dir = np.sign(y_test) == np.sign(test_pred)

    results = {
        'train_direction_accuracy': train_dir.mean(),
        'test_direction_accuracy': test_dir.mean(),
        'train_diff_realpred': train_pred_real,
        'test_diff_realpred': test_pred_real
    }

    print(f"Train direction accuracy: {results['train_direction_accuracy']:.4f}")
    print(f"Test direction accuracy: {results['test_direction_accuracy']:.4f}")
    #print(results['train_diff_realpred'])
    #print(results['test_diff_realpred'])

    return results
'''