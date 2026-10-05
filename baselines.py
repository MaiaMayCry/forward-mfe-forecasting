import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error


def get_baseline_models():
    return {
        "zero_return": DummyRegressor(
            strategy="constant",
            constant=0.0,
        ),
        "mean_target": DummyRegressor(
            strategy="mean",
        ),
        "median_target": DummyRegressor(
            strategy="median",
        ),
    }


def evaluate_baselines(X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}
    # combine train + val so baselines fit on the same rows the winner did
    X_fit = pd.concat([X_train, X_val], axis=0)
    y_fit = pd.concat([y_train, y_val], axis=0)

    for name, model in get_baseline_models().items():
        model.fit(X_fit , y_fit)
        predictions = model.predict(X_test)

        results[name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, predictions),
            "predictions": predictions,
        }

    return results