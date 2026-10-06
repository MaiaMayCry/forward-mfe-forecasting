import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.base import clone

def remove_correlated_features(X_train, threshold, filter_method):
    """remove features that have a correlation over the threshold"""
    # drop original OHLCV columns to favor the features made from them
    drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    X_train = X_train.drop(columns=drop_cols)

    feature_corr = X_train.corr(method=filter_method).abs()

    # take only values above the diagonal for threshold verification
    upper = feature_corr.where(
        np.triu(np.ones(feature_corr.shape), k=1).astype(bool)
    )
    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > threshold)
    ]

    X_train_low_corr = X_train.drop(columns=to_drop)
    print('-' * 40)
    print(f"Dropped columns with threshold of {threshold} ({len(to_drop)}): ", to_drop)
    print('-' * 40)
    print(f"Selected Features ({len(X_train_low_corr.columns)}): ", X_train_low_corr.columns)
    print('-' * 40)
    return X_train_low_corr


def calc_permutation_importance(search_results, X_train, y_train, train_fract=0.2, gap=5, random_state=42):
    """calculate permutation on given features on a fit/selection split of the train data"""
    permutation_results = {}
    fract_len = int(train_fract * len(X_train))

    # split train data for fit and permutation
    X_fit = X_train.iloc[:fract_len]
    y_fit = y_train.iloc[:fract_len]

    X_selection = X_train.iloc[fract_len + gap:]
    y_selection = y_train.iloc[fract_len + gap:]

    for name, result in search_results.items():
        # clone the best estimator so that the original search result is
        # not modified and each model is fitted only on X_fit.
        model = clone(result["best_model"])
        model.fit(X_fit, y_fit)
        perm = permutation_importance(
            estimator=model,
            X=X_selection,
            y=y_selection,
            scoring="neg_mean_absolute_error",
            n_repeats=20,
            random_state=random_state,
            n_jobs=5,
            max_samples=1.0,
        )

        importance_df = pd.DataFrame({
            "feature": X_selection.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })

        # filter out features with apparent low contribution to the predictions
        importance_df["importance_lower"] = (
            importance_df["importance_mean"] - 2 * importance_df["importance_std"]
        )

        importance_df = importance_df.sort_values(
            "importance_mean",
            ascending=False,
        ).reset_index(drop=True)

        permutation_results[name] = {
            "importance": perm,
            "importance_df": importance_df,
            "model": model,
            "fit_rows": len(X_fit),
            "selection_rows": len(X_selection),
        }
        print(f"\n{name}:")
        print(importance_df.head(10))
    return permutation_results


def select_important_features(permutation_results, feature_limit=None, require_lower=True):
    """select the features from the permutation to be used in the final model training"""
    top_features = {}
    for name, data in permutation_results.items():
        importance_df = data["importance_df"]

        # if true select only features with importance_lower greater than 0
        if require_lower:
            mask = (
                (importance_df["importance_mean"] > 0) &
                (importance_df["importance_lower"] > 0)
            )
        else:
            mask = importance_df["importance_mean"] > 0

        filtered = importance_df.loc[mask].sort_values('importance_mean', ascending=False)
        
        # set a limit for the number of features returned
        if feature_limit is not None:
            filtered = filtered.head(feature_limit)

        top_features[name] = filtered['feature'].tolist()
        print('-' * 40)
        if not top_features[name]:
            print(f"{name}: no feature passed the threshold, falling back to all features")
        else:
            print(f"{name} ({len(top_features[name])} features): {top_features[name]}")
    
    return top_features