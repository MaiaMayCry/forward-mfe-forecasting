import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

def remove_correlated_features(X_train, threshold, filter_method):
    # drop original OHLCV columns to favor the features made from them
    drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])

    # take only values above the diagonal for threshold verification
    feature_corr = X_train.corr(method=filter_method).abs()
    upper = feature_corr.where(
        np.triu(np.ones(feature_corr.shape), k=1).astype(bool)
    )
    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > threshold)
    ]

    X_train_low_corr = X_train.drop(columns=to_drop)
    print(f"Dropped columns with threshold of {threshold} ({len(to_drop)}): ", to_drop)
    print('-' * 40)
    print(f"Selected Features ({len(X_train_low_corr.columns)}): ", X_train_low_corr.columns)
    print('-' * 40)
    return X_train_low_corr, X_train_low_corr.columns


def calc_permutation_importance(search_results, best_model, X_val, y_val, random_state=42):
    permutation_results = {}

    for name, result in search_results.items():
        perm = permutation_importance(
            estimator=best_model,
            X=X_val,
            y=y_val,
            scoring="neg_mean_absolute_error",
            n_repeats=20,
            random_state=random_state,
            n_jobs=-1,
            max_samples=1.0,
        )
        feature_names = list(X_val.columns)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })

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
        }
    print(importance_df)
    return importance_df


def select_important_features(importance_df, n_top=None, require_lower=True):
    if require_lower:
        mask = (
            (importance_df["importance_mean"] > 0) &
            (importance_df["importance_lower"] > 0)
        )
    else:
        mask = importance_df["importance_mean"] > 0

    filtered = importance_df.loc[mask].sort_values('importance_mean', ascending=False)
    
    if n_top is not None:
        filtered = filtered.head(n_top)
    
    return filtered['feature'].tolist()