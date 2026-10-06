import argparse
import logging
from datetime import datetime
from logging_setup import setup_logging
from data_pipeline import download_df, df_feature_preparation, create_label_column, split_train_val_test_data
from feature_filtering import remove_correlated_features, calc_permutation_importance, select_important_features
from modeling import run_randomized_search, evaluate_top_model, get_best_by_mae
from baselines import evaluate_baselines

log = logging.getLogger(__name__)

def main():
    args = parse_args()

    # Generate timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(log_file= f"logs/results_{args.ticker}_{args.period}_{timestamp}.log", level=args.log_level)

    TICKER = args.ticker
    PERIOD = args.period
    CORR_THRESHOLD = args.corr_threshold
    LABEL_WINDOW = args.label_window
    N_ITER = args.n_iter
    RANDOM_STATE = args.random_state

    # get dataframe, create features and label column
    df = download_df(TICKER, PERIOD)
    df_feature = df_feature_preparation(df)
    df_feature_labeled = create_label_column(df_feature, window=LABEL_WINDOW)

    # split dataframes in train, val and test data
    # separate target from feature cols
    feature_cols = [col for col in df_feature_labeled.columns if col != 'max_future_return']
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test_data(df_feature_labeled, feature_cols, target_col='max_future_return', gap=LABEL_WINDOW)

    # remove features with correlation over the CORR_THRESHOLD
    # and run the randomized search on the returned features
    X_train_low_corr = remove_correlated_features(X_train, CORR_THRESHOLD, 'spearman')
    search_results = run_randomized_search(X_train_low_corr, y_train, gap=LABEL_WINDOW, n_iter=N_ITER, random_state=RANDOM_STATE)

    # calculate permutation importance and get top features
    # for the best models in the serach_results
    perm_results = calc_permutation_importance(search_results, X_train_low_corr, y_train, train_fract=0.2, gap=LABEL_WINDOW, random_state=RANDOM_STATE)
    top_features = select_important_features(perm_results, feature_limit=None, require_lower=True)

    # evaluate the top models and features on validation data
    # then choose best for final run on test data
    model_results = evaluate_top_model(
        search_results,
        top_features,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        suite='val'
    )
    best_name, best_val = get_best_by_mae(model_results)
    log.info(f'Best on validation: {best_name} (MAE {best_val["mae"]:.6f})')
    log.info(f'\nTest Data Results: ')
    final_test = evaluate_top_model(
        {best_name: search_results[best_name]},
        top_features,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        suite='test'
    )
    # get baseline results to compare the validation models
    # returns a mean, median and zero-return target
    baseline_test = evaluate_baselines(X_train, y_train, X_val, y_val, X_test, y_test)
    best_test_mae = final_test[best_name]["mae"]
    for name, res in baseline_test.items():
        ratio = res["mae"] / best_test_mae
        log.info(f'{name:16s} | MAE {res["mae"]:.6f} | {ratio:.2f}x model error')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--period", default="15y")
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--label-window", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO",)
    return parser.parse_args()
    

if __name__ == "__main__":
    main()