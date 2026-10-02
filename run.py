from rsi_data import download_df, df_feature_preparation, create_label_column, split_train_val_test_data
from feature_filtering import remove_correlated_features, calc_permutation_importance, select_important_features
from model_comparison import run_randomized_search, evaluate_top_model, get_best_by_mae

TICKER = 'AAPL'
PERIOD = '15y'
CORR_THRESHOLD = 0.95
LABEL_WINDOW = 5

def main():    
    df = download_df(TICKER, PERIOD)
    df_rsi = df_feature_preparation(df)
    df_rsi_labeled = create_label_column(df_rsi)

    feature_cols = [col for col in df_rsi_labeled.columns if col != 'label']
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test_data(df_rsi_labeled, feature_cols, target_col='label', gap=LABEL_WINDOW)

    X_train_low_corr = remove_correlated_features(X_train, CORR_THRESHOLD, 'spearman')
    search_results = run_randomized_search(X_train_low_corr, y_train, gap=LABEL_WINDOW)

    permutation_results = calc_permutation_importance(search_results, X_train_low_corr, y_train, train_fract=0.2, gap=LABEL_WINDOW)
    top_features = select_important_features(permutation_results, n_top=None, require_lower=True)

    model_results = evaluate_top_model(
        search_results,
        top_features,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        gap=LABEL_WINDOW,
        suite='val'
    )
    # select the winner on validation
    best_name, best_val = get_best_by_mae(model_results)
    print(f'\nBest on validation: {best_name} (MAE {best_val["mae"]:.6f})')
    evaluate_top_model(
        {best_name: search_results[best_name]},
        top_features,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        gap=LABEL_WINDOW,
        suite='test'
    )
    

if __name__ == "__main__":
    main()