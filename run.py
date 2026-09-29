from rsi_data import download_df, df_feature_preparation, create_label_column, split_train_val_test_data
from feature_filtering import remove_correlated_features, calc_permutation_importance, select_important_features
from model_comparison import run_randomized_search, evaluate_top_model

def main():
    ticker = 'MSFT'
    period = '15y'
    
    df = download_df(ticker, period)
    df_rsi = df_feature_preparation(df)
    df_rsi_labeled = create_label_column(df_rsi)

    feature_cols = [col for col in df_rsi_labeled.columns if col != 'label']
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test_data(df_rsi_labeled, feature_cols, 'label')

    X_train_low_corr,  features_low_corr = remove_correlated_features(X_train, 1, 'spearman')
    search_results, best_model = run_randomized_search(X_train_low_corr, y_train, gap=5)

    importance_df = calc_permutation_importance(search_results, best_model, X_val[features_low_corr], y_val, random_state=42)
    top_features = select_important_features(importance_df, n_top=None, require_lower=True)

    final_results = evaluate_top_model(best_model, X_train, y_train, X_val, y_val, top_features, predict='val')

if __name__ == "__main__":
    main()