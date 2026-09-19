from model_comparison import run_grid_search, plot_permutation_importance, evaluate_top_model, evaluate_direction
import yfinance as yf
import numpy as np
import talib as ta



def download_df(ticker, period):
    """get historical data for the ticker arg"""
    df = yf.download(ticker, period=period).droplevel('Ticker', axis=1)
    if df.empty:
        raise ValueError("No data returned for this ticker/period")
    else:
        return df


def df_rsi_preparation(df):
    """compute RSI features and signals for long and short positions"""
    df['Close'] = np.log(df['Close']).diff()
    df['High'] = np.log(df['High']).diff()
    df['Low'] = np.log(df['Low']).diff()
    df['Open'] = np.log(df['Open']).diff()
    df['Volume'] = np.log(df['Volume']).diff()
    df['MACD'], df['MACD_Signal'], _  = ta.MACD(df['Close'])
    df['RSI_5'] = ta.RSI(df['Close'], timeperiod=5) / 100
    df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14) / 100
    df['S_RSI_5'] = ta.EMA(df['RSI_14'], timeperiod=5)
    df['EMA_15'] = ta.EMA(df['Close'], timeperiod=15)
    df['EMA_9'] = ta.EMA(df['Close'], timeperiod=9)
    df['EMA_21'] = ta.EMA(df['Close'], timeperiod=21)
    df['EMA_100'] = ta.EMA(df['Close'], timeperiod=200)
    df["Returns"] = df['Close'].shift().rolling(5).sum()
    df.dropna(inplace=True)
    return df


def create_label_column(df, col='Close', window=5):
    """takes 'window' of rows and subtracts the highest 'Close' to calculate profit"""
    df[f'label'] = df[col].shift(-1).rolling(window=window, min_periods=1).max() - df[col]
    df.dropna(inplace=True)
    return df


def split_train_val_test_data(df, feature_cols, target_col, train_size=0.7, val_test_size=0.15):
    """split dataframe into train, val, and test sets"""
    train_len = int(train_size * len(df))
    val_test_len = int(val_test_size * len(df))

    train_df= df[:train_len]
    val_df = df[train_len:train_len + val_test_len]
    test_df = df[train_len + val_test_len:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    ticker = 'NVDA'
    period = '15y'
    df = download_df(ticker, period)
    df_rsi = df_rsi_preparation(df)
    df_rsi_max = create_label_column(df_rsi)

    feature_cols = [col for col in df_rsi_max.columns if col != 'label']
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test_data(df_rsi_max, feature_cols, 'label')

    results = run_grid_search(X_train, y_train, X_val, y_val)
    best_name = max(results, key=lambda x: results[x]['val_r2'])
    best_model = results[best_name]['best_model']

    top_features = plot_permutation_importance(best_model, X_train, y_train, feature_cols)
    final_results = evaluate_top_model(best_model, X_train, y_train, X_val, y_val, X_test, y_test, top_features)

    direction_results = evaluate_direction(best_model, X_train[top_features], y_train, X_test[top_features], y_test)