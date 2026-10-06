import yfinance as yf
import numpy as np
import talib as ta

def download_df(ticker, period):
    """get historical data for the ticker arg over the given period"""
    df = yf.download(ticker, period=period).droplevel('Ticker', axis=1)
    print('-' * 40)
    print(f"length of returned dataframe: {len(df)}")
    print('-' * 40)
    print(f"sum of nan values across dataframe: {df.isna().sum().sum()}")
    print('-' * 40)
    print(f"sum of zeroes across dataframe: {(df == 0).sum().sum()}")
    print('-' * 40)
    if df.empty:
        raise ValueError("No data returned for this ticker/period")
    else:
        return df


def df_feature_preparation(df):
    """compute features from the data for the model"""
    df = df.copy()

    open_price = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    log_close = np.log(close)

    macd, macd_signal, macd_hist = ta.MACD(
        close,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    ema_9 = ta.EMA(close, timeperiod=9)
    ema_12 = ta.EMA(close, timeperiod=12)
    ema_26 = ta.EMA(close, timeperiod=26)

    sma_9 = ta.SMA(close, timeperiod=9)
    sma_12 = ta.SMA(close, timeperiod=12)
    sma_26 = ta.SMA(close, timeperiod=26)

    ema_vars = {9: ema_9, 12: ema_12, 26: ema_26}
    sma_vars = {9: sma_9, 12: sma_12, 26: sma_26}

    # Moving-average relationships
    for val in [9, 12, 26]:
        df[f"Close_to_EMA_{val}"] = close / ema_vars[val] - 1
        df[f"Close_to_SMA_{val}"] = close / sma_vars[val] - 1
        df[f"EMA_{val}_change"] = ema_vars[val].pct_change()

    df["EMA_9_to_EMA_12"] = ema_9 / ema_12 - 1
    df["EMA_9_to_EMA_26"] = ema_9 / ema_26 - 1
    df["EMA_12_to_EMA_26"] = ema_12 / ema_26 - 1

    df["SMA_9_to_SMA_12"] = sma_9 / sma_12 - 1
    df["SMA_9_to_SMA_26"] = sma_9 / sma_26 - 1
    df["SMA_12_to_SMA_26"] = sma_12 / sma_26 - 1

    # MACD features
    df['MACD_relative'] = macd / ema_26
    df["MACD_signal_relative"] = macd_signal / ema_26
    df["MACD_histogram_relative"] = macd_hist / ema_26
    df["MACD_slope"] = macd.diff() / ema_26
        
    # RSI features
    for val in [5, 9, 14, 22]:
        df[f"RSI_{val}"] = ta.RSI(close, timeperiod=val) / 100

    df["RSI_14_above_70"] = (df["RSI_14"] > 0.70).astype(int)
    df["RSI_14_below_30"] = (df["RSI_14"] < 0.30).astype(int)
    df["RSI_14_change"] = df["RSI_14"].diff()

    # Log returns over lookback periods
    for val in [1, 3, 5, 10]:
        df[f"LogReturns_{val}"] = log_close.diff(val)

    # Ordinary percentage returns over lookback periods
    for val in [3, 5, 10, 15]:
        df[f"PctReturn_{val}"] = close.pct_change(val)

    # Candle features
    df["Range_relative"] = (high - low) / open_price
    df["Body_relative"] = (close - open_price) / open_price

    # OHLC log differences
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"{column}_log_diff"] = np.log(df[column]).diff()

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def create_label_column(df, window=5):
    """highest return during the next five trading days, relative to entry price"""
    entry_price = df["Open"].shift(-1)
    future_max = (
        df['High']
        .iloc[::-1]
        .rolling(window=window, min_periods=1)
        .max()
        .iloc[::-1]
        .shift(-1)
    )
    df["max_future_return"] = future_max / entry_price - 1
    return df.dropna(subset=["max_future_return"])


def split_train_val_test_data(df, feature_cols, target_col, train_size=0.7, val_test_size=0.15, gap=5):
    """split dataframe into train, val, and test sets"""
    train_len = int(train_size * len(df))
    val_test_len = int(val_test_size * len(df))

    # remove rows that would cause data leakage between splits
    train_df= df.iloc[:train_len - gap]
    val_df = df.iloc[train_len:train_len + val_test_len - gap]
    test_df = df.iloc[train_len + val_test_len:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print(
        f"train: {train_df.index.min()} -> {train_df.index.max()} "
        f"({len(train_df)} rows)"
    )
    print(
        f"validation: {val_df.index.min()} -> {val_df.index.max()} "
        f"({len(val_df)} rows)"
    )
    print(
        f"test: {test_df.index.min()} -> {test_df.index.max()} "
        f"({len(test_df)} rows)"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test