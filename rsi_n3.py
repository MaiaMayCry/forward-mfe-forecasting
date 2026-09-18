from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
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
    df['RSI_7'] = ta.RSI(df['Close'], timeperiod=7) / 100
    df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14) / 100
    df['S_RSI_7'] = ta.EMA(df['RSI_14'], timeperiod=14)
    df['EMA_9'] = ta.EMA(df['Close'], timeperiod=9)
    df['EMA_21'] = ta.EMA(df['Close'], timeperiod=21)
    df["Returns"] = df['Close'].shift().rolling(14).sum()
    df.dropna(inplace=True)
    print(df)
    return df


def create_label_column(df, col='Close', window=5):
    df[f'label'] = df[col].shift(-1).rolling(window=window, min_periods=1).max() - df[col]
    df.dropna(inplace=True)
    return df


def splt_train_val_test_data(df, feature_cols, target_col, train_size=0.7, val_test_size=0.15):
    """Split dataframe into train, val, and test sets"""

    train_len = int(train_size * len(df))
    val_test_len = int(val_test_size * len(df))

    train_df= df[:train_len]
    val_df = df[train_len:train_len + val_test_len]
    test_df = df[train_len + val_test_len:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test


def rf_grid_search(X_train, y_train, param_grid, class_weights=None, n_splits=10):
    tscv = TimeSeriesSplit(n_splits)
    grid_search = GridSearchCV(
        XGBRegressor(),
        param_grid,
        verbose=1,
        cv=tscv,
        scoring='r2'
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(f'Best parameters: {best_params}')
    return best_params, best_model


def get_permutation_importance(model, X_data, y_data, feature_cols):
    feature_names = np.array(feature_cols)
    result = permutation_importance(
        model,
        X_data,
        y_data,
        scoring="r2",
        n_repeats=20,
    )
    importance = result.importances_mean
    sorted_idx = importance.argsort()
    top_features = [feature_cols[i] for i in sorted_idx[-6:]]
    #top_features = feature_cols #**
    plt.barh(feature_names[sorted_idx], importance[sorted_idx])
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    return top_features


if __name__ == '__main__':
    ticker = 'MSFT'
    period = '15y'
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5,],
        'learning_rate': [0.01, 0.05, 0.1,],
        'subsample': [0.8, 1.0,],
        'colsample_bytree': [0.8, 1.0,],
        'min_child_weight': [1, 5,],
    }
    df = download_df(ticker, period)
    df_rsi = df_rsi_preparation(df)
    df_rsi_max = create_label_column(df_rsi)

    feature_cols = [col for col in df_rsi_max.columns if col != 'label']
    X_train, y_train, X_val, y_val, X_test, y_test = splt_train_val_test_data(df_rsi_max, feature_cols, 'label')

    print(X_train)

    best_params, best_model = rf_grid_search(X_train, y_train, param_grid, n_splits=5)
    top_features = get_permutation_importance(best_model, X_train, y_train, feature_cols)

    model = XGBRegressor(reg_alpha=0, reg_lambda=1, **best_params)
    model.fit(X_train[top_features], y_train)

    train_pred = model.predict(X_train[top_features])
    train_r2 = r2_score(y_train, train_pred)

    val_pred = model.predict(X_val[top_features])
    val_r2 = r2_score(y_val, val_pred)

    print(f"Train R2: {train_r2:.4f}")
    print(f"Val R2: {val_r2:.4f}")