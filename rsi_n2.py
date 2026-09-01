import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

def download_df(ticker):
    """get historical data for the ticker arg"""
    df = yf.download(ticker, period='10y')
    if df.empty:
        raise ValueError("No data returned for this ticker/period")
    else:
        return df

def df_rsi_preparation(ticker):
    """compute RSI features and signals for long and short positions"""
    df = download_df(ticker)
    df['MA200'] = df.Close.rolling(window=200).mean()
    df['Returns_pct'] = df.Close.pct_change()
    df['Upmove'] = df.Returns_pct.apply(lambda x: x if x > 0 else 0)
    df['Downmove'] = df.Returns_pct.apply(lambda x: abs(x) if x < 0 else 0)
    df['Avg_up'] = df.Upmove.ewm(span=19).mean()
    df['Avg_down'] = df.Downmove.ewm(span=19).mean()
    df.dropna(inplace=True)
    df['RS'] = df.Avg_up / df.Avg_down
    df['RSI'] = df.RS.apply(lambda x: 100 - (100 / (x + 1)))
    df['Buy'] = np.where(df.RSI < 30, True, False)
    df['Short'] = np.where(df.RSI > 70, True, False)
    return df

def getSignals(df):
    """use buy signals to find the nearest sell that matches the RSI criteria"""
    buying_days = []
    selling_days = []

    # if value in 'Buy' column equals True, open a position the next day
    for i in range(len(df) - 1):
        if df['Buy'].iloc[i]:
            buying_days.append(df.iloc[i + 1].name)
            # Check for RSI over 40 and sell the next day or close the position after 10 days
            for j in range(1, 11):
                # Check bounds before accessing
                if i + j < (len(df) - 1):  
                    if df.RSI.iloc[i + j] > 40:
                        selling_days.append(df.iloc[i + j + 1].name)
                        break
                else:
                    selling_days.append(df.iloc[-1].name)
                    print(f'Buy at index {i} is out of bounds at: {i} + {j}')
                    break
    return buying_days, selling_days

def plot_scatter_buy(df, buy):
    """create a scatter plot from the given dataframe, with markers for buy dates"""
    plt.figure(figsize=(12,5))
    plt.scatter(df.loc[buy].index, df.loc[buy]['Close'], marker='^', c='g')
    plt.plot(df['Close'], alpha=0.7)
    plt.show()


def calculate_profits(df, buying_days, selling_days):
    """returns for the respective buy and sell pairs, keeping only one position at a time"""
    profits = []
    i, j = 0, 0
    while i < len(buying_days) and j < len(selling_days):
        buy_date = buying_days[i]

        # Find the first sell that occurs AFTER the buy
        while j < len(selling_days) and selling_days[j] <= buy_date:
            j += 1
        if j >= len(selling_days):
            break
        sell_date = selling_days[j]

        # Calculate profit for this pair
        buy_price = df.loc[buy_date, 'Open']
        sell_price = df.loc[sell_date, 'Open']
        profit = (sell_price - buy_price) / buy_price
        profits.append(profit.item())

        # Move to next buy that occurs AFTER this sell
        i += 1
        while i < len(buying_days) and buying_days[i] <= sell_date:
            i += 1
        j += 1
    return profits

def calculate_win_rate(profits):
    wins = [i for i in profits if i > 0]
    return len(wins)/len(profits)

ticker = 'MSFT'
df = df_rsi_preparation(ticker)
buy, sell = getSignals(df)
#plot_scatter_buy(df, buy)
profits = calculate_profits(df, buy, sell)
print(calculate_win_rate(profits))