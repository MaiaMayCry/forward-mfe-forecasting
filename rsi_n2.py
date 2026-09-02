import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

def download_df(ticker, period):
    """get historical data for the ticker arg"""
    df = yf.download(ticker, period=period)
    if df.empty:
        raise ValueError("No data returned for this ticker/period")
    else:
        return df


def df_rsi_preparation(ticker, period='2y'):
    """compute RSI features and signals for long and short positions"""
    df = download_df(ticker, period)
    #df['MA200'] = df.Close.rolling(window=200).mean()
    df['Returns_pct'] = df.Close.pct_change()
    df['Upmove'] = df.Returns_pct.apply(lambda x: x if x > 0 else 0)
    df['Downmove'] = df.Returns_pct.apply(lambda x: abs(x) if x < 0 else 0)
    df['Avg_up'] = df.Upmove.ewm(span=19).mean()
    df['Avg_down'] = df.Downmove.ewm(span=19).mean()
    df.dropna(inplace=True)
    df['RS'] = df.Avg_up / df.Avg_down
    df['RSI'] = df.RS.apply(lambda x: 100 - (100 / (x + 1)))
    df['Long'] = np.where(df.RSI < 30, True, False)
    df['Short'] = np.where(df.RSI > 70, True, False)
    return df


def get_signals(df, exit_threshold, signal_type, lookahead=10):
    """Find entry and exit dates for long or short positions using RSI tresholds."""
    entry_days = []
    exit_days = []
    is_long = signal_type == 'Long'

    # if value in 'Long' column equals True, open a position the next day
    for i in range(len(df) - 1):
        if df[signal_type].iloc[i]:
            entry_days.append(df.iloc[i + 1].name)
            # Check for RSI over 40 and sell the next day or close the position after 10 days
            for j in range(1, lookahead  + 1):
                # Check bounds before accessing
                if i + j < (len(df) - 1):  
                    rsi_above = df.RSI.iloc[i + j] > exit_threshold
                    rsi_below = df.RSI.iloc[i + j] < exit_threshold
                    if (is_long and rsi_above) or (not is_long and rsi_below):
                        exit_days.append(df.iloc[i + j + 1].name)
                        break
                else:
                    print(f'index i: {i} is out of bounds at: {i} + {j}')
                    exit_days.append(df.iloc[-1].name)
                    break
    return entry_days, exit_days


def calculate_profits(df, entry_days, exit_days, signal_type):
    """returns for the respective long or short positions, keeping one position at a time"""
    profits = []
    i, j = 0, 0
    is_long = signal_type == 'Long'

    while i < len(entry_days) and j < len(exit_days):
        entry_date  = entry_days[i]

        # Find the first exit date that occurs AFTER the entry
        while j < len(exit_days) and exit_days[j] <= entry_date:
            j += 1
        if j >= len(exit_days):
            break
        exit_date = exit_days[j]

        # Calculate profit
        entry_price = df.loc[entry_date, 'Open']
        exit_price = df.loc[exit_date, 'Open']
        if is_long:
            profit = (exit_price - entry_price) / entry_price
        else:
            profit = (entry_price - exit_price) / entry_price
        profits.append(profit.item())

        # Move to next entry that occurs AFTER this exit
        i += 1
        while i < len(entry_days) and entry_days[i] <= exit_date:
            i += 1
        j += 1
    return profits


def plot_scatter_entry_dates(df, signals):
    """create a scatter plot from the given dataframe, with markers for entry dates"""
    plt.figure(figsize=(12,5))
    plt.scatter(df.loc[signals].index, df.loc[signals]['Close'], marker='^', c='g')
    plt.plot(df['Close'], alpha=0.7)
    plt.show()


def calculate_win_rate(profits):
    """Calculate win rate from a list of profit percentages."""
    wins = [p for p in profits if p > 0]
    return len(wins) / len(profits)




ticker = 'MSFT'
df = df_rsi_preparation(ticker, period='3y')

short_entry_days, short_exit_days = get_signals(df, exit_threshold=60, signal_type='Short')
short_profits = calculate_profits(df, short_entry_days, short_exit_days, signal_type='Short')
plot_scatter_entry_dates(df, short_entry_days)
short_win_rate = calculate_win_rate(short_profits)
print(f'Short result: {short_win_rate}')

long_entry_days, long_exit_days = get_signals(df, exit_threshold=40, signal_type='Long')
long_profits = calculate_profits(df, long_entry_days, long_exit_days, signal_type='Long')
plot_scatter_entry_dates(df, long_entry_days)
long_win_rate = calculate_win_rate(long_profits)
print(f'Long result: {long_win_rate}')

combined_lists = short_profits + long_profits
combined_result = calculate_win_rate(combined_lists)
print(f'Combined result: {combined_result}')