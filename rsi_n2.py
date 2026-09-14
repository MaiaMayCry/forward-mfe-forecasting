import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

def download_df(ticker, period):
    """get historical data for the ticker arg"""
    df = yf.download(ticker, period=period).droplevel('Ticker', axis=1)
    if df.empty:
        raise ValueError("No data returned for this ticker/period")
    else:
        return df


def df_rsi_preparation(ticker, long_threshold, short_threshold, period='2y'):
    """compute RSI features and signals for long and short positions"""
    df = download_df(ticker, period)
    df['Returns_pct'] = df.Close.pct_change()
    df['Volume_pct'] = df.Volume.pct_change(10)
    df['Intraday'] = df.Close - df.Open
    df['Gap'] = df.Open - df.Close.shift(1)
    df['2d_Gap'] = df.Close - df.Open.shift(1)
    Upmove = df.Returns_pct.apply(lambda x: x if x > 0 else 0)
    Downmove = df.Returns_pct.apply(lambda x: abs(x) if x < 0 else 0)
    df['Avg_up'] = Upmove.ewm(span=19).mean()
    df['Avg_down'] = Downmove.ewm(span=19).mean()
    df.dropna(inplace=True)
    RS = df.Avg_up / df.Avg_down
    df['RSI'] = RS.apply(lambda x: (100 - (100 / (x + 1))) / 100)
    df['Long'] = np.where(df.RSI < long_threshold, 1, 0)
    df['Short'] = np.where(df.RSI > short_threshold, 1, 0)
    for d in [19, 59, 119]:
        df[f'Close_lag_{d + 1}'] = df['Close'].shift(d)
    return df


def find_entry_and_exit_dates(df, exit_threshold, signal_type, lookahead=10):
    """returns two lists with next day entry and exit dates using RSI thresholds."""
    entry_days = []
    exit_days = []
    is_long = signal_type == 'Long'

    # if value in 'Long'/'Short' column equals True, open a position the next day
    for i in range(len(df) - 1):
        if df[signal_type].iloc[i]:
            entry_days.append(df.iloc[i + 1].name)
            # check RSI over/under threshold and close the position the next day or after the 'lookahead' period
            for j in range(1, lookahead  + 1):
                # check bounds before accessing
                if i + j < (len(df) - 1):  
                    rsi_above = df.RSI.iloc[i + j] > exit_threshold
                    rsi_below = df.RSI.iloc[i + j] < exit_threshold
                    if (is_long and rsi_above) or (not is_long and rsi_below):
                        exit_days.append(df.iloc[i + j + 1].name)
                        break
                else:
                    print(f"can't find exit day at index: {i} out of bounds at: {i} + {j}")
                    exit_days.append(df.iloc[-1].name)
                    break
    return entry_days, exit_days


def create_label_column(df, entry_days, exit_threshold, signal_type, lookahead=10):
    """create column of position outcomes for long and short positions"""
    df[f'l_{signal_type}'] = 0
    is_long = signal_type == 'Long'

    # check if there's enough data after each entry date
    for day in entry_days:
        idx = df.index.get_loc(day)
        if idx + lookahead > len(df):
            print(f'Not enough forward data at index: {idx}')
            continue

        # give label the value of 1 if position was profitable by the end of the lookahead period
        entry_price = df.iloc[idx]['Close']
        exit_prices = df.iloc[idx+1:idx+1+lookahead]['Close']
        if is_long:
            profitable = exit_prices.max() > entry_price
        else:
            profitable = exit_prices.min() < entry_price
        
        if profitable:
            df.loc[day, f'l_{signal_type}'] = 1
            
    return df


def calculate_profits(df, entry_days, exit_days, signal_type):
    """returns for long or short positions, keeping one open position at a time"""
    profits = []
    i, j = 0, 0
    is_long = signal_type == 'Long'

    while i < len(entry_days) and j < len(exit_days):
        entry_date  = entry_days[i]

        # find the first exit date that occurs AFTER the entry
        while j < len(exit_days) and exit_days[j] <= entry_date:
            j += 1
        if j >= len(exit_days):
            break
        exit_date = exit_days[j]

        # the prices are already the next day's 'Open' after previous 'Close' whithin RSI thresholds
        entry_price = df.loc[entry_date, 'Open']
        exit_price = df.loc[exit_date, 'Open']
        if is_long:
            profit = (exit_price - entry_price) / entry_price
        else:
            profit = (entry_price - exit_price) / entry_price
        profits.append(profit.item())

        # move to next entry that occurs AFTER this exit
        i += 1
        while i < len(entry_days) and entry_days[i] <= exit_date:
            i += 1
        j += 1
    return profits


def calculate_win_rate(profits):
    """calculate win rate from a list of profit percentages."""
    wins = [p for p in profits if p > 0]
    print(f"wins: {len(wins)} / total length: {len(profits)}")
    return len(wins) / len(profits)


def plot_scatter_entry_dates(df, signals):
    """create a scatter plot from the given dataframe, with markers for entry dates"""
    plt.figure(figsize=(12,5))
    plt.scatter(df.loc[signals].index, df.loc[signals]['Close'], marker='^', c='g')
    plt.plot(df['Close'], alpha=0.7)
    plt.show()


if __name__ == '__main__':
    ticker = 'MSFT'
    df = df_rsi_preparation(ticker, long_threshold=0.3, short_threshold=0.7, period='10y')

    short_entry_days, short_exit_days = find_entry_and_exit_dates(df, exit_threshold=0.6, signal_type='Short')
    short_profits = calculate_profits(df, short_entry_days, short_exit_days, signal_type='Short')
    df = create_label_column(df, short_entry_days, exit_threshold=0.6, signal_type='Short')
    plot_scatter_entry_dates(df, short_entry_days)
    short_win_rate = calculate_win_rate(short_profits)
    print(f'Short result: {short_win_rate}')


    long_entry_days, long_exit_days = find_entry_and_exit_dates(df, exit_threshold=0.4, signal_type='Long')
    long_profits = calculate_profits(df, long_entry_days, long_exit_days, signal_type='Long')
    df = create_label_column(df, long_entry_days, exit_threshold=0.4, signal_type='Long')
    plot_scatter_entry_dates(df, long_entry_days)
    long_win_rate = calculate_win_rate(long_profits)
    print(f'Long result: {long_win_rate}')


    combined_lists = short_profits + long_profits
    combined_result = calculate_win_rate(combined_lists)
    print(f'Combined result: {combined_result}')