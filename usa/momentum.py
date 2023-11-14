import pandas as p
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


# Fetch historical data for all stocks in the S&P 500
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, verify=False)


    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", {"class": "wikitable"})
        tickers = [row.find_all("td")[0].text.strip() for row in table.find_all("tr")[1:]]
        return tickers
    else:
        print("Failed to fetch data from Wikipedia.")
        return []


def get_sp500_data(tickers, start_date, end_date):
    sp500_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    sp500_data.reset_index(drop=True, inplace=True)
    return sp500_data


# Implement the momentum strategy with 2-day shift
def momentum_strategy(data, lookback_period):
    returns = data.pct_change()
    momentum_signal = returns.rolling(lookback_period).mean()
    return momentum_signal.shift(2)


# Select the top and bottom stocks based on momentum signals
def select_top_bottom_stocks(momentum_signals):
    mean_momentum = momentum_signals.mean(axis=0)
    top_10_tickers = mean_momentum.nlargest(10).index
    bottom_10_tickers = mean_momentum.nsmallest(10).index
    return top_10_tickers, bottom_10_tickers


# Implement long and short positions for the selected stocks
def backtest_momentum_strategy(asset_returns, top_tickers, bottom_tickers, transaction_cost_pct=0.001):
    long_positions = asset_returns[top_tickers]
    short_positions = -asset_returns[bottom_tickers]
    positions = long_positions.combine_first(short_positions).fillna(0)
    strategy_returns = positions.mean(axis=1)
    strategy_returns -= abs(strategy_returns) * transaction_cost_pct
    cumulative_strategy_returns = (1 + strategy_returns).cumprod()
    cumulative_strategy_returns.iloc[0] = 1
    return cumulative_strategy_returns


# Analyze and visualize the results
def analyze_performance(cumulative_returns):
    if cumulative_returns is None or cumulative_returns.empty or len(cumulative_returns) < 2:
        print("Insufficient data to analyze performance.")
        return


    daily_returns = cumulative_returns.pct_change().dropna()


    plt.figure(figsize=(10, 6))
    cumulative_returns.plot(label='Momentum Strategy', color='blue')
    plt.title('Momentum Strategy Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()


    total_return = cumulative_returns.iloc[-1] / cumulative_returns.iloc[0]
    print("Total Return:", total_return)
    
    daily_risk_free_rate = 0
    excess_returns = daily_returns - daily_risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)


    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()


    print("Sharpe Ratio: {:.2f}".format(sharpe_ratio))
    print("Maximum Drawdown: {:.2f}%".format(max_drawdown * 100))


if __name__ == "__main__":
    start_date = "2019-01-01"
    end_date = "2023-07-01"
    lookback_period = 60
    sp500_tickers = get_sp500_tickers()
    sp500_tickers.remove('BRK.B')
    sp500_tickers.remove('BF.B')
    
    momentum_signals = calculate_momentum_signals(sp500_tickers, start_date, end_date, lookback_period)


    top_10_tickers, bottom_10_tickers = select_top_bottom_stocks(momentum_signals)
    asset_data = get_sp500_data(sp500_tickers, start_date, end_date)


    cumulative_returns = backtest_momentum_strategy(asset_data.pct_change(), top_10_tickers, bottom_10_tickers)
    
    analyze_performance(cumulative_returns)