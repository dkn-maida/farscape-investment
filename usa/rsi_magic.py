import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt

def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    data['RSI_3'] = RSIIndicator(data['Close'], window=3).rsi()
    data['Previous_High'] = data['High'].shift(1)
    return data

def check_buy_signal(row):
    return row['RSI_3'] < 10

def check_sell_signal(row):
    return row['RSI_3'] > 60 or row['Close'] > row['Previous_High']

def backtest_strategy(tickers, start_date, end_date, initial_capital):
    stock_data = {ticker: load_data(ticker, start_date, end_date) for ticker in tickers}
    portfolio_value = pd.Series(index=pd.date_range(start=start_date, end=end_date), dtype=float)
    current_capital = initial_capital
    current_position = None
    entry_price = None

    for day in portfolio_value.index:
        if current_position:
            if day in stock_data[current_position].index:
                row = stock_data[current_position].loc[day]
                if check_sell_signal(row):
                    current_capital *= row['Close'] / entry_price
                    current_position = None
        
        if not current_position:
            for ticker in tickers:
                if day in stock_data[ticker].index:
                    row = stock_data[ticker].loc[day]
                    if check_buy_signal(row):
                        current_position = ticker
                        entry_price = row['Close']
                        break

        portfolio_value[day] = current_capital

    return portfolio_value


# List of stock tickers
tickers = ['XOM', 'PG', 'GOOG', 'INTL', 'MMM', 'TSLA', 'CSCO', 'TSM', 'AAPL', 'WMT', 'MA', 'KO', 'ADBE']

# Test the strategy
portfolio_value = backtest_strategy(tickers, '2018-01-01', '2022-12-31', 10000)

# Plotting the portfolio value
plt.figure(figsize=(12, 6))
plt.plot(portfolio_value, label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.show()
