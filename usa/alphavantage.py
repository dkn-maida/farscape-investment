import yfinance as yf
import pandas as pd
import numpy as np

# List of updated S&P 100 symbols
sp100_symbols = ['AAPL', 'ABBV', 'ABT', 'ACN', 'AIG', 'ALL', 'AMGN', 'AMZN', 'AXP', 'BA', 'BAC', 'BIIB', 'BK', 
                 'BLK', 'BMY', 'C', 'CAT', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CSCO', 'CVS', 'CVX', 'DD', 
                 'DHR', 'DIS', 'DOW', 'DUK', 'EMR', 'EXC', 'F', 'FDX', 'FOX', 'FOXA', 'GD', 'GE', 'GILD', 
                 'GM', 'GOOG', 'GOOGL', 'GS', 'HAL', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KMI', 'KO', 'LLY', 
                 'LMT', 'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE', 
                 'NKE', 'ORCL', 'OXY', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM', 'SBUX', 'SLB', 'SO', 
                 'SPG', 'T', 'TGT', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WBA', 'WFC']

# Fetch historical data
start_date = '2020-01-01'
end_date = '2022-01-01'
data = yf.download(sp100_symbols, start=start_date, end=end_date)['Adj Close']

# Calculate 21-day moving average and daily returns
ma_21 = data.rolling(window=21).mean().shift(1)
daily_returns = data.pct_change()

# Initialize DataFrame for strategy returns
strategy_returns = pd.DataFrame(index=daily_returns.index)

# Calculate daily returns when stock is above 21-day MA
for symbol in sp100_symbols:
    strategy_returns[symbol] = daily_returns[symbol].where(data[symbol] > ma_21[symbol])

# Calculate average daily return across all stocks
average_daily_return = strategy_returns.mean(axis=1)

# Calculate cumulative returns
cumulative_returns = (1 + average_daily_return).cumprod() - 1

# Performance metrics
annualized_return = average_daily_return.mean() * 252
volatility = average_daily_return.std() * np.sqrt(252)
sharpe_ratio = annualized_return / volatility
max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

# Displaying the performance metrics
print(f"Cumulative Return: {cumulative_returns[-1]:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Volatility: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
