# Import required libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Define the ticker symbol of the index you want to backtest
# Replace 'TICKER_SYMBOL' with the appropriate ticker symbol.
TICKER_SYMBOL = 'QQQ'

# Download historical data for the specified index
data = yf.download(TICKER_SYMBOL)

# Calculate the 200-day simple moving average
data['sma200'] = data['Close'].rolling(window=200).mean()

# Generate trading signals based on the crossover strategy
data['signal'] = 0
data['signal'][200:] = np.where(data['Close'][200:] > data['sma200'][200:], 1, 0)
data['positions'] = data['signal'].diff()

# Backtest the strategy and calculate returns
data['strategy_returns'] = data['Close'].pct_change() * data['signal'].shift(1)
data['market_returns'] = data['Close'].pct_change()

# Calculate the cumulative returns
data['strategy_cumulative_returns'] = (1 + data['strategy_returns']).cumprod()
data['market_cumulative_returns'] = (1 + data['market_returns']).cumprod()

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(data['strategy_cumulative_returns'], label='Strategy')
plt.plot(data['market_cumulative_returns'], label='Market')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title(f'{TICKER_SYMBOL} 200-Day Moving Average Crossover Strategy Backtest')
plt.show()