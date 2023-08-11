# Import libraries
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Download historical data
data = yf.download('BTC-USD', start='2000-01-01')

# Calculate the 10-day high
data['10_day_high'] = data['Close'].rolling(window=5).max()

# Create a signal when the close price crosses the 10_day_high
data['buy_signal'] = np.where(data['Close'] > data['10_day_high'].shift(), 1, 0)

# We sell one day after buying
data['sell_signal'] = data['buy_signal'].shift()

# Set the sell signal column to zero initially
data['sell_signal'].fillna(0, inplace=True)

# Calculate daily strategy returns
data['strategy_returns'] = np.where(data['sell_signal'] == 1, data['Close'].pct_change(), 0)

# Calculate cumulative strategy returns
data['strategy_cumulative_returns'] = (1 + data['strategy_returns']).cumprod()

# Plot the equity curve of the strategy and benchmark
plt.figure(figsize=(10,5))
# convert y-axis to Logarithmic scale
plt.plot(data['strategy_cumulative_returns'], label='10 Day High Strategy',)
plt.legend()
plt.show()
