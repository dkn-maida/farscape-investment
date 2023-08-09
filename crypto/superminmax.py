import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Download historical data
data = yf.download('BTC-USD', start='2016-01-01')

# Calculate the 10-day high and 5-day low
data['10_day_high'] = data['High'].rolling(window=7).max()
data['5_day_low'] = data['Low'].rolling(window=5).min()

# Signals for the first strategy
data['buy_signal_1'] = np.where(data['High'] > data['10_day_high'].shift(), 1, 0)
data['sell_signal_1'] = data['buy_signal_1'].shift()
data['sell_signal_1'].fillna(0, inplace=True)

# Signals for the second strategy
data['entry_signal_2'] = np.where(data['Close'] < data['5_day_low'].shift(), 1, 0)
data['exit_signal_2'] = data['entry_signal_2'].shift()
data['exit_signal_2'].fillna(0, inplace=True)

# Eliminate overlapping signals
data['entry_signal_2'] = np.where(data['buy_signal_1'] == 1, 0, data['entry_signal_2'])
data['exit_signal_2'] = np.where(data['sell_signal_1'] == 1, 0, data['exit_signal_2'])

# Calculate returns for the first strategy
data['strategy_returns_1'] = np.where(data['sell_signal_1'] == 1, np.where(data['High']/data['Open'] >= 1.05, 0.05, data['Close'].pct_change()), 0)

# Calculate returns for the second strategy
data['strategy_returns_2'] = np.where(data['exit_signal_2'] == 1, data['Close'].pct_change(), 0)

# Combine the returns
data['combined_strategy_returns'] = data['strategy_returns_1'] + data['strategy_returns_2']

# Calculate cumulative returns for all strategies
data['strategy_1_cumulative_returns'] = (1 + data['strategy_returns_1']).cumprod()
data['strategy_2_cumulative_returns'] = (1 + data['strategy_returns_2']).cumprod()
data['combined_strategy_cumulative_returns'] = (1 + data['combined_strategy_returns']).cumprod()

# Calculate "Buy and Hold" returns
data['buy_and_hold_returns'] = data['Close'].pct_change()
data['buy_and_hold_cumulative_returns'] = (1 + data['buy_and_hold_returns']).cumprod()

# Plot the equity curves
plt.figure(figsize=(15,7))
plt.plot(data['strategy_1_cumulative_returns'], label='10 Day High Strategy')
plt.plot(data['strategy_2_cumulative_returns'], label='5 Day Low Strategy')
plt.plot(data['combined_strategy_cumulative_returns'], label='Combined Strategy', linestyle='--')
plt.plot(data['buy_and_hold_cumulative_returns'], label='Buy and Hold', linestyle='-')
plt.legend()
plt.title("Strategies' Cumulative Returns over Time vs Buy and Hold")
plt.show()