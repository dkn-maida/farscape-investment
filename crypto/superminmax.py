import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Download historical data
data = yf.download('BTC-USD', start='2020-01-01', end='2023-07-26')

# Calculate the 10-day low and 10-day high
data['10_day_low'] = data['Close'].rolling(window=7).min()
data['10_day_high'] = data['Close'].rolling(window=7).max()

# Create a buy signal when the close price drops below the 10_day_low
data['buy_signal'] = data['Close'] < data['10_day_low'].shift()

# Create a sell signal when the close price rises above the 10_day_high
data['sell_signal'] = data['Close'] > data['10_day_high'].shift()

# Calculate daily percentage returns of strategy
# Buy when there's a buy signal and sell when there's a sell signal
data['strategy_returns'] = np.where(data['buy_signal'], data['Close'].pct_change(), 0)
data['strategy_returns'] = np.where(data['sell_signal'], -data['Close'].pct_change(), data['strategy_returns'])

# Calculate cumulative returns of strategy
data['strategy_cumulative_returns'] = (data['strategy_returns'] + 1).cumprod()

# Buy and Hold returns
data['buy_and_hold_returns'] = data['Close'].pct_change()

# Cumulative Buy and Hold returns
data['buy_and_hold_cumulative_returns'] = (data['buy_and_hold_returns'] + 1).cumprod()

# Calculate max drawdown
rolling_max = data['strategy_cumulative_returns'].cummax()
daily_drawdown = data['strategy_cumulative_returns']/rolling_max - 1.0
max_daily_drawdown = daily_drawdown.min()
print("Max Drawdown: ", max_daily_drawdown)

# Calculate CAGR
years = (data['strategy_cumulative_returns'].index[-1] - data['strategy_cumulative_returns'].index[0]).days / 365.25
cagr = (data['strategy_cumulative_returns'][-1])**(1/years) - 1
print("CAGR: ", cagr)

# Plot the equity curves of the strategy and the buy and hold
plt.figure(figsize=(10,5))
plt.plot(data['strategy_cumulative_returns'], label='Combined Min/Max Strategy')
plt.plot(data['buy_and_hold_cumulative_returns'], label='Buy and Hold')
plt.legend()
plt.show()
