import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Download historical data
data = yf.download('BTC-USD', start='2015-01-01')

# Calculate the 10-day low and EMA100
data['5_day_low'] = data['Low'].rolling(window=7).min()

# Create a signal when the close price drops below the 10_day_low and is above the EMA100
data['buy_signal'] = (data['Close'] < data['5_day_low'].shift())
data['sell_signal'] = data['buy_signal'].shift()

# Calculate daily percentage returns of strategy
data['strategy_returns'] = np.where(data['sell_signal'] == 1, data['Close'].pct_change(), 0)

# Calculate cumulative returns of strategy
data['strategy_cumulative_returns'] = (data['strategy_returns'] + 1).cumprod()

# Plot the equity curves of the strategy and the buy and hold
plt.figure(figsize=(10,5))
plt.plot(data['strategy_cumulative_returns'], label='5 Day Low Strategy')
plt.legend()
plt.show()