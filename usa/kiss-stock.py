import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fetch historical data for AAPL
data = yf.download('BTC-USD', start='2023-12-07', interval="1m")

# Calculate the 21-day moving average for the closing price
data['21_MA'] = data['Close'].rolling(window=2).mean().shift(1)

# Strategy Logic
data['Position'] = np.where(data['Close'] > data['21_MA'], 1, 0)

# Calculate daily returns and strategy returns
data['Daily_Return'] = data['Close'].shift().pct_change()
data['Strategy_Return'] = data['Daily_Return'] * data['Position']

# Avoid look-ahead bias
data = data.dropna()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data['Strategy_Return'].cumsum(), label='Strategy Returns')
plt.plot(data['Daily_Return'].cumsum(), label='AAPL Returns')
plt.title('Backtest of Modified Strategy on AAPL')
plt.legend()
plt.show()