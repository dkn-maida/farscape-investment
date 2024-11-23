import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download BTC data from Yahoo Finance
btc_data = yf.download('AAPL', start='2015-01-01', end='2024-01-01')

# Calculate the 50-day moving average
btc_data['50_MA'] = btc_data['Close'].rolling(window=50).mean()

# Generate trading signals
btc_data['Signal'] = 0
btc_data['Signal'][50:] = np.where(btc_data['Close'][50:] > btc_data['50_MA'][50:], 1, 0)

# Avoid lookahead bias by shifting the signal one day forward
btc_data['Position'] = btc_data['Signal'].shift(1)

# Calculate daily returns
btc_data['Daily_Return'] = btc_data['Close'].pct_change()

# Calculate strategy returns
btc_data['Strategy_Return'] = btc_data['Position'] * btc_data['Daily_Return']

# Calculate cumulative returns
btc_data['Cumulative_BTC_Return'] = (1 + btc_data['Daily_Return']).cumprod() - 1
btc_data['Cumulative_Strategy_Return'] = (1 + btc_data['Strategy_Return']).cumprod() - 1

# Calculate performance metrics
cagr = (1 + btc_data['Cumulative_Strategy_Return'].iloc[-1])**(365.0/len(btc_data)) - 1
sharpe_ratio = (btc_data['Strategy_Return'].mean() / btc_data['Strategy_Return'].std()) * np.sqrt(252)
max_drawdown = btc_data['Cumulative_Strategy_Return'].cummax() - btc_data['Cumulative_Strategy_Return']
max_drawdown = max_drawdown.max()

# Print performance metrics
print(f"CAGR: {cagr:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Plot the cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(btc_data['Cumulative_BTC_Return'], label='BTC Buy and Hold', color='blue')
plt.plot(btc_data['Cumulative_Strategy_Return'], label='Moving Average Strategy', color='red')
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()
