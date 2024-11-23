import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Download BTC data from Yahoo Finance for the 1-hour timeframe
btc_data = yf.download('BTC-USD', start='2023-08-18', interval='1h')
# Calculate the 5-period RSI
btc_data['RSI_5'] = calculate_rsi(btc_data, 5)

# Generate trading signals
btc_data['Signal'] = 0
btc_data['Signal'] = np.where(btc_data['RSI_5'] < 10, 1, btc_data['Signal'])
btc_data['Signal'] = np.where(btc_data['RSI_5'] > 90, 0, btc_data['Signal'])

# Avoid lookahead bias by shifting the signal one period forward
btc_data['Position'] = btc_data['Signal'].shift(1)

# Calculate hourly returns
btc_data['Hourly_Return'] = btc_data['Close'].pct_change()

# Apply 2x leverage to strategy returns
leverage = 1.5
btc_data['Strategy_Return'] = btc_data['Position'] * btc_data['Hourly_Return'] * leverage

# Calculate cumulative returns
btc_data['Cumulative_BTC_Return'] = (1 + btc_data['Hourly_Return']).cumprod() - 1
btc_data['Cumulative_Strategy_Return'] = (1 + btc_data['Strategy_Return']).cumprod() - 1

# Calculate performance metrics
hours_per_year = 365 * 24
cagr = (1 + btc_data['Cumulative_Strategy_Return'].iloc[-1])**(hours_per_year/len(btc_data)) - 1
sharpe_ratio = (btc_data['Strategy_Return'].mean() / btc_data['Strategy_Return'].std()) * np.sqrt(hours_per_year)
max_drawdown = btc_data['Cumulative_Strategy_Return'].cummax() - btc_data['Cumulative_Strategy_Return']
max_drawdown = max_drawdown.max()

# Print performance metrics
print(f"CAGR: {cagr:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Plot the cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(btc_data['Cumulative_BTC_Return'], label='BTC Buy and Hold', color='blue')
plt.plot(btc_data['Cumulative_Strategy_Return'], label='RSI 5 Strategy with 5x Leverage', color='red')
plt.title('Cumulative Returns (1-Hour Timeframe)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()
