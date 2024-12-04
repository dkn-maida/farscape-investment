import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download SPY data from Yahoo Finance
spy_data = yf.download('SPY', start='2019-01-01')

# Calculate daily returns
spy_data['Daily Return'] = spy_data['Adj Close'].pct_change()

# Calculate rolling volatilities
spy_data['12M Rolling Volatility'] = spy_data['Daily Return'].rolling(window=252).std() * (252 ** 0.5)
spy_data['3M Rolling Volatility'] = spy_data['Daily Return'].rolling(window=63).std() * (252 ** 0.5)
spy_data['1M Rolling Volatility'] = spy_data['Daily Return'].rolling(window=21).std() * (252 ** 0.5)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot SPY price
ax1.set_title('SPY Price and Rolling Volatility')
ax1.plot(spy_data['Adj Close'], label='SPY Price', color='blue')
ax1.set_ylabel('SPY Price', color='blue')

# Highlight periods where 12M Rolling Volatility > 0.15
high_volatility = spy_data['12M Rolling Volatility'] > 0.15
ax1.fill_between(
    spy_data.index,
    spy_data['Adj Close'].min(),
    spy_data['Adj Close'],
    where=high_volatility,
    color='red',
    alpha=0.3,
    label='High Volatility (>0.15)'
)
ax1.legend(loc='upper left')
ax1.grid()

# Plot rolling volatilities
ax2.plot(spy_data['12M Rolling Volatility'], label='12M Rolling Volatility', color='orange')
ax2.plot(spy_data['3M Rolling Volatility'], label='3M Rolling Volatility', color='green')
ax2.plot(spy_data['1M Rolling Volatility'], label='1M Rolling Volatility', color='purple')
ax2.axhline(y=0.15, color='red', linestyle='--', label='Volatility Threshold (0.15)')
ax2.set_ylabel('Annualized Rolling Volatility')
ax2.set_xlabel('Date')
ax2.legend(loc='upper left')
ax2.grid()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
