import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Download historical data for Apple
apple_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Calculate the 150-day simple moving average (SMA)
apple_data['150d_sma'] = apple_data['Adj Close'].rolling(window=20).mean()

# Calculate daily returns
apple_data['Daily Returns'] = apple_data['Adj Close'].pct_change()

# Determine the regimes
apple_data['Regime'] = apple_data.apply(lambda row: 'Above SMA' if row['Adj Close'] > row['150d_sma'] else 'Below SMA', axis=1)

# Calculate the cumulative returns for being invested in each regime
apple_data['Cumulative Returns'] = apple_data.groupby('Regime')['Daily Returns'].apply(lambda x: (1 + x).cumprod() - 1)

# Reset the cumulative returns to zero when regime changes
change_points = apple_data['Regime'] != apple_data['Regime'].shift(1)
apple_data.loc[change_points, 'Cumulative Returns'] = 0
apple_data['Cumulative Returns'] = apple_data.groupby((change_points).cumsum())['Daily Returns'].apply(lambda x: (1 + x).cumprod() - 1)

# Plot the cumulative returns for each regime
plt.figure(figsize=(14, 7))
for regime in apple_data['Regime'].unique():
    data = apple_data[apple_data['Regime'] == regime]
    plt.plot(data.index, data['Cumulative Returns'], label=f'Cumulative Returns {regime}')

plt.title('AAPL Cumulative Returns for Different Investment Regimes Based on 150-Day SMA')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()
