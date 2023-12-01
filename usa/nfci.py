import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Read the NFCI data
nfci_data = pd.read_csv('nfci.csv')
nfci_data['date'] = pd.to_datetime(nfci_data['DATE'])
nfci_data.set_index('date', inplace=True)

# Calculate the 14-day moving average for NFCI and shift it
nfci_data['nfci_sma_14'] = nfci_data['NFCI'].rolling(window=2).mean()
nfci_data['nfci_sma_14_shifted'] = nfci_data['nfci_sma_14'].shift(1)

# Download SPXL and SPXS data
symbols = ['SPXL', 'SPXS']
prices_data = yf.download(symbols, start='2010-01-01', end=nfci_data.index.max())['Close']

# Resample to weekly data
prices_data_weekly = prices_data.resample("W-FRI").last()

# Calculate the 12-week SMA for SPXL
prices_data_weekly['SPXL_sma_12'] = prices_data_weekly['SPXL'].rolling(window=12).mean()

# Merge the dataframes
data = prices_data_weekly.join(nfci_data, how='inner')

# Create a signal
data['signal_long'] = np.where((data['NFCI'] < data['nfci_sma_14_shifted']) & (data['SPXL'] > data['SPXL_sma_12']), 1, 0)
data['signal_short'] = np.where((data['NFCI'] > data['nfci_sma_14_shifted']) & (data['SPXL'] < data['SPXL_sma_12']), 1, 0)

# Calculate strategy returns
data['strategy_returns'] = np.where(data['signal_long'] == 1, data['SPXL'].shift(1).pct_change(), np.where(data['signal_short'] == 1, data['SPXS'].shift(1).pct_change(), 0))
data['SPXL_returns'] = data['SPXL'].pct_change()
data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod()

# Calculate the total period in years
total_years = (data.index[-1] - data.index[0]).days / 365.0
# Calculate CAGR using the final cumulative return value
end_value = data['cumulative_strategy_returns'].iloc[-1]
cagr_weekly = (end_value) ** (1 / total_years) - 1

print(f"Weekly CAGR: {cagr_weekly*100:.2f}%")

# Drop missing values
data.dropna(inplace=True)
# Plotting weekly strategy returns

plt.figure(figsize=(14, 7))
data['cumulative_strategy_returns'].plot()
plt.title('Weekly Strategy Returns Over Time')
plt.ylabel('Weekly Return')
plt.xlabel('Date')
plt.grid(True)
plt.show()

# Plotting weekly strategy returns
plt.figure(figsize=(14, 7))
data['cumulative_strategy_returns'].plot()
plt.title('Weekly Strategy Returns Over Time')
plt.ylabel('Weekly Return')
plt.yscale("log")
plt.xlabel('Date')
plt.grid(True)
plt.show()

# Splitting data into training (80%) and test (20%)
train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size].copy()  # use .copy() to create a copy
test_data = data.iloc[train_size:].copy()   # use .copy() to create a copy

# Modify strategy design on training and test data
for dataset in [train_data, test_data]:
    dataset['signal_long'] = np.where((dataset['NFCI'] < dataset['nfci_sma_14_shifted']) & (dataset['SPXL'] > dataset['SPXL_sma_12']), 1, 0)
    dataset['signal_short'] = np.where((dataset['NFCI'] > dataset['nfci_sma_14_shifted']) & (dataset['SPXL'] < dataset['SPXL_sma_12']), 1, 0)
    dataset['strategy_returns'] = np.where(dataset['signal_long'] == 1, dataset['SPXL'].shift(1).pct_change(), np.where(dataset['signal_short'] == 1, dataset['SPXS'].shift(1).pct_change(), 0))

years = (test_data.index[-1] - test_data.index[0]).days / 365.0
cagr_strategy = (1 + test_data['strategy_returns']).cumprod().iloc[-1] ** (1 / years) - 1

print(f"Strategy on Test Data: CAGR: {cagr_strategy:.2%}")
test_data['cumulative_strategy_returns'] = (1 + test_data['strategy_returns']-0.0001).cumprod()
# Plotting weekly strategy returns
plt.figure(figsize=(14, 7))
test_data['cumulative_strategy_returns'].plot()
plt.title('Test data returns')
plt.ylabel('Return')
plt.grid(True)
plt.show()
        
# Plot the NFCI value over time
nfci_data['NFCI'].plot(label='NFCI', color='b')
plt.title('NFCI Over Time')
plt.legend()
plt.show()

# Plot the trading signals over time
data['signal_long'].plot(label='Trading Signal', color='b')
data['signal_short'].plot(label='Trading Signal', color='r')
plt.title('Trading Signals Over Time')
plt.legend()
plt.show()

# Get the latest date and value
latest_date = data.index[-1]
latest_value = data['signal_long'].iloc[-1]

# Print the latest date and value
print(f"Latest Date: {latest_date}, Latest Signal Value: {latest_value}")

correlation = data['SPXL'].pct_change().corr(data['SPXS'].pct_change())
print(f"Correlation between SPXL and SPXS: {correlation:.4f}")

# Calculate annual returns
annual_returns = (data['strategy_returns'] + 1).groupby(data.index.year).prod() - 1

# Display annual returns
print("Annual Returns:")
print(annual_returns)