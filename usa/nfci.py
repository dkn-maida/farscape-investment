import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Read the NFCI data
nfci_data = pd.read_csv('nfci.csv')
nfci_data['date'] = pd.to_datetime(nfci_data['DATE'])
nfci_data.set_index('date', inplace=True)
nfci_data['nfci_sma_14'] = nfci_data['NFCI'].rolling(window=14).mean()

tickers = ['SPY']

# Download VIXM data
vixm_data = yf.download('SHV', start=nfci_data.index.min(), end=nfci_data.index.max())
vixm_data = vixm_data[['Close']].resample("W-FRI").last()

# Annualizing factor assuming trading weeks
annualizing_factor = 52

for ticker in tickers:
    stock_data = yf.download(ticker, start=nfci_data.index.min(), end=nfci_data.index.max())
    stock_data = stock_data[['Close']].resample("W-FRI").last()

    # Merge the stock, VIXM, and NFCI data
    data = stock_data.join(nfci_data, how='inner')
    data = data.join(vixm_data, rsuffix='_SH', how='left')

# Calculate strategy returns
data['strategy_returns'] = np.where(data['signal'] == 1, data['SPY'].shift(1).pct_change(), data['VIXM'].shift(1).pct_change())
data['spy_returns'] = data['SPY'].pct_change()
data['cumulative_strategy_returns'] = (1 + data['strategy_returns']-0.0001).cumprod()
# Calculate the total period in years
total_years = (data.index[-1] - data.index[0]).days / 365.0
# Calculate CAGR using the final cumulative return value
end_value = data['cumulative_strategy_returns'].iloc[-1]
cagr_weekly = (end_value) ** (1 / total_years) - 1

    for i in range(1, len(data)):
        data.loc[data.index[i], 'strategy_returns'] = (
            data['Close'].pct_change().iloc[i]
            if data['signal'].iloc[i - 1] == 1
            else data['Close_SH'].pct_change().iloc[i]
        )
    data['stock_returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    plt.figure(figsize=(10, 6))
    (1 + data['strategy_returns']).cumprod().plot(label='Strategy', color='b')
    (1 + data['stock_returns']).cumprod().plot(label=f'{ticker} Buy and Hold', color='r')
    plt.title(f'Performance Comparison for {ticker}')
    plt.legend()
    plt.show()

def evaluate_strategy(window_size, data):
    data['nfci_sma'] = data['NFCI'].rolling(window=window_size).mean()
    data['nfci_sma_shifted'] = data['nfci_sma'].shift(1)
    data['signal'] = np.where(data['NFCI'] < data['nfci_sma_shifted'], 1, 0)
    data['strategy_returns'] = np.where(data['signal'] == 1, data['SPY'].shift(1).pct_change(), data['VIXM'].shift(1).pct_change())
    
    years = (data.index[-1] - data.index[0]).days / 365.0
    cagr_strategy = (1 + data['strategy_returns']).cumprod().iloc[-1] ** (1 / years) - 1
    cumulative_returns_strategy = (1 + data['strategy_returns']).cumprod()
    rolling_max_strategy = cumulative_returns_strategy.expanding().max()
    daily_drawdown_strategy = cumulative_returns_strategy / rolling_max_strategy - 1
    max_drawdown_strategy = daily_drawdown_strategy.min()

    cumulative_returns_stock = (1 + data['stock_returns']).cumprod()
    rolling_max_stock = cumulative_returns_stock.expanding().max()
    daily_drawdown_stock = cumulative_returns_stock / rolling_max_stock - 1
    max_drawdown_stock = daily_drawdown_stock.min()

    # Sharpe Ratio (assuming risk-free rate to be 0)
    sharpe_ratio_strategy = np.mean(data['strategy_returns']) / np.std(data['strategy_returns']) * np.sqrt(annualizing_factor)
    sharpe_ratio_stock = np.mean(data['stock_returns']) / np.std(data['stock_returns']) * np.sqrt(annualizing_factor)

# Optional: plot the results to visualize them
results_df.set_index('Window Size').plot(subplots=True, figsize=(10, 8))
plt.tight_layout()
plt.show()

# Splitting data into training (80%) and test (20%)
train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size].copy()  # use .copy() to create a copy
test_data = data.iloc[train_size:].copy()   # use .copy() to create a copy


# Strategy design on training data
# For simplicity, let's stick with your 2-day moving average. However, you'd ideally want to test multiple periods here.
train_data['nfci_sma_14'] = train_data['NFCI'].rolling(window=2).mean()
train_data['nfci_sma_14_shifted'] = train_data['nfci_sma_14'].shift(1)
train_data['signal'] = np.where(train_data['NFCI'] < train_data['nfci_sma_14_shifted'], 1, 0)
train_data['strategy_returns'] = np.where(train_data['signal'] == 1, train_data['SPY'].shift(1).pct_change(), train_data['VIXM'].shift(1).pct_change())

# Applying strategy to test data
test_data['nfci_sma_14'] = test_data['NFCI'].rolling(window=2).mean()
test_data['nfci_sma_14_shifted'] = test_data['nfci_sma_14'].shift(1)
test_data['signal'] = np.where(test_data['NFCI'] < test_data['nfci_sma_14_shifted'], 1, 0)
test_data['strategy_returns'] = np.where(test_data['signal'] == 1, test_data['SPY'].shift(1).pct_change(), test_data['VIXM'].shift(1).pct_change())


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

# Determine where trades take place (i.e., where the signal changes)
data['trade_executed'] = data['signal'].diff().abs()
# Apply transaction costs on the days trades are executed

# Display the list of all trades with their date
trade_dates = data[data['trade_executed'] == 1].index

# Create a dataframe for the trade log
trade_log = pd.DataFrame(columns=['Start Date', 'End Date', 'Asset', 'Entry Price', 'Exit Price'])

# Iterate over the data to identify trades and record them in the trade log
current_position = None
entry_price = None
for i, row in data.iterrows():
    if row['trade_executed'] == 1:
        if current_position:  # If there's a current position, record the exit trade
            trade_data = pd.DataFrame({
                'Start Date': [trade_start],
                'End Date': [i],
                'Asset': [current_position],
                'Entry Price': [entry_price],
                'Exit Price': [row[current_position]]
            })
            trade_log = pd.concat([trade_log, trade_data], ignore_index=True)
        
        # Update current position and trade start
        current_position = 'SPY' if row['signal'] == 1 else 'VIXM'
        trade_start = i
        entry_price = row[current_position]
        
# Plot the NFCI value over time
nfci_data['NFCI'].plot(label='NFCI', color='b')
plt.title('NFCI Over Time')
plt.legend()
plt.show()

# Plot the trading signals over time
data['signal'].plot(label='Trading Signal', color='b')
plt.title('Trading Signals Over Time')
plt.legend()
plt.show()

# Get the latest date and value
latest_date = data.index[-1]
latest_value = data['signal'].iloc[-1]

# Print the latest date and value
print(f"Latest Date: {latest_date}, Latest Signal Value: {latest_value}")

correlation = data['SPY'].pct_change().corr(data['VIXM'].pct_change())
print(f"Correlation between SPY and VIXM: {correlation:.4f}")