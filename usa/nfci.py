import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Read the data
nfci_data = pd.read_csv('nfci.csv')
nfci_data['date'] = pd.to_datetime(nfci_data['DATE'])
nfci_data.set_index('date', inplace=True)

# Calculate the 14-day moving average
nfci_data['nfci_sma_14'] = nfci_data['NFCI'].rolling(window=2).mean()
# Shift the 14-day moving average by one day
nfci_data['nfci_sma_14_shifted'] = nfci_data['nfci_sma_14'].shift(1)

# Download SPY and SH data
symbols = ['SPY','VIXM']
prices_data = yf.download(symbols, start='2011-01-03', end=nfci_data.index.max())['Close']
prices_data = prices_data.resample("W-FRI").last()

# Saving trade log
prices_data.to_csv('prices_data.csv')

# Merge the dataframes
data = prices_data.join(nfci_data, how='inner')

# Create a signal
data['signal'] = np.where(data['NFCI'] < data['nfci_sma_14_shifted'], 1, 0)

# Calculate strategy returns
data['strategy_returns'] = np.where(data['signal'] == 1, data['SPY'].shift(1).pct_change(), data['VIXM'].shift(1).pct_change())
data['spy_returns'] = data['SPY'].pct_change()

# Drop missing values
data.dropna(inplace=True)

# Determine where trades take place (i.e., where the signal changes)
data['trade_executed'] = data['signal'].diff().abs()
# Apply transaction costs on the days trades are executed

# Determine the number of trades
number_of_trades = data['trade_executed'].sum()
print(f"Number of trades executed: {number_of_trades}")

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
        
# Calculate return for each trade
trade_log['Trade Return'] = (trade_log['Exit Price'] - trade_log['Entry Price']) / trade_log['Entry Price']
trade_log['Cumulative Return'] = (1 + trade_log['Trade Return']).cumprod() - 1

# Plotting cumulative returns
trade_log['Cumulative Return'].plot()
plt.title('Cumulative Returns Over Time')
plt.ylabel('Cumulative Return')
plt.yscale("log")
plt.xlabel('Trade Number')
plt.show()


# Convert the 'Start Date' and 'End Date' columns to datetime
trade_log['Start Date'] = pd.to_datetime(trade_log['Start Date'])
trade_log['End Date'] = pd.to_datetime(trade_log['End Date'])

# Calculate the total period in years
start_date = trade_log['Start Date'].iloc[0]
end_date = trade_log['End Date'].iloc[-1]
total_years = (end_date - start_date).days / 365.0

# Calculate CAGR
end_value = trade_log['Cumulative Return'].iloc[-1] + 1  # Adding 1 to convert to total value (not just the return)
start_value = 1
cagr = (end_value / start_value) ** (1 / total_years) - 1

print(f"CAGR: {cagr*100:.2f}%")

# Saving trade log
trade_log.to_csv('trade_log_with_returns.csv', index=False)

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

# Compute cumulative returns from daily strategy returns
data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod()
# Compute the rolling maximum value (peak)
data['rolling_peak'] = data['cumulative_strategy_returns'].expanding(min_periods=1).max()
# Calculate daily drawdown
data['daily_drawdown'] = data['cumulative_strategy_returns'] / data['rolling_peak'] - 1
# Determine the max drawdown
max_drawdown = data['daily_drawdown'].min()
print(f"Max Drawdown: {max_drawdown*100:.2f}%")

### Robustness checks 

def fetch_data():
    # Read the data
    nfci_data = pd.read_csv('nfci.csv')
    nfci_data['date'] = pd.to_datetime(nfci_data['DATE'])
    nfci_data.set_index('date', inplace=True)

    # Calculate the 14-day moving average and shift
    nfci_data['nfci_sma_14'] = nfci_data['NFCI'].rolling(window=2).mean()
    nfci_data['nfci_sma_14_shifted'] = nfci_data['nfci_sma_14'].shift(1)

    # Download SPY, URTH, SH, and VIXM data
    symbols = ['URTH', 'SPY', 'SH', 'VIXM']
    prices_data = yf.download(symbols, start='2008-01-01', end=nfci_data.index.max())['Close']
    prices_data = prices_data.resample("W-FRI").last()

    # Merge the dataframes
    data = prices_data.join(nfci_data, how='inner')

    return data

def run_strategy(data):
    data = data.copy()  # Make a copy of the data to avoid warnings
    data.loc[:, 'signal'] = np.where(data['NFCI'] < data['nfci_sma_14_shifted'], 1, 0)
    # Calculate strategy returns
    data.loc[:, 'strategy_returns'] = np.where(data['signal'] == 1, data['SPY'].shift(1).pct_change(), data['VIXM'].shift(1).pct_change())

    # Determine where trades take place (i.e., where the signal changes)
    data.loc[:, 'trade_executed'] = data['signal'].diff().abs()

    # Create a dataframe for the trade log
    trade_log = pd.DataFrame(columns=['Start Date', 'End Date', 'Asset', 'Entry Price', 'Exit Price'])

    # Iterate over the data to identify trades and record them in the trade log
    current_position = None
    entry_price = None
    for i, row in data.iterrows():
        if row['trade_executed'] == 1:
            if current_position:  
                trade_data = pd.DataFrame({
                    'Start Date': [trade_start],
                    'End Date': [i],
                    'Asset': [current_position],
                    'Entry Price': [entry_price],
                    'Exit Price': [row[current_position]]
                })
                trade_log = pd.concat([trade_log, trade_data], ignore_index=True)

            current_position = 'SPY' if row['signal'] == 1 else 'VIXM'
            trade_start = i
            entry_price = row[current_position]

    # Calculate return for each trade
    trade_log['Trade Return'] = (trade_log['Exit Price'] - trade_log['Entry Price']) / trade_log['Entry Price']
    trade_log['Cumulative Return'] = (1 + trade_log['Trade Return']).cumprod() - 1

    return trade_log

def performance_metrics(trade_log):
    # Convert 'Start Date' and 'End Date' to datetime
    trade_log['Start Date'] = pd.to_datetime(trade_log['Start Date'])
    trade_log['End Date'] = pd.to_datetime(trade_log['End Date'])

    # Calculate CAGR
    start_date = trade_log['Start Date'].iloc[0]
    end_date = trade_log['End Date'].iloc[-1]
    total_years = (end_date - start_date).days / 365.0
    end_value = trade_log['Cumulative Return'].iloc[-1] + 1
    cagr = (end_value / 1) ** (1 / total_years) - 1

    # Calculate Max Drawdown for cumulative returns
    cumulative_returns = trade_log['Cumulative Return'] + 1
    rolling_max = cumulative_returns.expanding().max()
    daily_drawdown = cumulative_returns / rolling_max - 1
    max_drawdown = daily_drawdown.min()

    return cagr, max_drawdown

# Fetch and process data
data = fetch_data()

# Split data for out-of-sample testing
in_sample_end_date = '2015-01-01'
in_sample_data = data[:in_sample_end_date]
out_of_sample_data = data[in_sample_end_date:]

# Run strategies and get performance metrics
in_sample_trade_log = run_strategy(in_sample_data)
out_of_sample_trade_log = run_strategy(out_of_sample_data)

in_sample_cagr, in_sample_dd = performance_metrics(in_sample_trade_log)
out_of_sample_cagr, out_of_sample_dd = performance_metrics(out_of_sample_trade_log)

print(f"In-Sample CAGR: {in_sample_cagr*100:.2f}%")
print(f"In-Sample Max Drawdown: {in_sample_dd*100:.2f}%")
print(f"Out-of-Sample CAGR: {out_of_sample_cagr*100:.2f}%")
print(f"Out-of-Sample Max Drawdown: {out_of_sample_dd*100:.2f}%")

# COVID-19 Stress Test
covid_period_data = data['2020-01-01':'2020-12-31']
covid_trade_log = run_strategy(covid_period_data)

covid_cagr, covid_dd = performance_metrics(covid_trade_log)

print(f"COVID Period CAGR: {covid_cagr*100:.2f}%")
print(f"COVID Period Max Drawdown: {covid_dd*100:.2f}%")
