import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the sector ETFs and SPY for benchmark
sector_etfs = {
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLE': 'Energy',
    'XLF': 'Financials',
    'XLV': 'Health Care',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    #'XLRE': 'Real Estate',
    'XLK': 'Technology',
    #'XLC': 'Communications',
    'XLU': 'Utilities',
}

# Add SPY for comparison
etfs = list(sector_etfs.keys()) + ['SPY']

# Download data from Yahoo Finance
data = yf.download(etfs, start="2010-01-01", end="2024-01-01")['Adj Close']

# Calculate 12-1 month momentum
lookback_12m = 252  # 12 months ~252 trading days
exclusion_1m = 21   # 1 month ~21 trading days
momentum_12_1 = data.pct_change(lookback_12m - exclusion_1m) - data.pct_change(exclusion_1m)

# Calculate 6-1 month momentum
lookback_6m = 126  # 6 months ~126 trading days
momentum_6_1 = data.pct_change(lookback_6m - exclusion_1m) - data.pct_change(exclusion_1m)

# Calculate the 12-month rolling volatility (standard deviation)
volatility_12m = data.pct_change().rolling(window=lookback_12m).std()

# Combine momentum scores and adjust by volatility
combined_momentum = (momentum_12_1 + momentum_6_1) / volatility_12m

# Resample the data to monthly frequency, taking the last available price in each month
combined_momentum = combined_momentum[sector_etfs.keys()].resample('M').last()
data = data.resample('M').last()

# Function to calculate cumulative returns for a strategy with a specific number of top or bottom ETFs
def calculate_cumulative_returns(num_top, num_bottom, combined_momentum, data):
    start_date = combined_momentum.dropna().index[0]
    cash = initial_balance = 100000
    portfolio_value = pd.Series(index=data.loc[start_date:].index)
    
    # Backtest
    for date in combined_momentum.loc[start_date:].index:
        current_momentum = combined_momentum.loc[date].dropna()
        
        # Select top and bottom ETFs
        top_etfs = current_momentum.nlargest(num_top).index
        bottom_etfs = current_momentum.nsmallest(num_bottom).index
        
        # Reallocate equally among the top and bottom ETFs
        weights = np.array([0.8/num_top]*num_top + [0.2/num_bottom]*num_bottom)
        selected_etfs = pd.concat([top_etfs, bottom_etfs])
        
        total_shares = {}
        total_value = 0
        for i, etf in enumerate(selected_etfs):
            if etf in data.columns:
                shares = (cash * weights[i]) // data.loc[date, etf]
                total_shares[etf] = shares
                total_value += shares * data.loc[date, etf]
        
        portfolio_value.loc[date] = total_value
        cash = initial_balance - total_value

    strategy_returns = portfolio_value.pct_change().fillna(0)
    cumulative_returns = (1 + strategy_returns).cumprod()
    return cumulative_returns

# Calculate cumulative returns for different strategies
cumulative_returns_top1_bottom1 = calculate_cumulative_returns(1, 1, combined_momentum, data)
cumulative_returns_top2_bottom2 = calculate_cumulative_returns(2, 2, combined_momentum, data)
cumulative_returns_top3_bottom3 = calculate_cumulative_returns(3, 3, combined_momentum, data)

# Plotting the strategy performance
plt.figure(figsize=(14, 7))
plt.plot(cumulative_returns_top1_bottom1, label='Top 1 & Bottom 1')
plt.plot(cumulative_returns_top2_bottom2, label='Top 2 & Bottom 2')
plt.plot(cumulative_returns_top3_bottom3, label='Top 3 & Bottom 3')
plt.title('Sector Momentum Rotation Strategy: Performance of Top and Bottom ETFs')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()
