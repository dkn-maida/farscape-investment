import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the tickers and data period
tickers = ['GC=F', 'SPY']  # Fixed symbols: using 'SPY' instead of '^GSPC'
start_date = '2004-01-01'

# Download the historical data
data = yf.download(tickers, start=start_date)

# Access the Adjusted Close prices
adj_close = data['Adj Close']

# Forward-fill missing data to handle any NaNs
adj_close.fillna(method='ffill', inplace=True)

# Calculate the monthly data
monthly_data = adj_close.resample('M').last()

# Calculate monthly returns
monthly_returns = monthly_data.pct_change()

# Calculate 12-month rolling returns for each asset
rolling_returns = monthly_data.pct_change(12)

# Shift the rolling returns by one period to avoid look-ahead bias
rolling_returns = rolling_returns.shift(1)

# Create a DataFrame to store the strategy's positions and performance
strategy = pd.DataFrame(index=monthly_returns.index)
strategy['Position'] = np.nan
strategy['Portfolio Value'] = np.nan

# Initialize the portfolio value at the first valid index after shifting
first_valid_idx = rolling_returns.index.get_loc(rolling_returns.first_valid_index())
start_idx = first_valid_idx + 1  # Start from the next month
strategy.iloc[start_idx, strategy.columns.get_loc('Portfolio Value')] = 1  # Start with an initial value of 1

# Backtest logic: choose the asset with the highest 12-month return each month
for i in range(start_idx, len(rolling_returns)):
    date = rolling_returns.index[i]
    prev_date = rolling_returns.index[i - 1]
    
    # Find the ticker with the highest rolling return over the past 12 months
    best_asset = rolling_returns.iloc[i].idxmax()
    strategy.loc[date, 'Position'] = best_asset

    # Calculate the portfolio value based on the chosen asset
    if not pd.isna(strategy.loc[prev_date, 'Portfolio Value']):
        prev_portfolio_value = strategy.loc[prev_date, 'Portfolio Value']
        asset_return = monthly_returns.loc[date, best_asset]
        if pd.isna(asset_return):
            asset_return = 0  # Assume zero return if data is missing
        strategy.loc[date, 'Portfolio Value'] = prev_portfolio_value * (1 + asset_return)
    else:
        strategy.loc[date, 'Portfolio Value'] = strategy['Portfolio Value'].ffill().iloc[-1]

# Drop NaN values from the beginning of the portfolio
strategy = strategy.dropna(subset=['Portfolio Value'])

# Compute cumulative returns for SPY as a benchmark
spy_returns = monthly_returns['SPY']
spy_cum_returns = (1 + spy_returns).cumprod()
spy_cum_returns = spy_cum_returns.loc[strategy.index]  # Align with strategy dates

# Compute performance metrics
def compute_performance_metrics(cum_returns):
    # Remove any NaN values
    cum_returns = cum_returns.dropna()
    # Calculate CAGR
    n_years = (cum_returns.index[-1] - cum_returns.index[0]).days / 365.25
    cagr = (cum_returns.iloc[-1] / cum_returns.iloc[0]) ** (1 / n_years) - 1

    # Calculate monthly returns from cumulative returns
    monthly_returns = cum_returns.pct_change().dropna()

    # Calculate annualized volatility
    vol = monthly_returns.std() * np.sqrt(12)

    # Calculate Sharpe Ratio (Assuming risk-free rate = 0)
    sharpe_ratio = cagr / vol if vol != 0 else np.nan

    # Calculate Maximum Drawdown
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns / rolling_max) - 1
    max_drawdown = drawdown.min()

    return cagr, vol, sharpe_ratio, max_drawdown

# Strategy performance metrics
strategy_returns = strategy['Portfolio Value']
strategy_cagr, strategy_vol, strategy_sharpe, strategy_max_dd = compute_performance_metrics(strategy_returns)

# SPY performance metrics
spy_cagr, spy_vol, spy_sharpe, spy_max_dd = compute_performance_metrics(spy_cum_returns)

# Print performance metrics
print("Strategy Performance Metrics:")
print(f"CAGR: {strategy_cagr:.2%}")
print(f"Volatility: {strategy_vol:.2%}")
print(f"Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"Maximum Drawdown: {strategy_max_dd:.2%}")
print("\nSPY Performance Metrics:")
print(f"CAGR: {spy_cagr:.2%}")
print(f"Volatility: {spy_vol:.2%}")
print(f"Sharpe Ratio: {spy_sharpe:.2f}")
print(f"Maximum Drawdown: {spy_max_dd:.2%}")

# Plot the portfolio value and SPY cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(strategy_returns, label='Strategy Portfolio Value')
plt.plot(spy_cum_returns, label='SPY Benchmark')
plt.title('Strategy vs. SPY Benchmark')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.show()
