import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Download SPY data
spy = yf.download('^GSPC', start='1900-01-01', end='2024-11-26')

# Resample to monthly frequency
spy_monthly = spy['Adj Close'].resample('ME').last()

# Create a DataFrame to hold rolling returns for different periods
rolling_returns = pd.DataFrame(index=spy_monthly.index)

# Calculate rolling returns for periods from 12 months to 1 month
for months in range(12, 0, -1):  # From 12 months down to 1 month
    rolling_returns[f'{months}M'] = spy_monthly.pct_change(months)

# Calculate the average rolling returns
rolling_returns['Average Rolling Return'] = rolling_returns.mean(axis=1)

# Merge the average rolling return into the main DataFrame
spy_monthly = pd.concat([spy_monthly, rolling_returns['Average Rolling Return']], axis=1)

# Define strategy using the average rolling returns
threshold = 0.01  # 1% threshold

# Lag the signal to avoid look-ahead bias
spy_monthly['Signal'] = spy_monthly['Average Rolling Return'].shift(1) > threshold

# Multiply returns by the lagged signal to act only on past information
spy_monthly['Strategy Monthly Return'] = spy_monthly['Adj Close'].pct_change() * spy_monthly['Signal'].fillna(0)

# Calculate cumulative returns for the strategy and SPY
spy_monthly['Cumulative Strategy Return'] = (1 + spy_monthly['Strategy Monthly Return']).cumprod()
spy_monthly['Cumulative SPY Return'] = (1 + spy_monthly['Adj Close'].pct_change()).cumprod()

# Performance metrics function
def compute_metrics_monthly(daily_returns, cumulative_returns):
    # CAGR
    start_date = daily_returns.index.min()
    end_date = daily_returns.index.max()
    years = (end_date - start_date).days / 365.25
    ending_value = cumulative_returns.iloc[-1]
    cagr = (ending_value) ** (1 / years) - 1

    # Max Drawdown
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Volatility
    volatility = daily_returns.std() * np.sqrt(12)  # Annualized for monthly data

    return cagr, max_drawdown, volatility

# Compute metrics for the strategy
strategy_cagr, strategy_max_drawdown, strategy_volatility = compute_metrics_monthly(
    spy_monthly['Strategy Monthly Return'], spy_monthly['Cumulative Strategy Return']
)

# Compute metrics for SPY
spy_cagr, spy_max_drawdown, spy_volatility = compute_metrics_monthly(
    spy_monthly['Adj Close'].pct_change(), spy_monthly['Cumulative SPY Return']
)

# Print results
print("Performance Metrics (Based on Average Rolling Returns):")
print(f"Strategy:   CAGR: {strategy_cagr:.2%}, Max Drawdown: {strategy_max_drawdown:.2%}, Volatility: {strategy_volatility:.2%}")
print(f"SPY:        CAGR: {spy_cagr:.2%}, Max Drawdown: {spy_max_drawdown:.2%}, Volatility: {spy_volatility:.2%}")

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot cumulative returns for the strategy and SPY
axs[0].plot(spy_monthly.index, spy_monthly['Cumulative Strategy Return'], label='Cumulative Strategy Return', color='green')
axs[0].plot(spy_monthly.index, spy_monthly['Cumulative SPY Return'], label='Cumulative SPY Return (Baseline)', color='blue')
axs[0].set_title('Cumulative Returns: Strategy vs. SPY (Monthly)')
axs[0].set_ylabel('Cumulative Return')
axs[0].axhline(1, color='black', linestyle='--', linewidth=0.8, label='Baseline (1)')
axs[0].grid(True)
axs[0].legend()

# Plot average rolling returns
axs[1].plot(rolling_returns.index, rolling_returns['Average Rolling Return'], label='Average Rolling Return', color='darkred')
axs[1].axhline(threshold, color='orange', linestyle='--', linewidth=0.8, label='1% Threshold')
axs[1].set_title('Average Rolling Returns (12M to 1M)')
axs[1].set_ylabel('Average Return')
axs[1].grid(True)
axs[1].legend()

# Plot strategy signal
axs[2].plot(spy_monthly.index, spy_monthly['Signal'], label='Signal (Active Periods)', color='purple')
axs[2].set_title('Strategy Signal (Based on Average Rolling Returns)')
axs[2].set_ylabel('Signal')
axs[2].grid(True)
axs[2].legend()

# Adjust layout
plt.xlabel('Date')
plt.tight_layout()
plt.show()
