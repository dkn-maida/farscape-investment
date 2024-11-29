import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Suppress FutureWarnings (optional)
# You can comment this out if you prefer to see all warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the style to a built-in matplotlib style
plt.style.use('fivethirtyeight')  # Modern style good for financial charts

# Define the tickers
tickers = ['XLK','XLY','XLV','XLP','XLB','XLI','XLE', 'SHY']

# Download data for all tickers
# Use 'Adj Close' for accurate return calculations
data = yf.download(tickers, start='1900-05-09', end='2024-11-28', group_by='ticker')

# Function to extract adjusted close and resample to month-end
def get_monthly_adj_close(ticker):
    return data[ticker]['Adj Close'].resample('ME').last()  # 'ME' for month-end

# Create a DataFrame to hold monthly adjusted close prices
monthly_prices = pd.DataFrame({ticker: get_monthly_adj_close(ticker) for ticker in tickers})

# Drop any columns with all NaN values (in case some tickers don't have data for the entire period)
monthly_prices.dropna(axis=1, how='all', inplace=True)

# Calculate monthly returns for each ETF
monthly_returns = monthly_prices.pct_change()

# Calculate cumulative returns for each ETF (Buy and Hold)
cumulative_returns = (1 + monthly_returns).cumprod()

# Strategy Parameters
threshold = 0.01  # 1% lower threshold for selecting to invest
upper_limit = 0.20  # 20% upper limit for visualization
lower_limit = -0.16  # -16% lower limit for visualization

# Define rolling periods
rolling_periods = range(1, 13)  # 1 month to 12 months

# Calculate average rolling returns for each ETF
rolling_returns = pd.DataFrame(index=monthly_prices.index, columns=tickers)

for ticker in tickers:
    # Calculate rolling returns for each period and take the mean
    temp = pd.DataFrame()
    for months in rolling_periods:
        temp[f'{months}M'] = monthly_prices[ticker].pct_change(months)
    rolling_returns[ticker] = temp.mean(axis=1)

# Drop the initial rows where rolling returns are NaN
rolling_returns.dropna(inplace=True)
monthly_returns = monthly_returns.loc[rolling_returns.index]
monthly_prices = monthly_prices.loc[rolling_returns.index]
cumulative_returns = cumulative_returns.loc[rolling_returns.index]

# Strategy Implementation: Invest each month in the ETF with the highest average rolling return
# Initialize a DataFrame to store signals and strategy returns
strategy_df = pd.DataFrame(index=rolling_returns.index)

# Select the ETF with the highest average rolling return up to month T-1 for investment in month T
# To avoid look-ahead bias, shift the selection by one month
strategy_df['Selected ETF'] = rolling_returns.idxmax(axis=1).shift(1)

# Handle the first month where shift introduces NaN by assuming no investment or default ETF
# Option 1: Backfill
strategy_df['Selected ETF'] = strategy_df['Selected ETF'].fillna(method='bfill')
# Option 2: Set a default ETF (e.g., SPY)
# strategy_df['Selected ETF'] = strategy_df['Selected ETF'].fillna('SPY')

# Replace deprecated 'lookup' with 'apply' and 'lambda'
# This line selects the return of the ETF selected each month
strategy_df['Selected ETF Return'] = strategy_df.apply(
    lambda row: monthly_returns.loc[row.name, row['Selected ETF']], axis=1
)

# Handle any potential NaN values in returns (e.g., first month)
strategy_df['Selected ETF Return'] = strategy_df['Selected ETF Return'].fillna(0)

# Calculate cumulative strategy returns
strategy_df['Cumulative Strategy Return'] = (1 + strategy_df['Selected ETF Return']).cumprod()

# Align buy-and-hold cumulative returns to strategy start date
first_valid_date = strategy_df.index.min()
buy_hold_cumulative_returns = cumulative_returns.loc[first_valid_date:]
buy_hold_cumulative_returns = buy_hold_cumulative_returns.loc[strategy_df.index]

# Re-initialize buy_hold_cumulative_returns to start at 1
buy_hold_cumulative_returns = buy_hold_cumulative_returns.div(buy_hold_cumulative_returns.iloc[0])

# Compute cumulative returns for buy-and-hold to start at 1
# This ensures all cumulative return curves start at the same point
buy_hold_cumulative_returns = buy_hold_cumulative_returns.fillna(1)

# Performance metrics function
def compute_metrics_monthly(monthly_returns, cumulative_returns):
    # CAGR
    start_date = cumulative_returns.index.min()
    end_date = cumulative_returns.index.max()
    years = (end_date - start_date).days / 365.25
    ending_value = cumulative_returns.iloc[-1]
    cagr = (ending_value) ** (1 / years) - 1

    # Max Drawdown
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Volatility
    volatility = monthly_returns.std() * np.sqrt(12)  # Annualized for monthly data

    return cagr, max_drawdown, volatility

# Compute metrics for the strategy
strategy_cagr, strategy_max_drawdown, strategy_volatility = compute_metrics_monthly(
    strategy_df['Selected ETF Return'], strategy_df['Cumulative Strategy Return']
)

# Compute metrics for each ETF
performance_metrics = {}
for ticker in tickers:
    cagr, max_drawdown, volatility = compute_metrics_monthly(
        monthly_returns[ticker], buy_hold_cumulative_returns[ticker]
    )
    performance_metrics[ticker] = {
        'CAGR': cagr,
        'Max Drawdown': max_drawdown,
        'Volatility': volatility
    }

# Print Performance Metrics
print("Performance Metrics:")
print(f"Strategy:   CAGR: {strategy_cagr:.2%}, Max Drawdown: {strategy_max_drawdown:.2%}, Volatility: {strategy_volatility:.2%}")
for ticker in tickers:
    metrics = performance_metrics[ticker]
    print(f"{ticker}:     CAGR: {metrics['CAGR']:.2%}, Max Drawdown: {metrics['Max Drawdown']:.2%}, Volatility: {metrics['Volatility']:.2%}")

# --------------------- Enhancements Start Here ---------------------

# 1. Identify periods when the strategy is invested in SHY
strategy_df['Invested_in_SHY'] = strategy_df['Selected ETF'] == 'SHY'

# 2. Create a group identifier for consecutive SHY investments
strategy_df['Invested_in_SHY_Group'] = (strategy_df['Invested_in_SHY'] != strategy_df['Invested_in_SHY'].shift()).cumsum()

# 3. Filter only SHY investment periods
shy_periods_df = strategy_df[strategy_df['Invested_in_SHY']].copy()

# 4. Group by the consecutive SHY investment groups
grouped_shy = shy_periods_df.groupby('Invested_in_SHY_Group')

# 5. Extract start and end dates for each SHY investment period
periods = []

for name, group in grouped_shy:
    start = group.index.min()
    end = group.index.max()
    periods.append((start, end))

# 6. Plotting

# Initialize the plot with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# --- Plot 1: Cumulative Returns ---
ax1.plot(strategy_df.index, strategy_df['Cumulative Strategy Return'], label='Strategy Return', color='green', linewidth=2)

for ticker in tickers:
    ax1.plot(buy_hold_cumulative_returns.index, buy_hold_cumulative_returns[ticker], label=f'Buy & Hold {ticker}', linewidth=1.5)

# Shade the SHY investment periods
for start, end in periods:
    ax1.axvspan(start, end, color='yellow', alpha=0.3, label='Invested in SHY' if start == periods[0][0] else "")

ax1.set_title('Cumulative Returns: Strategy vs. Buy & Hold ETFs (Monthly, Log Scale)', fontsize=16, pad=20)
ax1.set_ylabel('Cumulative Return')
ax1.set_yscale('log')
ax1.axhline(1, color='black', linestyle='--', linewidth=1, label='Baseline (1)')
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.legend(loc='upper left', framealpha=0.9)

# --- Plot 2: Selected ETF Over Time ---
ax2.plot(strategy_df.index, strategy_df['Selected ETF'], label='Selected ETF', color='blue', linewidth=1)

# Shade the SHY investment periods
for start, end in periods:
    ax2.axvspan(start, end, color='yellow', alpha=0.3, label='Invested in SHY' if start == periods[0][0] else "")

ax2.set_title('Selected ETF Each Month Based on Average Rolling Returns', fontsize=16, pad=20)
ax2.set_ylabel('ETF Ticker')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_yticks(range(len(tickers)))
ax2.set_yticklabels(tickers)
ax2.grid(True, alpha=0.2)
ax2.legend(loc='upper left', framealpha=0.9)

# Improve date formatting
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%Y-%m')
plt.gca().xaxis.set_major_formatter(myFmt)

plt.tight_layout()
plt.show()

# --------------------- Additional Enhancements ---------------------

# 7. Create a Tabular View of Monthly Investments
# This table will show the selected ETF and whether it was invested in SHY each month.

investment_table = strategy_df[['Selected ETF', 'Invested_in_SHY']].copy()
investment_table.index.name = 'Date'
investment_table.reset_index(inplace=True)

# Display the table
print("\nMonthly Investment Strategy:")
print(investment_table.to_string(index=False))

# Optional: Save the table to a CSV file
# investment_table.to_csv('monthly_investments.csv', index=False)
