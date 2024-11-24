import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from itertools import permutations

# Configure seaborn aesthetics
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Define currencies
currencies = ["EUR", "USD", "JPY", "CNY", "GBP", "CAD", "AUD", "CHF"]

# Generate all currency pairs
currency_pairs = [f"{base}{quote}=X" for base, quote in permutations(currencies, 2)]

# Fetch FX data
def fetch_data(pairs, start, end):
    data = {}
    for pair in pairs:
        df = yf.download(pair, start=start, end=end)['Adj Close']
        data[pair] = df
    return pd.DataFrame(data)

# Calculate momentum signals
def calculate_signals(data, lookback=[2, 3, 4]):
    signals = pd.DataFrame(index=data.index)
    for col in data.columns:
        prices = data[col]
        is_bullish = np.logical_and.reduce(
            [prices > prices.shift(months) for months in lookback]
        )
        is_bearish = np.logical_and.reduce(
            [prices < prices.shift(months) for months in lookback]
        )
        signals[col] = np.where(is_bullish, 1, np.where(is_bearish, -1, 0))
    return signals

# Calculate volatility (standard deviation of returns)
def calculate_volatility(data, window=3):
    return data.pct_change().rolling(window=window).std()

# Simulate strategy with fixed 4x leverage and volatility filter
def backtest(data, signals, volatility, vol_threshold=0.03):
    positions = signals.shift(1)  # Use signals from the previous month to avoid look-ahead bias
    monthly_returns = data.pct_change()  # Monthly percentage returns
    
    strategy_returns = pd.Series(0, index=data.index)
    for date in data.index:
        # Skip trading if volatility exceeds the threshold
        if volatility.loc[date].max() > vol_threshold:
            continue
        
        active_pairs = positions.loc[date] != 0  # Identify active positions (bullish or bearish)
        num_active_pairs = active_pairs.sum()  # Count active pairs
        
        if num_active_pairs > 0:
            active_returns = monthly_returns.loc[date, active_pairs]
            active_positions = positions.loc[date, active_pairs]
            # Apply 4x leverage split equally among active pairs
            leverage_per_pair = 4 / num_active_pairs
            portfolio_return = (active_returns * active_positions * leverage_per_pair).sum()
            strategy_returns.loc[date] = portfolio_return
    
    return strategy_returns, positions

# Calculate drawdowns
def calculate_drawdowns(cumulative_returns):
    peaks = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - peaks) / peaks
    return drawdowns

# Parameters
start_date = "2010-01-01"
end_date = "2023-01-01"

# Fetch FX data
fx_data = fetch_data(currency_pairs, start=start_date, end=end_date)

# Resample to monthly frequency
fx_data_monthly = fx_data.resample('M').last()

# Generate signals
signals = calculate_signals(fx_data_monthly)

# Calculate volatility
volatility = calculate_volatility(fx_data_monthly)

# Backtest strategy
strategy_returns, positions = backtest(fx_data_monthly, signals, volatility)

# Calculate cumulative returns
cumulative_returns = (1 + strategy_returns).cumprod()

# Calculate drawdowns
drawdowns = calculate_drawdowns(cumulative_returns)
max_drawdown = drawdowns.min()

# Performance metrics
annualized_return = (cumulative_returns.iloc[-1] ** (1 / (len(cumulative_returns) / 12))) - 1
annualized_volatility = strategy_returns.std() * np.sqrt(12)
sharpe_ratio = annualized_return / annualized_volatility

# Determine current strategy allocation
current_signals = signals.iloc[-1]  # Latest signals
allocation_today = ", ".join(
    f"{'Long' if val == 1 else 'Short' if val == -1 else 'Neutral'} {pair}"
    for pair, val in current_signals.items()
)

# Plot cumulative returns, drawdowns, and performance metrics
fig, axes = plt.subplots(3, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [3, 2, 1]})

# Cumulative returns plot
sns.lineplot(ax=axes[0], x=cumulative_returns.index, y=cumulative_returns.values, label="Cumulative Returns", color="blue")
axes[0].set_title("Cumulative Returns")
axes[0].set_ylabel("Cumulative Returns")
axes[0].legend()
axes[0].grid()
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format x-axis as years

# Drawdowns plot
sns.lineplot(ax=axes[1], x=drawdowns.index, y=drawdowns.values, label="Drawdowns", color="red")
axes[1].set_title("Drawdowns Over Time")
axes[1].set_ylabel("Drawdown (%)")
axes[1].legend()
axes[1].grid()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format x-axis as years

# Performance metrics and current allocation
axes[2].axis("off")
metrics_text = (
    f"Performance Metrics:\n"
    f"Annualized Return: {annualized_return:.2%}\n"
    f"Annualized Volatility: {annualized_volatility:.2%}\n"
    f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
    f"Maximum Drawdown: {max_drawdown:.2%}\n\n"
    f"Current Allocation: {allocation_today}"
)
axes[2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment="center", horizontalalignment="left", family="monospace")

# Adjust layout
plt.tight_layout()
plt.show()

# Plot the rotation (trade positions) heatmap in a separate window
plt.figure(figsize=(16, 8))
sns.heatmap(
    positions.T,
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Position (-1: Short, 1: Long)"},
    xticklabels=False  # Remove x-axis labels (dates)
)
plt.title("Trade Positions Over Time")
plt.ylabel("Currency Pair")
plt.xlabel("Date")
plt.tight_layout()
plt.show()
