import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Download historical data for AAPL (weekly data)
aapl_data = yf.download('TSLA', start='2020-01-01', end='2023-12-26', interval='1wk')

# Calculate the 4-week moving average
aapl_data['4_WK_MA'] = aapl_data['Close'].rolling(window=3).mean()

# Define the trading strategy for weekly data
def backtest_strategy_weekly(data):
    trades = pd.DataFrame(index=data.index)
    trades['returns'] = np.nan

    for i in range(1, len(data)):
        if data['Open'][i] > data['4_WK_MA'][i-1]:
            trades['returns'][i] = data['Close'][i] / data['Open'][i] - 1

    return trades

# Run the backtest on weekly data
trades_weekly = backtest_strategy_weekly(aapl_data)

# Calculate cumulative returns
trades_weekly['cumulative_returns'] = (1 + trades_weekly['returns']).cumprod()

# Calculate cumulative returns for the AAPL stock as a benchmark (weekly data)
aapl_data['stock_cumulative_returns'] = (1 + aapl_data['Close'].pct_change()).cumprod()

# Remove NaN values for plotting
trades_weekly.dropna(inplace=True)

# Plot the cumulative returns and the AAPL stock curve (weekly data)
plt.figure(figsize=(12, 6))
plt.plot(trades_weekly.index, trades_weekly['cumulative_returns'], label='Strategy Cumulative Returns')
plt.plot(aapl_data.index, aapl_data['stock_cumulative_returns'], label='AAPL Stock Cumulative Returns', alpha=0.7)
plt.title('Cumulative Returns of AAPL Trading Strategy vs AAPL Stock (Weekly Data)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()
