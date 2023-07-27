import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Download historical data
data = yf.download('BTC-USD', start='2023-01-01')

# Calculate the 10-day high and 10-day low
data['10_day_high'] = data['Close'].rolling(window=1).max()
data['10_day_low'] = data['Close'].rolling(window=1).min()

# Create a signal when the close price crosses the 10_day_high
data['buy_signal_one'] = np.where(data['Close'] > data['10_day_high'].shift(), 1, 0)

# Create a column that contains the price at which we last bought
data['buy_price'] = np.where(data['buy_signal_one'], data['Close'], np.nan)
data['buy_price'].ffill(inplace=True)

# Create a signal when the high price reaches 4% above the buy price or we exit at the close price
data['take_profit_signal'] = np.where(data['High'] > data['buy_price'] * 1.015, 1, 0).astype(int)
data['exit_signal'] = data['buy_signal_one'].shift().fillna(0).astype(int)
data['sell_signal_one'] = np.where(data['take_profit_signal'] | data['exit_signal'], 1, 0).astype(int)

# Create a signal when the close price drops below the 10_day_low and the first strategy is not active
data['buy_signal_two'] = np.where((data['Close'] < data['10_day_low'].shift()) & (data['buy_signal_one'] == 0), 1, 0)

# Sell the next day after buying using the second strategy
data['sell_signal_two'] = data['buy_signal_two'].shift()

# Combine the buy and sell signals
data['buy_signal'] = np.where(data['buy_signal_one'] == 1, 1, data['buy_signal_two'])
data['sell_signal'] = np.where(data['sell_signal_one'] == 1, 1, data['sell_signal_two'])

# Calculate daily percentage returns of strategy
data['strategy_returns'] = np.where(data['sell_signal'] == 1, np.where(data['take_profit_signal'] == 1, data['Close'].pct_change(), 0.015 ), 0)

# Calculate cumulative returns of strategy
data['strategy_cumulative_returns'] = (1 + data['strategy_returns']).cumprod()

# Calculate CAGR
years = (data.index[-1] - data.index[0]).days / 365.25
cagr = (data['strategy_cumulative_returns'][-1]) ** (1 / years) - 1
print("CAGR: ", cagr)

# Calculate max drawdown
rolling_max = data['strategy_cumulative_returns'].cummax()
daily_drawdown = data['strategy_cumulative_returns'] / rolling_max - 1.0
max_daily_drawdown = daily_drawdown.min()
print("Max Drawdown: ", max_daily_drawdown)

# Calculate the number of trades
number_of_trades = data['sell_signal'].sum()
print("Number of trades: ", number_of_trades)

# Calculate the percentage of winning and losing trades
trades = data.loc[data['sell_signal'] == 1, 'strategy_returns']
winning_trades = trades[trades > 0]
losing_trades = trades[trades < 0]

winning_trades_percentage = len(winning_trades) / number_of_trades if number_of_trades != 0 else 0
losing_trades_percentage = len(losing_trades) / number_of_trades if number_of_trades != 0 else 0

print("Winning trades percentage: ", winning_trades_percentage)
print("Losing trades percentage: ", losing_trades_percentage)

# Calculate the average win and average loss
average_win = winning_trades.mean()
average_loss = losing_trades.mean()
print("Average win: ", average_win)
print("Average loss: ", average_loss)

# Plot the equity curve of the strategy
plt.figure(figsize=(10,5))
plt.plot(data['strategy_cumulative_returns'], label='Combined Strategy')
plt.legend()
plt.show()