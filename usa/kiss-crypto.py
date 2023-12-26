import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download historical data for BTC-USD
btc_data = yf.download('BTC-USD', start='2023-10-20', interval="5m")

# Calculate the 21-day moving average
btc_data['MA21'] = btc_data['Close'].rolling(window=10).mean()

# Create a signal for when to buy and sell
# Buy (1) when the close price is above the 21-day MA, sell (-1) when below
btc_data['Signal'] = 0 
btc_data.loc[btc_data['Close'] > btc_data['MA21'], 'Signal'] = 1
#btc_data.loc[(btc_data['Close'] > btc_data['MA21']) & (btc_data['MA21'] > btc_data['MA63']), 'Signal'] = 1
#btc_data.loc[btc_data['Close'] < btc_data['MA21'], 'Signal'] = -1

# Calculate daily returns
btc_data['Daily_Return'] = btc_data['Close'].pct_change()
# Calculate strategy returns
btc_data['Strategy_Return'] = btc_data['Daily_Return'] * btc_data['Signal'].shift(1)

# Calculate cumulative returns
btc_data['Cumulative_Return'] = (1 + btc_data['Strategy_Return']).cumprod()

# Plotting the strategy cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(btc_data['Cumulative_Return'], label='Strategy Cumulative Returns')
plt.title('BTC-USD MA21 Strategy Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()