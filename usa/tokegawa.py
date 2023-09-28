import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch data using yfinance
df = yf.download('AAPL', start='2000-01-01', end='2023-08-17')
df['25_MA'] = df['Close'].rolling(window=25).mean()

# Buy when the close price is 25% below the 25-day MA
df['Buy_Signal'] = np.where(df['Close'] < df['25_MA'] * 0.75, 1, 0)

# Sell when the close price crosses above the 25-day MA
df['Sell_Signal'] = np.where(df['Close'] > df['25_MA'], 1, 0)

# Backtesting
initial_balance = 10000
balance = initial_balance
in_position = False
for index, row in df.iterrows():
    if row['Buy_Signal'] == 1 and not in_position:
        # Buy and deduct from balance
        balance -= row['Close']
        in_position = True
    elif row['Sell_Signal'] == 1 and in_position:
        # Sell and add to balance
        balance += row['Close']
        in_position = False

# Add back the last position if still holding
if in_position:
    balance += df.iloc[-1]['Close']

# Results
print(f'Initial Balance: ${initial_balance}')
print(f'Final Balance: ${balance}')

# Visualization
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
plt.plot(df.index, df['25_MA'], label='25 Days MA', alpha=0.8)
plt.scatter(df.index, df[df['Buy_Signal'] == 1]['Close'], label='Buy Signal', marker='^', alpha=1, color='green')
plt.scatter(df.index, df[df['Sell_Signal'] == 1]['Close'], label='Sell Signal', marker='v', alpha=1, color='red')
plt.title('Apple Stock and Buy/Sell signals')
plt.legend()
plt.show()
