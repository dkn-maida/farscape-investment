import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Fetch the SPY data using yfinance
ticker = "SPY"
data = yf.download(ticker, start="2000-01-01", end="2024-12-31")

# Step 2: Calculate the 200-day moving average (SMA)
data['SMA_200'] = data['Adj Close'].rolling(window=200).mean()

# Step 3: Calculate the slope of the 200-day SMA based on the last 5 days
data['SMA_Slope'] = data['SMA_200'].diff(5)

# Step 4: Create a signal: 1 if price is above the 200-day SMA and the slope is positive, otherwise 0
data['Signal'] = np.where((data['Adj Close'] > data['SMA_200'].shift(1)) & (data['SMA_Slope'] > 0), 1, 0)

# Step 5: Calculate daily returns
data['Daily_Return'] = data['Adj Close'].pct_change()

# Step 6: Calculate strategy returns based on the signal
data['Strategy_Return'] = data['Daily_Return'] * data['Signal'].shift(1)  # Shift to apply today's signal tomorrow

# Step 7: Calculate cumulative returns
data['Cumulative_Market_Return'] = (1 + data['Daily_Return']).cumprod()
data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()

# Step 8: Calculate CAGR
years = (data.index[-1] - data.index[0]).days / 365.25
cagr_market = (data['Cumulative_Market_Return'].iloc[-1]) ** (1/years) - 1
cagr_strategy = (data['Cumulative_Strategy_Return'].iloc[-1]) ** (1/years) - 1

# Step 9: Calculate max drawdown
data['Market_Drawdown'] = data['Cumulative_Market_Return'] / data['Cumulative_Market_Return'].cummax() - 1
data['Strategy_Drawdown'] = data['Cumulative_Strategy_Return'] / data['Cumulative_Strategy_Return'].cummax() - 1
max_drawdown_market = data['Market_Drawdown'].min()
max_drawdown_strategy = data['Strategy_Drawdown'].min()

# Step 10: Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(data['Cumulative_Market_Return'], label='Market Return (Buy and Hold)', color='blue')
plt.plot(data['Cumulative_Strategy_Return'], label='Strategy Return (200-day SMA with Slope Filter)', color='green')
plt.title('Market vs Strategy Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Print final metrics
print(f"Final Cumulative Market Return: {data['Cumulative_Market_Return'].iloc[-1]:.2f}")
print(f"Final Cumulative Strategy Return: {data['Cumulative_Strategy_Return'].iloc[-1]:.2f}")
print(f"CAGR (Market): {cagr_market:.2%}")
print(f"CAGR (Strategy): {cagr_strategy:.2%}")
print(f"Max Drawdown (Market): {max_drawdown_market:.2%}")
print(f"Max Drawdown (Strategy): {max_drawdown_strategy:.2%}")
