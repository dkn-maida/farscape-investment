import yfinance as yf
import matplotlib.pyplot as plt

# Fetch AAPL stock data
aapl = yf.download('^SPX', start='2000-01-01', end='2023-01-01')

# Calculate open-to-close returns
aapl['Open_Close_Returns'] = (aapl['Close'] - aapl['Open']) / aapl['Open']

# Identify days where open-to-close returns are less than -3%
significant_drop_days = aapl['Open_Close_Returns'] < -0.02

# Calculate returns for the following day
aapl['Next_Day_Returns'] = aapl['Open_Close_Returns'].shift(-1)

# Filter to get returns after a significant drop
next_day_returns_after_drop = aapl['Next_Day_Returns'][significant_drop_days]

# Plotting the distribution of next day returns after a significant drop
plt.figure(figsize=(10, 6))
next_day_returns_after_drop.hist(bins=50, alpha=0.6, color='purple')
plt.title('Distribution of AAPL Returns Following a >3% Drop Day')
plt.xlabel('Next Day Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


import yfinance as yf
import matplotlib.pyplot as plt

# Fetch S&P 500 stock data
sp500 = yf.download('^SPX', start='2000-01-01', end='2023-01-01')

# Calculate open-to-close returns
sp500['Open_Close_Returns'] = (sp500['Close'] - sp500['Open']) / sp500['Open']

# Identify days where open-to-close returns are less than -2%
significant_drop_days = sp500['Open_Close_Returns'] < -0.03

# Calculate returns for the following day
sp500['Next_Day_Returns'] = sp500['Open_Close_Returns'].shift(-1)

# Filter to get returns after a significant drop
next_day_returns_after_drop = sp500['Next_Day_Returns'][significant_drop_days]

# Compute the Expected Value (EV) of being invested the day after a -2% drop
ev_next_day_returns = next_day_returns_after_drop.mean()

# Display the EV
print(f"Expected Value of being invested the day after a -2% drop: {ev_next_day_returns}")

# Plotting the distribution of next day returns after a significant drop
plt.figure(figsize=(10, 6))
next_day_returns_after_drop.hist(bins=50, alpha=0.6, color='purple')
plt.title('Distribution of S&P 500 Returns Following a >2% Drop Day')
plt.xlabel('Next Day Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

