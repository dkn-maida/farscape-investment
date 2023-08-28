import yfinance as yf
import matplotlib.pyplot as plt

# Fetch EUR/USD hourly data from Yahoo Finance
symbol = "EURUSD=X"
start_date = "2022-01-01"
end_date = "2023-08-27"  # You can adjust this to the current date
interval = "1h"

data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

# Compute SMA for 150 and 200 periods
data['SMA_150'] = data['Close'].rolling(window=150).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Compute Rate of Change for 240 and 120 periods
data['ROC_240'] = data['Close'].pct_change(periods=240) * 100
data['ROC_120'] = data['Close'].pct_change(periods=120) * 100

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

# Plot EUR/USD and SMAs
ax1.plot(data['Close'], label='EUR/USD', color='blue')
ax1.plot(data['SMA_150'], label='SMA 150', color='green')
ax1.plot(data['SMA_200'], label='SMA 200', color='red')
ax1.set_title('EUR/USD Hourly Data with SMA Indicators')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()

# Plot Rate of Change indicators
ax2.plot(data['ROC_240'], label='ROC 240', color='purple')
ax2.plot(data['ROC_120'], label='ROC 120', color='orange')
ax2.set_title('Rate of Change Indicators')
ax2.set_xlabel('Date')
ax2.set_ylabel('Rate of Change (%)')
ax2.legend()

plt.show()
