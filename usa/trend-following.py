import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Fetch TSLA hourly data
ticker = "TSLA"
data = yf.download(ticker, interval="1h", period="730d")

# Plot the stock price
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='TSLA Close Price')

# Compute and plot rate of change
for period in [60, 120, 240]:
    roc = data['Close'].pct_change(periods=period) * 100
    plt.plot(data.index, roc, label=f'Rate of Change {period}h')

# Compute and plot the 150h and 200h SMA
data['150h SMA'] = data['Close'].rolling(window=150).mean()
data['200h SMA'] = data['Close'].rolling(window=200).mean()
plt.plot(data['150h SMA'], label='150h SMA', alpha=0.7)
plt.plot(data['200h SMA'], label='200h SMA', alpha=0.7)

# Configuring plot
plt.title('TSLA Hourly Data with Rate of Change and SMA')
plt.legend()
plt.tight_layout()
plt.show()
