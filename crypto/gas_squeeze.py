import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

def compute_rsi(data, window):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    average_gain = up.rolling(window=window).mean()
    average_loss = -down.rolling(window=window).mean()
    
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Define the ticker symbol for BTC-USD
ticker_symbol = 'BTC-USD'

# Fetch hourly data
data = yf.download(ticker_symbol, start="2023-01-01", interval='1h')

# Calculate the 10-period SMA and Std Dev for the Bollinger Bands
data['SMA'] = data['Close'].rolling(window=10).mean()
data['Std Dev'] = data['Close'].rolling(window=10).std()

# Determine the 10-period Bollinger Bands
data['Upper Band'] = data['SMA'] + (2 * data['Std Dev'])
data['Lower Band'] = data['SMA'] - (2 * data['Std Dev'])
data['bbspread'] = data['Upper Band'] - data['Lower Band']

# Calculate the 10-period RSI of bbspread
data['bbspreadrsi'] = compute_rsi(data['bbspread'], 10)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot BTC-USD hourly closing prices on the first subplot
ax1.plot(data['Close'], color='tab:blue')
ax1.set_ylabel('Closing Price')
ax1.set_title(f'{ticker_symbol} Hourly Closing Prices')
ax1.grid(True)

# Plot bbspreadrsi on the second subplot
ax2.plot(data['bbspreadrsi'], color='tab:red')
ax2.set_xlabel('Date')
ax2.set_ylabel('bbspreadrsi')
ax2.set_title('10-period RSI of 10-period bbspread Indicator')
ax2.grid(True)

fig.tight_layout()
plt.show()


