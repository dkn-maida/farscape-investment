import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Download SPY and oil price data with monthly frequency
spy = yf.download('SPY', start='2000-01-01', end='2023-12-31', interval='1mo', progress=False)
oil = yf.download('CL=F', start='2000-01-01', end='2023-12-31', interval='1mo', progress=False)  # WTI Crude Oil Futures

# Interpolate missing values in oil data
oil['Adj Close'] = oil['Adj Close'].interpolate()

# Calculate rolling returns
def calculate_rolling_returns(data, window):
    return data['Adj Close'].pct_change(periods=window).shift(-window) * 100

# 1-month and 12-month rolling returns
spy['1M Rolling'] = calculate_rolling_returns(spy, 1)
oil['1M Rolling'] = calculate_rolling_returns(oil, 1)
spy['12M Rolling'] = calculate_rolling_returns(spy, 12)
oil['12M Rolling'] = calculate_rolling_returns(oil, 12)

# Calculate ratios
spy_1m_decimal = 1 + (spy['1M Rolling'] / 100)
oil_1m_decimal = 1 + (oil['1M Rolling'] / 100)
spy_12m_decimal = 1 + (spy['12M Rolling'] / 100)
oil_12m_decimal = 1 + (oil['12M Rolling'] / 100)

# Avoid division by zero or negative values
ratio_1m = spy_1m_decimal / oil_1m_decimal
ratio_12m = spy_12m_decimal / oil_12m_decimal
ratio_1m_to_12m = ratio_1m / ratio_12m

# Take the logarithm of ratios to reduce the effect of outliers
log_ratio_1m = np.log(ratio_1m)
log_ratio_12m = np.log(ratio_12m)
log_ratio_1m_to_12m = np.log(ratio_1m_to_12m)

# Drop NaN or infinite values
log_ratio_1m.dropna(inplace=True)
log_ratio_12m.dropna(inplace=True)
log_ratio_1m_to_12m.dropna(inplace=True)

# Calculate 3-month rolling returns of the 1-month ratio
rolling_return_3m = ratio_1m.pct_change(periods=3) * 100
rolling_return_3m.dropna(inplace=True)

# Plot SPY price, ratios, and 3-month rolling returns of the ratio
plt.figure(figsize=(15, 25))

# SPY Adjusted Closing Price
plt.subplot(5, 1, 1)
plt.plot(spy.index, spy['Adj Close'], label='SPY Price', color='black', alpha=0.8)
plt.title('SPY Adjusted Closing Price (Monthly Data)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()

# Logarithm of 1-Month Ratio
plt.subplot(5, 1, 2)
plt.plot(log_ratio_1m.index, log_ratio_1m, label='Log of 1-Month Ratio', color='blue', alpha=0.7)
plt.title('Logarithm of SPY to Oil Rolling Return Ratio (1 Month)')
plt.ylabel('Log Ratio (1M)')
plt.legend()
plt.grid()

# Logarithm of 12-Month Ratio
plt.subplot(5, 1, 3)
plt.plot(log_ratio_12m.index, log_ratio_12m, label='Log of 12-Month Ratio', color='green', alpha=0.7)
plt.title('Logarithm of SPY to Oil Rolling Return Ratio (12 Months)')
plt.ylabel('Log Ratio (12M)')
plt.legend()
plt.grid()

# Logarithm of 1-Month to 12-Month Ratio
plt.subplot(5, 1, 4)
plt.plot(log_ratio_1m_to_12m.index, log_ratio_1m_to_12m, label='Log of 1M to 12M Ratio', color='purple', alpha=0.7)
plt.title('Logarithm of 1-Month to 12-Month Rolling Return Ratio')
plt.ylabel('Log Ratio (1M/12M)')
plt.legend()
plt.grid()

# 3-Month Rolling Returns of 1-Month Ratio
plt.subplot(5, 1, 5)
plt.plot(rolling_return_3m.index, rolling_return_3m, label='3-Month Rolling Returns of 1M Ratio', color='orange', alpha=0.7)
plt.title('3-Month Rolling Returns of SPY to Oil 1-Month Ratio')
plt.ylabel('Rolling Return (%)')
plt.legend()
plt.grid()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
