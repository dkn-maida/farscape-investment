import yfinance as yf
import matplotlib.pyplot as plt

# Fetch data for SPY
spy = yf.Ticker("SPY")
data = spy.history(period="30mo")  # Adjust the period as needed

# Plot the histogram of intraday highs
plt.hist( (data['Low']-data['Open'])/data['Open']*100, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of SPY Intraday Lows Distribution')
plt.xlabel('Intraday Highs')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

plt.show()


# Plot the histogram of intraday highs
plt.hist( (data['High']-data['Open'])/data['Open']*100, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of SPY Intraday Highs Distribution')
plt.xlabel('Intraday Highs')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

plt.show()


# Plot the histogram of intraday highs
plt.hist( (data['Close']-data['Open'])/data['Open']*100, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of SPY Intraday Close Distribution')
plt.xlabel('Intraday Close')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

plt.show()



# Calculate the percentage difference from high to close
data['HighClosePercent'] = ((data['High'] - data['Open']) / data['Open']) * 100
# Determine the number of days where this percentage is greater than 0.5%
days_over_half_percent = data[(data['HighClosePercent'] >= 0.5) & (data['HighClosePercent'] <= 1.5)]
# Calculate the percentage of such days
percentage_days_over_half_percent = (len(days_over_half_percent) / len(data)) * 100
print(f"Percentage of days where high - close > 0.5%: {percentage_days_over_half_percent:.2f}%")

# Calculate the percentage difference from high to close
data['LowClosePercent'] = ((data['Low'] - data['Open']) / data['Open']) * 100
# Determine the number of days where this percentage is greater than 0.5%
days_over_half_percent = data[(data['LowClosePercent'] <= -0.5) & (data['LowClosePercent'] >= -1.5)]
# Calculate the percentage of such days
percentage_days_over_half_percent = (len(days_over_half_percent) / len(data)) * 100
print(f"Percentage of days where low - close < 0.5%: {percentage_days_over_half_percent:.2f}%")


# Calculate the percentage difference from high to close
data['closePercent'] = ((data['Close'] - data['Open']) / data['Open']) * 100
# Determine the number of days where this percentage is greater than 0.5%
days_over_half_percent = data[(data['closePercent'] > 0.3 ) | (data['closePercent'] < -0.3)]
# Calculate the percentage of such days
percentage_days_over_half_percent = (len(days_over_half_percent) / len(data)) * 100
print(f"Percentage of days where open - close between 0.4%: {percentage_days_over_half_percent:.2f}%")


# Fetch data for SPY
spy = yf.Ticker("SPY")

# Get intraday data, adjust the interval and period as needed
data = spy.history(interval="1h", period="6mo")
data = data.between_time('9:30', '12:30')  # First hour of trading

# Calculate the percentage changes relative to the opening price
data['HighPercent'] = (data['High'] - data['Open']) / data['Open'] * 100
data['LowPercent'] = (data['Low'] - data['Open']) / data['Open'] * 100


# Plot histogram of intraday highs
plt.hist(data['HighPercent'], bins=100, color='blue', alpha=0.7)
plt.title('Histogram of SPY Intraday Highs during First Hour')
plt.xlabel('Percentage Change from Open to High')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Plot histogram of intraday lows
plt.hist(data['LowPercent'], bins=100, color='red', alpha=0.7)
plt.title('Histogram of SPY Intraday Lows during First Hour')
plt.xlabel('Percentage Change from Open to Low')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


# Insights based on conditions for Highs and Lows
highs_over_1_percent = data[data['HighPercent'] > 0.5]
lows_under_minus_1_percent = data[data['LowPercent'] < -0.5]

# Calculate percentages for insights
percentage_highs_over_1 = len(highs_over_1_percent) / len(data) * 100
percentage_lows_under_minus_1 = len(lows_under_minus_1_percent) / len(data) * 100

print(f"Percentage of days with high in the first 3 hours > 0.5% above open: {percentage_highs_over_1:.2f}%")
print(f"Percentage of days with low in the first 3 hours > 0.5% below open: {percentage_lows_under_minus_1:.2f}%")

# Insights based on conditions for Highs and Lows
highs_over_1_percent = data[data['HighPercent'] >= 1]
lows_under_minus_1_percent = data[data['LowPercent'] <= -1]

# Calculate percentages for insights
percentage_highs_over_1 = len(highs_over_1_percent) / len(data) * 100
percentage_lows_under_minus_1 = len(lows_under_minus_1_percent) / len(data) * 100

print(f"Percentage of days with high > 1% above open: {percentage_highs_over_1:.2f}%")
print(f"Percentage of days with low > 1% below open: {percentage_lows_under_minus_1:.2f}%")
