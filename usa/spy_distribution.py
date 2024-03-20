import yfinance as yf
import matplotlib.pyplot as plt

# Fetch data for SPY
spy = yf.Ticker("SPY")
data = spy.history(period="5y")  # Adjust the period as needed

# Plot the histogram of intraday highs
plt.hist( (data['High']-data['Open'])/data['Open']*100, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of SPY Intraday Highs Distribution')
plt.xlabel('Intraday Highs')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

plt.show()


# Calculate the percentage difference from high to close
data['HighClosePercent'] = ((data['High'] - data['Close']) / data['Close']) * 100

# Determine the number of days where this percentage is greater than 0.5%
days_over_half_percent = data[data['HighClosePercent'] > 0.3]

# Calculate the percentage of such days
percentage_days_over_half_percent = (len(days_over_half_percent) / len(data)) * 100

print(f"Percentage of days where high - close > 0.3%: {percentage_days_over_half_percent:.2f}%")
