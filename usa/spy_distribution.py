import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Fetch data for SPY
spy = yf.Ticker("SPY")

# Get intraday data using 1-hour intervals
data = spy.history(interval="1h", period="20mo")  # Adjust the period as needed

# Resample the data to 4-hour intervals, calculating the max for highs and min for lows
four_hour_highs = data['High'].resample('4H').max()
four_hour_lows = data['Low'].resample('4H').min()
four_hour_opens = data['Open'].resample('4H').first()

# Calculate the percentage changes for these 4-hour periods
four_hour_data = pd.DataFrame({
    'Open': four_hour_opens,
    'High': four_hour_highs,
    'Low': four_hour_lows
})
four_hour_data['HighPercent'] = (four_hour_data['High'] - four_hour_data['Open']) / four_hour_data['Open'] * 100
four_hour_data['LowPercent'] = (four_hour_data['Low'] - four_hour_data['Open']) / four_hour_data['Open'] * 100

# Plot histogram of 4-hour interval highs
plt.hist(four_hour_data['HighPercent'], bins=100, color='blue', alpha=0.7)
plt.title('Histogram of SPY 4-Hour Interval Highs')
plt.xlabel('Percentage Change from Open to High')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Plot histogram of 4-hour interval lows
plt.hist(four_hour_data['LowPercent'], bins=100, color='red', alpha=0.7)
plt.title('Histogram of SPY 4-Hour Interval Lows')
plt.xlabel('Percentage Change from Open to Low')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Insights based on conditions for Highs and Lows
highs_over_point_five_percent = four_hour_data[four_hour_data['HighPercent'] >= 0.5]
lows_under_minus_point_five_percent = four_hour_data[four_hour_data['LowPercent'] <= -0.5]

# Calculate percentages for insights
percentage_highs_over_point_five = len(highs_over_point_five_percent) / len(four_hour_data) * 100
percentage_lows_under_minus_point_five = len(lows_under_minus_point_five_percent) / len(four_hour_data) * 100

print(f"Percentage of 4-hour intervals with high > 0.3% above open: {percentage_highs_over_point_five:.2f}%")
print(f"Percentage of 4-hour intervals with low > 0.3% below open: {percentage_lows_under_minus_point_five:.2f}%")
