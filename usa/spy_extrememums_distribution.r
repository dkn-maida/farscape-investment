import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetch historical data for SPY
spy = yf.Ticker("SPY")
data = spy.history(period="1y")  # You can adjust the period as needed (e.g., "5y", "max")

# Calculate the ratios in percentages
data['High_Open_Percent'] = ((data['High'] - data['Open']) / data['Open']) * 100
data['Low_Open_Percent'] = ((data['Low'] - data['Open']) / data['Open']) * 100

# Calculate the percentage of days where the maximum variation stays between -0.3% and 0.3%
within_range = ((data['High_Open_Percent'] <= 0.7) & (data['High_Open_Percent'] >= -0.7) &
                (data['Low_Open_Percent'] <= 0.7 ) & (data['Low_Open_Percent'] >= -0.7)).mean() * 100

# Print statistical summary
print("High-Open/Open Percentage Statistics:")
print(data['High_Open_Percent'].describe())

print("Low-Open/Open Percentage Statistics:")
print(data['Low_Open_Percent'].describe())

print(f"\nPercentage of days with maximum variation (High-Open/Open or Low-Open/Open) between range {within_range:.2f}%")

# Plot the distribution of the ratios
plt.figure(figsize=(14, 7))

# Plot the distribution of High-Open/Open percentage
plt.subplot(1, 2, 1)
plt.hist(data['High_Open_Percent'], bins=50, color='blue', alpha=0.7)
plt.title('Distribution of (High-Open)/Open Percentage')
plt.xlabel('Percentage')
plt.ylabel('Frequency')

# Plot the distribution of Low-Open/Open percentage
plt.subplot(1, 2, 2)
plt.hist(data['Low_Open_Percent'], bins=50, color='red', alpha=0.7)
plt.title('Distribution of (Low-Open)/Open Percentage')
plt.xlabel('Percentage')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
