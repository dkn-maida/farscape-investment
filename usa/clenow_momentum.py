import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Fetch the last 126 days of NVDA stock data
nvda_data = yf.download('META', period='126d', interval='1d')

# Keep only the closing prices
closing_prices = nvda_data['Close']

# Apply natural logarithm to get exponential transformation
log_prices = np.log(closing_prices)

# Prepare data for regression
X = np.arange(len(log_prices)).reshape(-1, 1)  # Independent variable (time)
y = log_prices.values.reshape(-1, 1)          # Dependent variable (log prices)

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Calculate the slope (coefficient) and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Calculate R-squared
r_squared = model.score(X, y)

# Create a line for the linear regression
predicted_line = model.predict(X)

# Plot the closing prices and the regression line
plt.figure(figsize=(12, 6))
plt.plot(closing_prices.index, closing_prices, label='NVDA Closing Prices', color='blue')
plt.plot(closing_prices.index, np.exp(predicted_line), label='Exponential Regression Line', color='red')

plt.title('NVDA Closing Prices and Exponential Linear Regression')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

