import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Initialize the FRED API client (replace 'your_api_key_here' with your actual FRED API key)
fred = Fred(api_key='2f15c96b46530fde1b1992a64c64650e')

# Fetch Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL)
cpi_data = fred.get_series('INDCPIALLMINMEI')

# Convert the series into a DataFrame
cpi_df = pd.DataFrame(cpi_data, columns=['CPI'])

# Compute the natural logarithm of the CPI
cpi_df['log_CPI'] = np.log(cpi_df['CPI'])
cpi_df = cpi_df.dropna(subset=['log_CPI'])

# Linear regression model requires 2D array, so we'll convert our dates to numbers and reshape
X = cpi_df.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)  # independent variable (time)
y = cpi_df['log_CPI'].values  # dependent variable (log_CPI)

# Train the 1-year model
model_1 = LinearRegression()
model_1.fit(X[-12:], y[-12:])  # Use the last 12 months of data

# Train the 7-year model
model_7 = LinearRegression()
model_7.fit(X[-12*7:], y[-12*7:])  # Use the last 7*12 months of data

# Compute predictions
y_pred_1 = model_1.predict(X)
y_pred_7 = model_7.predict(X)

# Add predictions to the DataFrame
cpi_df['1_year_regression'] = y_pred_1
cpi_df['7_year_regression'] = y_pred_7

# Plot the natural log of the CPI data and the regressions
plt.figure(figsize=(14, 7))
plt.plot(cpi_df.index, cpi_df['log_CPI'], label='log(CPI)')
plt.plot(cpi_df.index, cpi_df['1_year_regression'], label='1-year regression', linestyle='--')
plt.plot(cpi_df.index, cpi_df['7_year_regression'], label='7-year regression', linestyle='--')
plt.title('Natural Logarithm of Consumer Price Index and Linear Regressions over Time')
plt.xlabel('Year')
plt.ylabel('log(CPI)')
plt.legend()
plt.show()

# Loop through each data point
for i in range(len(cpi_df)):
    
    # Check if there's enough data for the 1-year model
    if i >= 12:
        # Train the 1-year model
        model_1 = LinearRegression()
        X = cpi_df.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)[:i]  # independent variable (time)
        y = cpi_df['log_CPI'].values[:i]  # dependent variable (log_CPI)
        model_1.fit(X[-12:], y[-12:])  # Use the last 12 months of data
        # Predict for the current point
        cpi_df.loc[cpi_df.index[i], '1_year_regression'] = model_1.predict([X[-1]])
    
    # Check if there's enough data for the 7-year model
    if i >= 12 * 7:
        # Train the 7-year model
        model_7 = LinearRegression()
        model_7.fit(X[-12*7:], y[-12*7:])  # Use the last 7*12 months of data
        # Predict for the current point
        cpi_df.loc[cpi_df.index[i], '7_year_regression'] = model_7.predict([X[-1]])
        
    # Compute the indicator
    if not np.isnan(cpi_df.loc[cpi_df.index[i], '1_year_regression']) and not np.isnan(cpi_df.loc[cpi_df.index[i], '7_year_regression']):
        cpi_df.loc[cpi_df.index[i], 'indicator'] = int(cpi_df.loc[cpi_df.index[i], '1_year_regression'] < cpi_df.loc[cpi_df.index[i], '7_year_regression'])


# Plot the natural log of the CPI data, the regressions, and the indicator
plt.figure(figsize=(14, 7))
plt.plot(cpi_df.index, cpi_df['log_CPI'], label='log(CPI)')
plt.plot(cpi_df.index, cpi_df['1_year_regression'], label='1-year regression', linestyle='--')
plt.plot(cpi_df.index, cpi_df['7_year_regression'], label='7-year regression', linestyle='--')
plt.title('Natural Logarithm of Consumer Price Index and Linear Regressions over Time')
plt.xlabel('Year')
plt.ylabel('log(CPI)')
plt.legend(loc='upper left')
plt.twinx()
plt.plot(cpi_df.index, cpi_df['indicator'], label='Indicator (1-year < 7-year)', color='red', alpha=0.5)
plt.ylabel('Indicator')
plt.legend(loc='upper right')
plt.show()