import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# Initialize the FRED API client
fred = Fred(api_key='2f15c96b46530fde1b1992a64c64650e')  # Replace with your FRED API key

# Fetch Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL)
cpi_data = fred.get_series('INDCPIALLMINMEI')

# Convert the series into a DataFrame
cpi_df = pd.DataFrame(cpi_data, columns=['CPI'])

# Compute the natural logarithm of the CPI
cpi_df['log_CPI'] = np.log(cpi_df['CPI'])
cpi_df = cpi_df.dropna(subset=['log_CPI'])

# Merge datasets
combined_df = cpi_df.resample('M').last().fillna(method='ffill')

# Fetch historical data for SPY (daily data)
spy_data_daily = yf.download('SPY', start=combined_df.index[0], end=combined_df.index[-1])
spy_data = spy_data_daily['Adj Close'].resample('M').last()

# Compute daily returns
combined_df['spy_returns'] = spy_data.pct_change()

# Compute the indicator for each data point
for i in range(12*7, len(combined_df)):
    X = combined_df.index[:i].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = combined_df['log_CPI'].iloc[:i].values

    # Standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    model_1 = LinearRegression()
    model_1.fit(X[-12:], y[-12:])
    
    model_7 = LinearRegression()
    model_7.fit(X[-12*7:], y[-12*7:])
    
    y_pred_1 = model_1.predict(X[-12:])
    y_pred_7 = model_7.predict(X[-12*7:])
    
    combined_df.loc[combined_df.index[i], 'indicator'] = 1 if y_pred_7[-1] > y_pred_1[-1] else 0

# Compute strategy returns
combined_df['strategy_returns'] = combined_df['spy_returns'] * combined_df['indicator']

# Compute cumulative returns
combined_df['cumulative_spy_returns'] = (1 + combined_df['spy_returns']).cumprod()
combined_df['cumulative_strategy_returns'] = (1 + combined_df['strategy_returns']).cumprod()

# Plot the cumulative returns
fig, ax1 = plt.subplots(figsize=(14, 7))

ax2 = ax1.twinx()
ax1.plot(combined_df.index, combined_df['cumulative_spy_returns'], label='INDA')
ax1.plot(combined_df.index, combined_df['cumulative_strategy_returns'], label='Strategy')
ax2.plot(combined_df.index, combined_df['indicator'], label='Indicator', linestyle='--', color='gray')

ax1.set_title('Cumulative Returns of Strategy with Standardized Linear Regression vs. SPY')
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Returns')
ax2.set_ylabel('Indicator')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()