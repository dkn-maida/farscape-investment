import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from fredapi import Fred
from pandas.tseries.offsets import DateOffset
from sklearn.linear_model import LinearRegression


# Define a function to convert date to ordinal
def convert_date_to_ordinal(date):
    return date.toordinal()

# Fetch SPY data
spy = yf.download('INDA')
spy_close = spy['Close']

# Replace 'my_api_key' with your actual FRED API key
fred = Fred(api_key='2f15c96b46530fde1b1992a64c64650e')

# Fetch WTI Crude Oil Price data
oil_price = fred.get_series('DCOILWTICO')

# Remove or fill in missing values
spy_close = spy_close.dropna()
oil_price = oil_price.dropna()

# Replace zero values (if any exist)
spy_close = spy_close.replace(0, np.nan)
oil_price = oil_price.replace(0, np.nan)

# Calculate the logarithms
spy_log = np.log(spy_close)
oil_log = np.log(oil_price)

# Remove any new NaN values that might have been created
spy_log = spy_log.dropna()
oil_log = oil_log.dropna()

spy_oil_log_ratio = (spy_log / oil_log)


def calculate_slope(X, y):
    """Fit a linear regression model and return the slope"""
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0]


# Initialize the indicator series
indicator = pd.Series(index=spy_oil_log_ratio.index)

# Compute the 1-year and 7-year slopes for each day
for date in spy_oil_log_ratio.index:
    # Compute the 1-year slope
    one_year_ago = date - DateOffset(years=1)
    last_year_log_ratio = spy_oil_log_ratio[
        (spy_oil_log_ratio.index >= one_year_ago) & (spy_oil_log_ratio.index < date)]
    if not last_year_log_ratio.empty:
        X1 = np.array(last_year_log_ratio.index.map(convert_date_to_ordinal)).reshape(-1, 1)
        y1 = last_year_log_ratio.values
        if pd.Series(y1).isna().all():
            y1 = pd.Series(y1).fillna(0).values
        else:
            y1 = pd.Series(y1).fillna(pd.Series(y1).mean()).values
        one_year_slope = calculate_slope(X1, y1)

        # Compute the 7-year slope
        seven_years_ago = date - DateOffset(years=7)
        last_seven_years_log_ratio = spy_oil_log_ratio[
            (spy_oil_log_ratio.index >= seven_years_ago) & (spy_oil_log_ratio.index < date)]
        if not last_seven_years_log_ratio.empty:
            X7 = np.array(last_seven_years_log_ratio.index.map(convert_date_to_ordinal)).reshape(-1, 1)
            y7 = last_seven_years_log_ratio.values
            if pd.Series(y7).isna().all():
                y7 = pd.Series(y7).fillna(0).values
            else:
                y7 = pd.Series(y7).fillna(pd.Series(y7).mean()).values
            seven_years_slope = calculate_slope(X7, y7)

            # Set the indicator
            indicator[date] = int(one_year_slope > seven_years_slope)

# Combine the SPY closing prices and the indicator into one DataFrame
spy_close = spy_close.reindex(spy_oil_log_ratio.index)
backtest_data = pd.DataFrame({'spy_close': spy_close, 'indicator': indicator})

# Forward fill missing values in the SPY closing prices
backtest_data['spy_close'].fillna(method='ffill', inplace=True)

# Calculate the daily returns when indicator is 1
backtest_data['daily_returns_1'] = backtest_data['spy_close'].pct_change() * (backtest_data['indicator'] == 1)

# Calculate the daily returns when indicator is 0
backtest_data['daily_returns_0'] = backtest_data['spy_close'].pct_change() * (backtest_data['indicator'] == 0)

# Calculate the cumulative returns when indicator is 1
backtest_data['cumulative_returns_1'] = (1 + backtest_data['daily_returns_1']).cumprod()

# Calculate the cumulative returns when indicator is 0
backtest_data['cumulative_returns_0'] = (1 + backtest_data['daily_returns_0']).cumprod()

# Calculate the cumulative returns for buy and hold strategy for comparison
backtest_data['buy_and_hold_returns'] = (1 + backtest_data['spy_close'].pct_change()).cumprod()

# Plot the backtesting results
plt.figure(figsize=(14, 7))
plt.plot(backtest_data['cumulative_returns_1'], label='Strategy (Indicator=1)')
plt.plot(backtest_data['cumulative_returns_0'], label='Strategy (Indicator=0)')
plt.plot(backtest_data['buy_and_hold_returns'], label='Buy and Hold')
plt.title('Backtesting Results')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns (Log Scale)')
plt.yscale('log')  # Setting y-axis to log scale
plt.legend()
plt.show()