import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime

# Define start and end dates
start_date = '2010-01-01'
end_date = '2023-10-31'

# Fetch SHY ETF data from Yahoo Finance
shy = pdr.get_data_yahoo('SHY', start=start_date, end=end_date)['Adj Close']

# Fetch 1-year real interest rate data from FRED
# The FRED series for 1-year real interest rate is 'DFII1' (Market Yield on U.S. Treasury Securities at 1-Year Constant Maturity, Inflation-Indexed)
real_rate = pdr.DataReader('DFII1', 'fred', start_date, end_date)

# Merge the two data series
df = pd.merge(shy, real_rate, left_index=True, right_index=True, how='inner')
df.columns = ['SHY', '1Y Real Rate']

# Plotting
fig, ax1 = plt.subplots(figsize=(12,6))

ax1.set_xlabel('Date')
ax1.set_ylabel('SHY Price', color='tab:blue')
ax1.plot(df.index, df['SHY'], color='tab:blue', label='SHY ETF')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

ax2.set_ylabel('1Y Real Interest Rate (%)', color='tab:red')  # we already handled the x-label with ax1
ax2.plot(df.index, df['1Y Real Rate'], color='tab:red', label='1Y Real Interest Rate')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()  # Otherwise the right y-label is slightly clipped
plt.title('SHY ETF and 1-Year Real Interest Rate')
plt.show()
