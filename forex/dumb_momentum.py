import yfinance as yf
import pandas as pd
from datetime import datetime

# Currency pairs for Yahoo Finance
currency_pairs = ["EURUSD=X", "JPYUSD=X", "GBPUSD=X", "CHFUSD=X", "AUDUSD=X", "CADUSD=X"]

# Define start and end dates for data retrieval
start_date = datetime(2010, 1, 1)
dataframes = {}

# Fetch data using yfinance
for pair in currency_pairs:
    try:
        data = yf.download(pair, start=start_date)
        data = data[['Close']]  # Only the 'Close' column
        data.columns = [pair]  # Rename column to the currency pair name
        dataframes[pair] = data
    except Exception as e:
        print(f"Error fetching data for {pair}: {e}")

# Merging dataframes
merged_df = pd.concat(dataframes.values(), axis=1)

def calculate_6_month_momentum(data, pair_name):
    return data[pair_name].shift(1) / data[pair_name].shift(365) - 1  # Assuming about 30 trading days per month

# Calculating momentum
for pair in currency_pairs:
    merged_df[f'{pair}_momentum'] = calculate_6_month_momentum(merged_df, pair)

capital = 10000  # Initial capital
portfolio = []

# Backtesting monthly rebalancing
for date in merged_df.index[180:]:  # start from 180 due to momentum calc
    monthly_momentum = {pair: merged_df.at[date, f"{pair}_momentum"] for pair in currency_pairs}
    
    strongest_currency = max(monthly_momentum, key=monthly_momentum.get).replace("=X", "")
    weakest_currency = min(monthly_momentum, key=monthly_momentum.get).replace("=X", "")
    
    print(f"date -> {date} weakest-> {weakest_currency} strongest-> {strongest_currency}")