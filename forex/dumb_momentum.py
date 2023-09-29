import yfinance as yf
import pandas as pd
from datetime import datetime

# Currency pairs for Yahoo Finance
currency_pairs = ["EURUSD=X", "JPYUSD=X", "GBPUSD=X", "CHFUSD=X", "AUDUSD=X", "CADUSD=X", "NZDUSD=X", "SGDUSD=X", "SEKUSD=X", "NOKUSD=X", "BRLUSD=X", "INRUSD=X"]

# Define start date for data retrieval
start_date = datetime(2022, 1, 1)
end_date = datetime.today()

dataframes = {}

# Fetch daily data using yfinance
for pair in currency_pairs:
    try:
        data = yf.download(pair, start=start_date, end=end_date, interval='1d')
        data = data[['Close']]  # Only the 'Close' column
        data.columns = [pair]  # Rename column to the currency pair name
        dataframes[pair] = data
    except Exception as e:
        print(f"Error fetching data for {pair}: {e}")

# Merging dataframes
merged_df = pd.concat(dataframes.values(), axis=1)

def calculate_6_month_momentum(data, pair_name):
    return data[pair_name].shift(1) / data[pair_name].shift(20*6) - 1  # Assuming about 20 trading days a month
def calculate_3_month_momentum(data, pair_name):
    return data[pair_name].shift(1) / data[pair_name].shift(20*3) - 1  # Assuming about 20 trading days a month
def calculate_1_month_momentum(data, pair_name):
    return data[pair_name].shift(1) / data[pair_name].shift(20) - 1  # Assuming about 20 trading days a month

# Calculating momentum
for pair in currency_pairs:
    merged_df[f'{pair}_6_month_momentum'] = calculate_6_month_momentum(merged_df, pair)
    merged_df[f'{pair}_3_month_momentum'] = calculate_3_month_momentum(merged_df, pair)
    merged_df[f'{pair}_1_month_momentum'] = calculate_1_month_momentum(merged_df, pair)
    merged_df[f'{pair}_momentum'] = merged_df[f'{pair}_6_month_momentum'] + merged_df[f'{pair}_3_month_momentum'] + merged_df[f'{pair}_1_month_momentum']

# File to log the results
with open("currency_strength_log.txt", "w") as file:
    for date in merged_df.index[20*6:]:  # Start after 6 months due to momentum calculation
        daily_momentum = {pair: merged_df.at[date, f"{pair}_momentum"] for pair in currency_pairs}
        sorted_momentum = sorted(daily_momentum.items(), key=lambda x: x[1], reverse=True)
        
        top_2_strongest = [x[0].replace("=X", "") for x in sorted_momentum[:2]]
        top_2_weakest = [x[0].replace("=X", "") for x in sorted_momentum[-2:]]
        
        log_str = f"date -> {date} weakest-> {top_2_weakest} strongest-> {top_2_strongest}\n"
        file.write(log_str)
