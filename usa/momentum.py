import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NFCI Data Preparation (assuming nfci.csv is correctly formatted)
nfci_data = pd.read_csv('nfci.csv')
nfci_data['date'] = pd.to_datetime(nfci_data['DATE'])
nfci_data.set_index('date', inplace=True)
nfci_data['nfci_sma_14'] = nfci_data['NFCI'].rolling(window=2).mean()
nfci_data['signal'] = np.where(nfci_data['NFCI'] < nfci_data['nfci_sma_14'], 1, 0)

# S&P 100 Symbols and Benchmark
symbols = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
    "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM"
]  # Your list of S&P 100 symbols
benchmark_symbol = "^OEX"
start_date = "2013-06-01"
end_date = "2023-06-01"

# Fetch Historical Stock Data
historical_data = pd.DataFrame()
for symbol in symbols:
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data['Symbol'] = symbol
    historical_data = pd.concat([historical_data, stock_data], ignore_index=False)
historical_data.reset_index(inplace=True)

# Calculate 6-Month Returns and Rank Stocks
historical_data['6M_Return'] = historical_data.groupby('Symbol')['Adj Close'].pct_change(periods=126)
historical_data['Rank'] = historical_data.groupby('Date')['6M_Return'].rank(ascending=False)

# Resample NFCI Data
nfci_weekly = nfci_data.resample('W-FRI').last()

# Merge Historical Data with NFCI Data
historical_data['Date'] = pd.to_datetime(historical_data['Date'])
merged_data = pd.merge(historical_data, nfci_weekly, left_on='Date', right_index=True, how='left')

# Select Top Stocks Considering NFCI Signal
merged_data['Top_Ranked'] = np.where(merged_data['Rank'] <= 3, 1, 0)
merged_data['Top_Ranked'] = merged_data['Top_Ranked'] * merged_data['signal']

# Fetch Benchmark Data
benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date)
benchmark_data['Benchmark_Return'] = benchmark_data['Adj Close'].pct_change()

# Initialize Portfolio Variables
initial_balance = 1000000
cash_balance = initial_balance
portfolio = {}
portfolio_values = []
last_rebalance_date = None

# Helper Functions for Rebalancing and Value Calculation
def rebalance_portfolio(date, group, portfolio, cash_balance):
    # Sell existing positions
    cash_balance += sum([group[group['Symbol'] == symbol]['Adj Close'].values[0] * shares 
                         for symbol, shares in portfolio.items()])
    portfolio = {}
    # Buy new positions if NFCI signal is positive
    if group['signal'].iloc[0] == 1:
        top_stocks = group[group['Top_Ranked'] == 1]['Symbol'].unique()
        if len(top_stocks) > 0:
            position_size = cash_balance / len(top_stocks)
            for symbol in top_stocks:
                stock_price = group[group['Symbol'] == symbol]['Adj Close'].values[0]
                shares_to_buy = int(position_size / stock_price)
                portfolio[symbol] = shares_to_buy
                cash_balance -= shares_to_buy * stock_price
    return cash_balance, portfolio

def calculate_portfolio_value(date, group, portfolio, cash_balance):
    total_value = cash_balance
    total_value += sum([group[group['Symbol'] == symbol]['Adj Close'].values[0] * shares 
                        for symbol, shares in portfolio.items()])
    return total_value

# Backtest the Strategy
for date, group in merged_data.groupby('Date'):
    if last_rebalance_date is None or date.month != last_rebalance_date.month:
        cash_balance, portfolio = rebalance_portfolio(date, group, portfolio, cash_balance)
        last_rebalance_date = date
    total_value = calculate_portfolio_value(date, group, portfolio, cash_balance)
    portfolio_values.append(total_value)

# Plot the Results
portfolio_performance = pd.DataFrame({'Portfolio_Value': portfolio_values}, index=benchmark_data.index)
portfolio_performance['Portfolio_Return'] = portfolio_performance['Portfolio_Value'].pct_change()
portfolio_performance['Cumulative_Return'] = (portfolio_performance['Portfolio_Return'] + 1).cumprod() - 1
benchmark_data['Cumulative_Return'] = (benchmark_data['Benchmark_Return'] + 1).cumprod() - 1

plt.figure(figsize=(12, 6))
plt.plot(portfolio_performance.index, portfolio_performance['Cumulative_Return'], label='Portfolio')
plt.plot(benchmark_data.index, benchmark_data['Cumulative_Return'], label='Benchmark (S&P 100 Index)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Portfolio vs. Benchmark Performance')
plt.legend()
plt.grid(True)
plt.show()

# Final Portfolio Value
final_portfolio_value = portfolio_performance['Portfolio_Value'].iloc[-1]
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
