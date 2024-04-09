import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
symbols = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
    "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM", "TLT",
    "GLD"
]
benchmark_symbol = "^OEX"
start_date = "2019-01-01"
initial_balance = 25000
rolling_window = 252 

# Fetch Historical Stock Data
historical_data = yf.download(symbols + [benchmark_symbol], start=start_date)['Adj Close']
benchmark_data = historical_data[benchmark_symbol]
benchmark_returns = benchmark_data.pct_change()
benchmark_cumulative_return = (1 + benchmark_returns).cumprod() - 1
stock_data = historical_data.drop(columns=[benchmark_symbol])



# Calculate Daily Returns
daily_returns = stock_data.pct_change()

# Calculate 6-month momentum for each stock
momentum_scores = stock_data.pct_change(periods=126)  # Approximate 6 months with 126 trading days

# Initialize Portfolio Variables
portfolio = {}
cash_balance = initial_balance
portfolio_values = []
purchase_prices = {}

# Rebalance Portfolio Function
def rebalance_portfolio(date, top_stocks, portfolio, cash_balance, prices):
    global purchase_prices
    new_portfolio = {}
    new_cash_balance = cash_balance

    for symbol, shares in portfolio.items():
        if symbol in prices:
            price = prices[symbol]
            new_cash_balance += price * shares

    num_stocks = len(top_stocks)
    if num_stocks > 0:
        position_size = new_cash_balance / num_stocks
        for symbol in top_stocks:
            if symbol in prices:
                stock_price = prices[symbol]
                shares_to_buy = position_size // stock_price
                new_portfolio[symbol] = shares_to_buy
                new_cash_balance -= shares_to_buy * stock_price
                purchase_prices[symbol] = stock_price

    portfolio.clear()
    return new_cash_balance, new_portfolio

# Calculate Portfolio Value Function
def calculate_portfolio_value(portfolio, cash_balance, prices):
    total_value = cash_balance
    for symbol, shares in portfolio.items():
        if symbol in prices:
            total_value += prices[symbol] * shares
    return total_value

# Backtest the Strategy
for date, row in daily_returns.iterrows():
    prices = historical_data.loc[date]
    if date >= pd.to_datetime(start_date) + pd.DateOffset(weeks=1):
        top_stocks = momentum_scores.loc[date].nlargest(1).dropna().index.tolist()
        cash_balance, portfolio = rebalance_portfolio(date, top_stocks, portfolio, cash_balance, prices)
        portfolio_value = calculate_portfolio_value(portfolio, cash_balance, prices)
        portfolio_values.append({'Date': date, 'Portfolio_Value': portfolio_value})

portfolio_performance = pd.DataFrame(portfolio_values).set_index('Date')
portfolio_performance['Portfolio_Return'] = portfolio_performance['Portfolio_Value'].pct_change()
portfolio_performance['Cumulative_Return'] = (1 + portfolio_performance['Portfolio_Return']).cumprod() - 1

# Plot the Results
plt.figure(figsize=(12, 6))
plt.plot(portfolio_performance.index, portfolio_performance['Cumulative_Return'], label='Portfolio')
plt.plot(benchmark_cumulative_return.index, benchmark_cumulative_return, label='Benchmark (S&P 100 Index)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Portfolio vs. Benchmark Performance')
plt.legend()
plt.grid(True)
plt.show()

# Print Final Portfolio Value
print(f"Final Portfolio Value: ${portfolio_performance['Portfolio_Value'].iloc[-1]:.2f}")

# Fetch the latest date from the daily_returns dataset to display top stocks based on momentum
latest_date = daily_returns.index[-1]
latest_top_stocks = momentum_scores.loc[latest_date].nlargest(1).index.dropna().tolist()
print("Current Top-List Stocks based on the latest 6-month momentum calculation:")
for stock in latest_top_stocks:
    print(stock)
