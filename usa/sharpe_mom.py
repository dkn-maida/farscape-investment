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
    "GE", "GILD", "GM", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
    "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM"
]

benchmark_symbol = "^OEX"
start_date = "2014-01-01"
initial_balance = 20000
rolling_window = 126
risk_free_rate = 0.04  # Assuming 1% annual risk-free rate

# Fetch Historical Stock Data
historical_data = yf.download(symbols, start=start_date)['Adj Close']

# Calculate Daily Returns
daily_returns = historical_data.pct_change()

# Calculate Rolling Mean Returns, Standard Deviation, and Sharpe Ratio
rolling_mean_returns = daily_returns.rolling(window=rolling_window).mean()
rolling_std_deviation = daily_returns.rolling(window=rolling_window).std()

# Sharpe Ratio Calculation
annualized_mean_returns = rolling_mean_returns * 252  # Annualize the mean returns
annualized_std_deviation = rolling_std_deviation * np.sqrt(252)  # Annualize the standard deviation
sharpe_ratios = (annualized_mean_returns - risk_free_rate) / annualized_std_deviation

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
            purchase_price = purchase_prices.get(symbol, 0)
            sale_price = max(price, price)
            new_cash_balance += sale_price * shares

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
    if date >= pd.to_datetime(start_date) + pd.DateOffset(days=21):
                # Calculate Rolling Mean Returns, Standard Deviation, and Sharpe Ratio
        rolling_mean_returns = daily_returns.rolling(window=rolling_window).mean()
        rolling_std_deviation = daily_returns.rolling(window=rolling_window).std()

        # Sharpe Ratio Calculation
        annualized_mean_returns = rolling_mean_returns * 252  # Annualize the mean returns
        annualized_std_deviation = rolling_std_deviation * np.sqrt(252)  # Annualize the standard deviation
        sharpe_ratios = (annualized_mean_returns - risk_free_rate) / annualized_std_deviation
        top_stocks = row[sharpe_ratios.loc[date].nlargest(5).index].dropna().index.tolist()
        prices = historical_data.loc[date]
        cash_balance, portfolio = rebalance_portfolio(date, top_stocks, portfolio, cash_balance, prices)
        portfolio_value = calculate_portfolio_value(portfolio, cash_balance, prices)
        portfolio_values.append({'Date': date, 'Portfolio_Value': portfolio_value})

portfolio_performance = pd.DataFrame(portfolio_values).set_index('Date')
portfolio_performance['Portfolio_Return'] = portfolio_performance['Portfolio_Value'].pct_change()
portfolio_performance['Cumulative_Return'] = (1 + portfolio_performance['Portfolio_Return']).cumprod() - 1

# Calculate Monthly Returns
monthly_performance = portfolio_performance['Portfolio_Value'].resample('M').last()
monthly_returns = monthly_performance.pct_change().dropna() * 100  # Convert to percentage

# Fetch Benchmark Data
benchmark_data = yf.download(benchmark_symbol, start=start_date)['Adj Close'].pct_change()
benchmark_cumulative_return = (1 + benchmark_data).cumprod() - 1

# Calculate CAGR
ending_value = portfolio_performance['Portfolio_Value'].iloc[-1]
beginning_value = initial_balance
num_years = (portfolio_performance.index[-1] - portfolio_performance.index[0]).days / 365.25
cagr = (ending_value / beginning_value) ** (1 / num_years) - 1

# Calculate Maximum Drawdown
rolling_max = portfolio_performance['Portfolio_Value'].cummax()
drawdown = (portfolio_performance['Portfolio_Value'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

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

# Print Final Portfolio Value, CAGR, Max Drawdown, and Monthly Performance
print(f"Final Portfolio Value: ${portfolio_performance['Portfolio_Value'].iloc[-1]:.2f}")
print(f"CAGR: {cagr:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")

print("\nMonthly Returns (%):")
print(monthly_returns.to_string(index=True, float_format="%.2f"))

# Robustness Check: Run the strategy for different rolling windows
for rw in [21, 63, 126, 252]:  # Quarterly, semi-annual, and annual rolling windows
    rolling_mean_returns = daily_returns.rolling(window=rw).mean()
    rolling_std_deviation = daily_returns.rolling(window=rw).std()
    annualized_mean_returns = rolling_mean_returns * 252
    annualized_std_deviation = rolling_std_deviation * np.sqrt(252)
    sharpe_ratios = (annualized_mean_returns - risk_free_rate) / annualized_std_deviation

    portfolio = {}
    cash_balance = initial_balance
    portfolio_values = []
    purchase_prices = {}

    for date, row in daily_returns.iterrows():
        if date >= pd.to_datetime(start_date) + pd.DateOffset(days=21):
            top_stocks = row[sharpe_ratios.loc[date].nlargest(5).index].dropna().index.tolist()
            prices = historical_data.loc[date]
            cash_balance, portfolio = rebalance_portfolio(date, top_stocks, portfolio, cash_balance, prices)
            portfolio_value = calculate_portfolio_value(portfolio, cash_balance, prices)
            portfolio_values.append({'Date': date, 'Portfolio_Value': portfolio_value})

    portfolio_performance = pd.DataFrame(portfolio_values).set_index('Date')
    portfolio_performance['Portfolio_Return'] = portfolio_performance['Portfolio_Value'].pct_change()
    portfolio_performance['Cumulative_Return'] = (1 + portfolio_performance['Portfolio_Return']).cumprod() - 1

    # Calculate CAGR
    ending_value = portfolio_performance['Portfolio_Value'].iloc[-1]
    beginning_value = initial_balance
    num_years = (portfolio_performance.index[-1] - portfolio_performance.index[0]).days / 365.25
    cagr = (ending_value / beginning_value) ** (1 / num_years) - 1

    # Calculate Maximum Drawdown
    rolling_max = portfolio_performance['Portfolio_Value'].cummax()
    drawdown = (portfolio_performance['Portfolio_Value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Print Robustness Check Results
    print(f"\nRobustness Check for Rolling Window {rw} Days:")
    print(f"Final Portfolio Value: ${portfolio_performance['Portfolio_Value'].iloc[-1]:.2f}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
