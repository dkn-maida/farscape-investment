import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# List of S&P 100 constituents' ticker symbols
# Ensure this list is up-to-date and contains unique tickers
tickers = [
    "MMM", "ABT", "ABBV", "ACN", "GOOG", "MO", "AMZN", "AEP", "AXP", "AIG",
    "AMGN", "AAPL", "T", "BAC", "BK", "BAX", "BIIB", "BA", "BMY",
    "AVGO", "COF", "CAT", "CVX", "CSCO", "C", "KO", "CL", "CMCSA", "COP",
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX",
    "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM",
    "INTC", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS",
    "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM"
] 

# Fetch historical data
historical_data = {}
for ticker in tickers:
    historical_data[ticker] = yf.Ticker(ticker).history(period="2y")['Close']  # Fetching 2 years of data

# Convert to DataFrame
df = pd.DataFrame(historical_data)

# Calculate 126-day moving averages
moving_averages = df.rolling(window=63).mean()

# Backtest logic
# Backtest logic
portfolio_returns = []
for i in range(62, len(df) - 1):  # Exclude the last day to avoid out-of-bounds error
    # Stocks above their 126-day moving average
    selected_stocks = df.columns[(df.iloc[i] > moving_averages.iloc[i])]
    
    if len(selected_stocks) > 0:
        # Equal weight allocation
        daily_return = df.iloc[i + 1][selected_stocks].mean() / df.iloc[i][selected_stocks].mean() - 1
    else:
        daily_return = 0  # No investment if no stocks meet the criterion

    portfolio_returns.append(daily_return)

# Calculate cumulative return of the strategy
cumulative_return = pd.Series(portfolio_returns).add(1).cumprod()


plt.figure(figsize=(10, 6))
plt.plot(cumulative_return, label='Strategy Cumulative Return')
plt.xlabel('Days')
plt.ylabel('Cumulative Return')
plt.title('Backtest Cumulative Returns')
plt.legend()
plt.show()
