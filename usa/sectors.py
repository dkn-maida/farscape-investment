import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def fetch_data(ticker):
    data = yf.download(ticker, start="2013-01-01")
    data["50_MA"] = data["Close"].rolling(window=50).mean()
    return data 

# Fetch data using yfinance
xlk = fetch_data("XLK")
xlu = fetch_data("XLU")
xlp = fetch_data("XLP")
xly = fetch_data("XLY")

# Create a dataframe to hold daily returns
returns = pd.DataFrame(index=xlk.index)
returns["XLK"] = (xlk["Close"].shift(1) > xlk["50_MA"].shift(1)) * xlk["Close"].pct_change()
returns["XLU"] = (xlu["Close"].shift(1) > xlu["50_MA"].shift(1)) * xlu["Close"].pct_change()
returns["XLP"] = (xlp["Close"].shift(1) > xlp["50_MA"].shift(1)) * xlp["Close"].pct_change()
returns["XLY"] = (xly["Close"].shift(1) > xly["50_MA"].shift(1)) * xly["Close"].pct_change()

# Equal capital allocation for simplicity
returns["Portfolio"] = returns[["XLK", "XLU", "XLP", "XLY"]].mean(axis=1)

# Calculate cumulative returns for the equity curve
returns["Equity_Curve"] = (1 + returns["Portfolio"].fillna(0)).cumprod()

# Buy-and-Hold returns for each ticker
returns["BH_XLK"] = (1 + xlk["Close"].pct_change()).cumprod()
returns["BH_XLU"] = (1 + xlu["Close"].pct_change()).cumprod()
returns["BH_XLP"] = (1 + xlp["Close"].pct_change()).cumprod()
returns["BH_XLY"] = (1 + xly["Close"].pct_change()).cumprod()

# Equal capital allocation for buy-and-hold returns
returns["BH_Portfolio"] = returns[["BH_XLK", "BH_XLU", "BH_XLP", "BH_XLY"]].mean(axis=1)

# Plot the equity curve
plt.figure(figsize=(12, 6))
returns[["Equity_Curve", "BH_Portfolio"]].plot(title="Equity Curve vs. Buy-and-Hold", grid=True, ax=plt.gca())
plt.ylabel("Cumulative Returns")
plt.tight_layout()
plt.legend(["Strategy", "Buy and Hold"])
plt.show()
