import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fetch hourly data for Apple for the last 730 days
end_date = pd.to_datetime('today')
start_date = end_date - pd.Timedelta(days=730)
ticker = "AI"
data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)

# Compute the 200 periods moving average
data['200_MA'] = data['Close'].rolling(window=200).mean()

# Compute the 5 period RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)
avg_gain = gain.rolling(window=5).mean()
avg_loss = loss.rolling(window=5).mean()
rs = avg_gain / avg_loss
data['5_RSI'] = 100 - (100 / (1 + rs))

# Compute the 120 and 240 period rate of change
data['120_ROC'] = data['Close'].pct_change(periods=120)
data['240_ROC'] = data['Close'].pct_change(periods=240)

# Generate the signal
data['Signal'] = np.where(
    (data['Close'] > data['200_MA']) &
    (data['5_RSI'] < 25) &
    (data['120_ROC'] > 0) &
    (data['240_ROC'] > 0),
    1, 0)

# Implement the strategy
buy_price = 0
data['Strategy_Returns'] = np.nan
for i in range(1, len(data)):
    if data['Signal'].iloc[i-1] == 1:
        buy_price = data['Open'].iloc[i]
        if i+5 < len(data):
            sell_price = data['Close'].iloc[i+5]
            data.loc[data.index[i+5], 'Strategy_Returns'] = (sell_price - buy_price) / buy_price

# Calculate cumulative returns
data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns'].fillna(0)).cumprod()

# Buy and Hold returns
data['Buy_and_Hold'] = (1 + data['Close'].pct_change()).cumprod()

# Compute CAGR
years = (data.index[-1] - data.index[0]).days / 365.25
CAGR = (data['Cumulative_Strategy_Returns'].iloc[-1])**(1/years) - 1
Buy_and_Hold_CAGR = (data['Buy_and_Hold'].iloc[-1])**(1/years) - 1

# Compute Max Drawdown
data['Cumulative_Roll_Max'] = data['Cumulative_Strategy_Returns'].cummax()
data['Drawdown'] = data['Cumulative_Strategy_Returns'] / data['Cumulative_Roll_Max'] - 1
Max_Drawdown = data['Drawdown'].min()

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))
data['Cumulative_Strategy_Returns'].plot(ax=ax, label="Strategy")
data['Buy_and_Hold'].plot(ax=ax, label="Buy and Hold", linestyle="--")
ax.set_title('Cumulative Returns')
ax.set_ylabel('Cumulative Returns')
ax.legend()
plt.tight_layout()
plt.show()

print(f"Strategy CAGR: {CAGR * 100:.2f}%")
print(f"Buy and Hold CAGR: {Buy_and_Hold_CAGR * 100:.2f}%")
print(f"Max Drawdown: {Max_Drawdown * 100:.2f}%")
