import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Read the NFCI data
nfci_data = pd.read_csv('nfci.csv')
nfci_data['date'] = pd.to_datetime(nfci_data['DATE'])
nfci_data.set_index('date', inplace=True)
nfci_data['nfci_sma_14'] = nfci_data['NFCI'].rolling(window=2).mean()

tickers = ['AAPL', 'TSLA', 'GOOG', 'NVDA', 'V', 'ADBE', 'META', 'AMZN', 'LLY', 'JPM', 'AVGO']
portfolio_value = pd.DataFrame()

for ticker in tickers:
    # Download data for the ticker
    stock_data = yf.download(ticker, start='2018-01-01', end=nfci_data.index.max())
    stock_data = stock_data[['Close']].resample("W-FRI").last()

    # Calculate the 3-week moving average and shift it by one week
    stock_data['3wk_MA'] = stock_data['Close'].rolling(window=3).mean().shift(1)

    # Merge with NFCI data
    stock_data = stock_data.join(nfci_data[['NFCI', 'nfci_sma_14']].shift(1), how='inner')

    # Signals: 1 for long, -1 for short, 0 for no position
    stock_data['signal'] = np.where((stock_data['NFCI'] < stock_data['nfci_sma_14']) & 
                                    (stock_data['Close'] > stock_data['3wk_MA']), 1, 
                                    np.where((stock_data['NFCI'] == 0) & 
                                             (stock_data['Close'] < stock_data['3wk_MA']), -1, 0))

    # Weekly returns (positive for long, negative for short)
    stock_data['weekly_return'] = stock_data['Close'].pct_change() * stock_data['signal']

    # Add to portfolio
    portfolio_value[ticker] = stock_data['weekly_return']

# Calculate portfolio returns
portfolio_value['strategy_return'] = portfolio_value.mean(axis=1)

# Cumulative returns
portfolio_value['cumulative_return'] = (1 + portfolio_value['strategy_return']).cumprod()

# Plotting the results
plt.figure(figsize=(12, 6))
portfolio_value['cumulative_return'].plot(label='Long/Short Strategy', color='b')
plt.title('Cumulative Returns of the Long/Short Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()
