import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Read the NFCI data
nfci_data = pd.read_csv('nfci.csv')
nfci_data['date'] = pd.to_datetime(nfci_data['DATE'])
nfci_data.set_index('date', inplace=True)
nfci_data['nfci_sma_14'] = nfci_data['NFCI'].rolling(window=14).mean()

tickers = ['SPY']

# Download VIXM data
vixm_data = yf.download('SHV', start=nfci_data.index.min(), end=nfci_data.index.max())
vixm_data = vixm_data[['Close']].resample("W-FRI").last()

# Annualizing factor assuming trading weeks
annualizing_factor = 52

for ticker in tickers:
    stock_data = yf.download(ticker, start=nfci_data.index.min(), end=nfci_data.index.max())
    stock_data = stock_data[['Close']].resample("W-FRI").last()

    # Merge the stock, VIXM, and NFCI data
    data = stock_data.join(nfci_data, how='inner')
    data = data.join(vixm_data, rsuffix='_SH', how='left')

    data['signal'] = np.where(data['NFCI'] < data['nfci_sma_14'], 1, 0)
    data['strategy_returns'] = np.nan

    for i in range(1, len(data)):
        data.loc[data.index[i], 'strategy_returns'] = (
            data['Close'].pct_change().iloc[i]
            if data['signal'].iloc[i - 1] == 1
            else data['Close_SH'].pct_change().iloc[i]
        )
    data['stock_returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    plt.figure(figsize=(10, 6))
    (1 + data['strategy_returns']).cumprod().plot(label='Strategy', color='b')
    (1 + data['stock_returns']).cumprod().plot(label=f'{ticker} Buy and Hold', color='r')
    plt.title(f'Performance Comparison for {ticker}')
    plt.legend()
    plt.show()

    # CAGR
    years = (data.index[-1] - data.index[0]).days / 365.25
    cagr_strategy = ((1 + data['strategy_returns']).cumprod().iloc[-1]) ** (1 / years) - 1
    cagr_stock = ((1 + data['stock_returns']).cumprod().iloc[-1]) ** (1 / years) - 1

    # Max Drawdown
    cumulative_returns_strategy = (1 + data['strategy_returns']).cumprod()
    rolling_max_strategy = cumulative_returns_strategy.expanding().max()
    daily_drawdown_strategy = cumulative_returns_strategy / rolling_max_strategy - 1
    max_drawdown_strategy = daily_drawdown_strategy.min()

    cumulative_returns_stock = (1 + data['stock_returns']).cumprod()
    rolling_max_stock = cumulative_returns_stock.expanding().max()
    daily_drawdown_stock = cumulative_returns_stock / rolling_max_stock - 1
    max_drawdown_stock = daily_drawdown_stock.min()

    # Sharpe Ratio (assuming risk-free rate to be 0)
    sharpe_ratio_strategy = np.mean(data['strategy_returns']) / np.std(data['strategy_returns']) * np.sqrt(annualizing_factor)
    sharpe_ratio_stock = np.mean(data['stock_returns']) / np.std(data['stock_returns']) * np.sqrt(annualizing_factor)

    print(f"For {ticker}:")
    print(f"Strategy: CAGR: {cagr_strategy:.2%}, Max Drawdown: {max_drawdown_strategy:.2%}, Sharpe Ratio: {sharpe_ratio_strategy:.2f}")
    print(f"{ticker} Buy and Hold: CAGR: {cagr_stock:.2%}, Max Drawdown: {max_drawdown_stock:.2%}, Sharpe Ratio: {sharpe_ratio_stock:.2f}")
    print("-" * 50)