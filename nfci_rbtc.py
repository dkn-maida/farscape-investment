import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


def fetch_data():
    nfci_data = pd.read_csv('nfci.csv')
    nfci_data['date'] = pd.to_datetime(nfci_data['DATE'])
    nfci_data.set_index('date', inplace=True)

    spy_data = yf.download('^GSPC', start=nfci_data.index.min(), end=nfci_data.index.max())
    spy_data = spy_data[['Close']].resample("W-FRI").last()

    return spy_data.join(nfci_data, how='inner')


def backtest_strategy(data, sma_window):
    data = data.copy()
    data['nfci_sma'] = data['NFCI'].rolling(window=int(sma_window)).mean()
    data['signal'] = np.where(data['NFCI'] < data['nfci_sma'], 1, 0)
    data['strategy_returns'] = data['Close'].pct_change() * data['signal'].shift(1)
    return data


def walk_forward_optimization(data, optimization_window, walk_forward_window):
    n = (len(data) - optimization_window) // walk_forward_window
    wf_performance = []

    for i in range(n):
        optimization_data = data.iloc[i * walk_forward_window: i * walk_forward_window + optimization_window]
        walk_forward_data = data.iloc[i * walk_forward_window + optimization_window : i * walk_forward_window + optimization_window + walk_forward_window]

        # For simplicity, let's assume you found that sma_window = 14 is the best for this window
        sma_window = 2

        wf_data = backtest_strategy(walk_forward_data, sma_window)
        cagr = (1 + wf_data['strategy_returns']).cumprod().iloc[-1] - 1
        wf_performance.append(cagr)

    return wf_performance


def monte_carlo_simulation(returns, simulations=1000, trading_days=252):
    simulated_portfolios = np.zeros(simulations)

    for s in range(simulations):
        log_returns = np.random.normal(np.mean(returns), np.std(returns), trading_days)
        simulated_portfolios[s] = np.exp(np.sum(log_returns))

    return simulated_portfolios


def stress_testing(data, shift=-0.03):
    data = data.copy()
    data['strategy_returns'] -= shift
    return data


# Fetch the data
data = fetch_data()

# Split the data into in-sample and out-of-sample datasets
split_date = '2007-01-01'
in_sample_data = data[data.index < split_date]
out_of_sample_data = data[data.index >= split_date]

# Optimal SMA window
sma_window = 2

# Run in-sample backtest
in_sample_results = backtest_strategy(in_sample_data, sma_window)

# Run out-of-sample backtest
out_of_sample_results = backtest_strategy(out_of_sample_data, sma_window)

# Walk-Forward Optimization
optimization_window = 30
walk_forward_window = 10
wf_performance = walk_forward_optimization(in_sample_data, optimization_window, walk_forward_window)

# Monte Carlo Simulation
monte_carlo_results = monte_carlo_simulation(out_of_sample_results['strategy_returns'])

# Stress Testing
stress_test_data = stress_testing(out_of_sample_results)

# Plotting Walk-Forward Optimization Results
plt.figure()
plt.plot(wf_performance, label='Walk Forward Performance')
plt.title('Walk Forward Optimization')
plt.xlabel('Iteration')
plt.ylabel('CAGR')
plt.legend()
plt.show()

# Plotting Monte Carlo Simulation Results
plt.figure()
plt.hist(monte_carlo_results, bins=50, alpha=0.6, color='blue', label='Monte Carlo Distribution')
plt.title('Monte Carlo Simulation')
plt.legend()
plt.show()

# Plotting Strategy with Stress Test using Logarithmic Scale
plt.figure()
stress_test_data['cum_strategy_returns'] = (1 + stress_test_data['strategy_returns']).cumprod()
plt.semilogy(stress_test_data.index, stress_test_data['cum_strategy_returns'], label='Stress Tested Strategy', color='orange')
plt.title('Stress Test - Strategy Performance (Logarithmic Scale)')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns (Log Scale)')
plt.legend()
plt.show()