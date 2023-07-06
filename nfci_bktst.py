import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import ffn

def fetch_data():
    nfci_data = pd.read_csv('nfci.csv')
    nfci_data['date'] = pd.to_datetime(nfci_data['DATE'])
    nfci_data.set_index('date', inplace=True)

    spy_data = yf.download('URTH', start=nfci_data.index.min(), end=nfci_data.index.max())
    spy_data = spy_data[['Close']].resample("W-FRI").last()

    return spy_data.join(nfci_data, how='inner')

def backtest_strategy(data, sma_window, transaction_cost):
    data = data.copy()  # Adding this line to explicitly work on a copy
    data['nfci_sma'] = data['^GDAXI'].rolling(window=int(sma_window)).mean()
    data['signal'] = np.where(data['NFCI'] < data['nfci_sma'], 1, 0)
    data['strategy_returns'] = data['Close'].pct_change() * data['signal'].shift(1)
    data['strategy_returns'] = data['strategy_returns'] - transaction_cost * np.abs(data['signal'].diff())
    return data


def performance_metrics(data):
    annualizing_factor = 52
    years = (data.index[-1] - data.index[0]).days / 365.25
    
    # CAGR
    cagr_strategy = (1 + data['strategy_returns']).cumprod().iloc[-1] ** (1 / years) - 1
    
    # Max Drawdown
    cumulative_returns_strategy = (1 + data['strategy_returns']).cumprod()
    rolling_max_strategy = cumulative_returns_strategy.expanding().max()
    daily_drawdown_strategy = cumulative_returns_strategy / rolling_max_strategy - 1
    max_drawdown_strategy = daily_drawdown_strategy.min()
    
    # Sharpe Ratio
    sharpe_ratio_strategy = np.mean(data['strategy_returns']) / np.std(data['strategy_returns']) * np.sqrt(annualizing_factor)
    
    # Sortino Ratio
    target_return = 0
    excess_return = data['strategy_returns'] - target_return
    downside_risk = np.std(excess_return[excess_return < 0])
    sortino_ratio = np.mean(excess_return) / downside_risk

    return {
        'cagr_strategy': cagr_strategy,
        'max_drawdown_strategy': max_drawdown_strategy,
        'sharpe_ratio_strategy': sharpe_ratio_strategy,
        'sortino_ratio_strategy': sortino_ratio
    }

def objective(params):
    sma_window = params['sma_window']
    transaction_cost = params['transaction_cost']
    
    tscv = TimeSeriesSplit(n_splits=3)
    total_sharpe_ratio = 0
    
    for train_index, test_index in tscv.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        
        test_data = backtest_strategy(test_data, sma_window, transaction_cost)
        
        metrics = performance_metrics(test_data)
        total_sharpe_ratio += metrics['sharpe_ratio_strategy']
    
    average_sharpe_ratio = total_sharpe_ratio / 3
    return {'loss': -average_sharpe_ratio, 'status': STATUS_OK}

# Fetch data
data = fetch_data()

# Bayesian hyperparameter optimization
space = {
    'sma_window': hp.quniform('sma_window', 5, 30, 1),
    'transaction_cost': hp.uniform('transaction_cost', 0.0001, 0.005)
}

trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

# Extract the optimal parameters
optimal_sma_window = int(best['sma_window'])
optimal_transaction_cost = best['transaction_cost']

# Backtest the strategy using the optimal parameters on the full dataset
final_data = backtest_strategy(data, optimal_sma_window, optimal_transaction_cost)

# Calculate and display performance metrics
final_metrics = performance_metrics(final_data)
print(final_metrics)

# Plotting
(final_data['strategy_returns'] + 1).cumprod().plot(label='Strategy', color='b')
(final_data['Close'].pct_change() + 1).cumprod().plot(label='SPY Buy and Hold', color='r')
plt.title('Performance Comparison')
plt.legend()
plt.show()
