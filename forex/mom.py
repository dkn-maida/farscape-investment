import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Currency and cross definitions
currencies = ['USD', 'EUR', 'JPY', 'CNY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD']
crosses = [
    'EURUSD', 'USDJPY', 'USDCNY', 'EURJPY', 'EURCNY', 'JPYCNY', 'GBPUSD', 'USDCHF',
    'EURCHF', 'GBPCHF', 'CHFJPY', 'USDCAD','CADJPY',
    'EURCAD', 'GBPCAD','CADCHF'
]

def get_base_quote(cross):
    pairs = {
        'EURUSD': ('EUR', 'USD'),
        'USDJPY': ('USD', 'JPY'),
        'USDCNY': ('USD', 'CNY'),
        'EURJPY': ('EUR', 'JPY'),
        'EURCNY': ('EUR', 'CNY'),
        'JPYCNY': ('JPY', 'CNY'),
        'GBPUSD': ('GBP', 'USD'),
        'USDCHF': ('USD', 'CHF'),
        'EURCHF': ('EUR', 'CHF'),
        'GBPCHF': ('GBP', 'CHF'),
        'CHFJPY': ('CHF', 'JPY'),
        'USDCAD': ('USD', 'CAD'),
        'AUDUSD': ('AUD', 'USD'),
        'AUDCAD': ('AUD', 'CAD'),
        'AUDJPY': ('AUD', 'JPY'),
        'CADJPY': ('CAD', 'JPY'),
        'EURCAD': ('EUR', 'CAD'),
        'GBPCAD': ('GBP', 'CAD'),
        'EURAUD': ('EUR', 'AUD'),
        'GBPAUD': ('GBP', 'AUD'),
        'AUDCHF': ('AUD', 'CHF'),
        'CADCHF': ('CAD', 'CHF'),
        'AUDNZD': ('AUD', 'NZD'),
        'NZDUSD': ('NZD', 'USD'),
    }
    return pairs[cross]

def get_yahoo_ticker(cross):
    tickers = {
        'EURUSD': 'EURUSD=X',
        'USDJPY': 'JPY=X',
        'USDCNY': 'CNY=X',
        'EURJPY': 'EURJPY=X',
        'EURCNY': 'EURCNY=X',
        'JPYCNY': 'CNYJPY=X',
        'GBPUSD': 'GBPUSD=X',
        'USDCHF': 'CHF=X',
        'EURCHF': 'EURCHF=X',
        'GBPCHF': 'GBPCHF=X',
        'CHFJPY': 'CHFJPY=X',
        'USDCAD': 'CAD=X',
        'CADJPY': 'CADJPY=X',
        'EURCAD': 'EURCAD=X',
        'GBPCAD': 'GBPCAD=X',
        'CADCHF': 'CADCHF=X',
    }
    return tickers[cross]

def fetch_exchange_rates(crosses, start_date, end_date):
    exchange_rates_list = []
    for cross in crosses:
        ticker = get_yahoo_ticker(cross)
        print(f"Fetching data for {cross} ({ticker})...")
        data = yf.download(ticker, start=start_date, end=end_date, interval='1mo')['Adj Close']
        data.name = cross
        exchange_rates_list.append(data)
    exchange_rates = pd.concat(exchange_rates_list, axis=1)
    exchange_rates.index = pd.to_datetime(exchange_rates.index)
    exchange_rates = exchange_rates.sort_index()
    return exchange_rates

def parse_interest_rate_data(file_name):
    # Read the data from CSV file
    df = pd.read_csv(file_name, dayfirst=True)
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    # Identify the column that contains interest rate values
    rate_column = 'Interest Rate' if 'Interest Rate' in df.columns else 'Close'
    # Convert the rate column to numeric
    df[rate_column] = pd.to_numeric(df[rate_column], errors='coerce')
    # Sort the DataFrame by date
    df.sort_index(inplace=True)
    # Divide by 100 to convert percentage to decimal if necessary
    interest_rates = df[rate_column] / 100
    return interest_rates

def fetch_interest_rates(dates):
    # Fetch U.S. interest rates
    us_rates = parse_interest_rate_data('usd.csv')
    us_rates = us_rates.reindex(dates).fillna(method='ffill')

    # Fetch China's interest rates
    china_rates = parse_interest_rate_data('cny.csv')
    china_rates = china_rates.reindex(dates).fillna(method='ffill')

    # Fetch Euro interest rates
    euro_rates = parse_interest_rate_data('eur.csv')
    euro_rates = euro_rates.reindex(dates).fillna(method='ffill')

    # Fetch Japan interest rates
    japan_rates = parse_interest_rate_data('jpy.csv')
    japan_rates = japan_rates.reindex(dates).fillna(method='ffill')

    # Fetch UK interest rates
    uk_rates = parse_interest_rate_data('gbp.csv')
    uk_rates = uk_rates.reindex(dates).fillna(method='ffill')

    # Fetch Swiss interest rates
    swiss_rates = parse_interest_rate_data('chf.csv')
    swiss_rates = swiss_rates.reindex(dates).fillna(method='ffill')

    # Fetch Canadian interest rates
    cad_rates = parse_interest_rate_data('cad.csv')
    cad_rates = cad_rates.reindex(dates).fillna(method='ffill')

    # Create the interest rates DataFrame
    interest_rates = pd.DataFrame(index=dates)
    interest_rates['USD'] = us_rates.values
    interest_rates['CNY'] = china_rates.values
    interest_rates['EUR'] = euro_rates.values
    interest_rates['JPY'] = japan_rates.values
    interest_rates['GBP'] = uk_rates.values
    interest_rates['CHF'] = swiss_rates.values
    interest_rates['CAD'] = cad_rates.values
    return interest_rates

def compute_interest_rate_differential(interest_rates, crosses):
    interest_rate_diff = pd.DataFrame(index=interest_rates.index, columns=crosses)
    for cross in crosses:
        base, quote = get_base_quote(cross)
        interest_rate_diff[cross] = interest_rates[base] - interest_rates[quote]
    return interest_rate_diff

def adjust_exchange_rates(exchange_rates, interest_rate_diff):
    adjusted_exchange_rates = exchange_rates * (1 + interest_rate_diff * (1/12))
    return adjusted_exchange_rates

def compute_reverse_exchange_rates(exchange_rates):
    reverse_exchange_rates = pd.DataFrame(index=exchange_rates.index, columns=crosses)
    for cross in crosses:
        reverse_exchange_rates[cross] = 1 / exchange_rates[cross]
    return reverse_exchange_rates

def adjust_reverse_exchange_rates(reverse_exchange_rates, interest_rate_diff):
    adjusted_reverse_exchange_rates = reverse_exchange_rates * (1 + (-interest_rate_diff) * (1/12))
    return adjusted_reverse_exchange_rates

def generate_signals(adjusted_exchange_rates, adjusted_reverse_exchange_rates, exchange_rates, reverse_exchange_rates, lookback_periods):
    signals = pd.DataFrame(index=adjusted_exchange_rates.index, columns=crosses)
    for cross in crosses:
        adj_rate_t = adjusted_exchange_rates[cross]
        bullish = True
        bearish = True
        for lb in lookback_periods:
            rate_t_lb = exchange_rates[cross].shift(lb)
            bullish &= adj_rate_t > rate_t_lb
            rate_reverse_t_lb = reverse_exchange_rates[cross].shift(lb)
            bearish &= adjusted_reverse_exchange_rates[cross] > rate_reverse_t_lb
        signal = np.where(bullish, 1, np.where(bearish, -1, 0))
        signals[cross] = signal
    return signals

def compute_returns(signals, exchange_rates, interest_rate_diff, initial_capital):
    positions = pd.DataFrame(index=signals.index, columns=crosses)
    pnl = pd.DataFrame(index=signals.index, columns=crosses)
    for cross in crosses:
        position_size = initial_capital  # Fixed position size per signal
        position = signals[cross].shift(1) * position_size  # Position held during period t-1 to t
        positions[cross] = position
        fx_return = exchange_rates[cross].pct_change()
        carry_return = interest_rate_diff[cross].shift(1) * (1/12)
        total_return = fx_return + carry_return
        pnl[cross] = position * total_return
    return pnl, positions

def compute_total_returns(pnl, initial_capital):
    total_pnl = pnl.sum(axis=1)
    portfolio_return = total_pnl / initial_capital  # Return relative to initial capital
    return portfolio_return

def compute_performance_metrics(portfolio_return):
    cumulative_returns = (1 + portfolio_return).cumprod()
    total_period = len(portfolio_return) / 12
    annualized_return = cumulative_returns.iloc[-1] ** (1 / total_period) - 1
    annualized_volatility = portfolio_return.std() * np.sqrt(12)
    sharpe_ratio = annualized_return / annualized_volatility
    cumulative_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / cumulative_max - 1
    max_drawdown = drawdown.min()
    return annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, cumulative_returns

def main():
    initial_capital = 10000  # Fixed capital
    start_date = '2010-01-01'  # Adjusted start date to match the interest rate data
    end_date = '2024-10-30'
    # Fetch exchange rates
    print("Fetching exchange rate data...")
    exchange_rates = fetch_exchange_rates(crosses, start_date, end_date)
    exchange_rates.dropna(inplace=True)
    dates = exchange_rates.index
    # Fetch interest rates
    interest_rates = fetch_interest_rates(dates)
    # Compute interest rate differentials
    interest_rate_diff = compute_interest_rate_differential(interest_rates, crosses)
    # Adjust exchange rates
    adjusted_exchange_rates = adjust_exchange_rates(exchange_rates, interest_rate_diff)
    # Compute reverse exchange rates
    reverse_exchange_rates = compute_reverse_exchange_rates(exchange_rates)
    # Adjust reverse exchange rates
    adjusted_reverse_exchange_rates = adjust_reverse_exchange_rates(reverse_exchange_rates, interest_rate_diff)
    # Generate signals
    lookback_periods = [2, 3, 4]
    signals = generate_signals(adjusted_exchange_rates, adjusted_reverse_exchange_rates, exchange_rates, reverse_exchange_rates, lookback_periods)
    # Compute P&L and positions
    pnl, positions = compute_returns(signals, exchange_rates, interest_rate_diff, initial_capital)
    # Compute portfolio returns
    portfolio_return = compute_total_returns(pnl, initial_capital)
    # Compute performance metrics
    annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, cumulative_returns = compute_performance_metrics(portfolio_return)
    # Print performance metrics
    print(f"Annualized Return: {annualized_return*100:.2f}%")
    print(f"Annualized Volatility: {annualized_volatility*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    # Plot cumulative returns
    plt.figure(figsize=(12,6))
    plt.plot(cumulative_returns)
    plt.title('Cumulative Returns of the Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()
    # Plot leverage over time
    total_exposure = positions.abs().sum(axis=1)
    leverage = total_exposure / initial_capital
    plt.figure(figsize=(12,6))
    plt.plot(leverage)
    plt.title('Leverage Over Time')
    plt.xlabel('Date')
    plt.ylabel('Leverage (Total Exposure / Initial Capital)')
    plt.grid(True)
    plt.show()
    avg_leverage = leverage.mean()
    print(f"Average Leverage: {avg_leverage:.2f}")

if __name__ == "__main__":
    main()