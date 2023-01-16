
import pandas as pd
from binance.client import Client
import datetime as dt
import matplotlib.pyplot as plt
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest
from backtesting.test import GOOG


# client configuration
api_key = 'dFGaAHxaE3hRu1bpUooBjBS1oXlqjYTVdLF7dVxssb6mJCRq1BBp43YtXwtegIOC' 
api_secret = 'NdT69wFUGhCUdzpUaOWQMRxhFM2NlkmyaYXqc0MBz968SlApkdKKevUMKF3Hbfz2'
client = Client(api_key, api_secret)


blue_chips={'BTCBUSD', 'ETHBUSD', 'SOLBUSD', 'XRPBUSD', 'ADABUSD', 'MATICBUSD', 'LTCBUSD'}

fig = plt.figure()


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()

class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 50
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()


pos = 1
for symbol in blue_chips:

    print('symbol -> {}'.format(symbol))
    interval='1d'
    print('interval -> {}'.format(interval))
    Client.KLINE_INTERVAL_1DAY 
    klines = client.get_historical_klines(symbol, interval, "1 Jan,2021")
    data = pd.DataFrame(klines)
    # create colums name
    data.columns = ['open_time','Open', 'High', 'Low', 'Close', 'Volume','close_time', 'qav','num_trades','taker_base_vol', 'taker_quote_vol', 'ignore']
                
    # change the timestamp
    data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]

    print('symbol -> {}'.format(symbol))
    data.to_csv(symbol + '.csv', index = None, header=True)

    #convert data to float and plot
    df=pd.read_csv(symbol+'.csv')
    bt = Backtest(df, SmaCross, cash=100_000, commission=.002)
    stats = bt.run()
    print(stats)
    bt.plot(filename=symbol+'_backtest.html')