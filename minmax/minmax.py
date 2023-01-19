
import pandas as pd
from binance.client import Client
import datetime as dt
import matplotlib.pyplot as plt
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest


# client configuration
api_key = 'dFGaAHxaE3hRu1bpUooBjBS1oXlqjYTVdLF7dVxssb6mJCRq1BBp43YtXwtegIOC' 
api_secret = 'NdT69wFUGhCUdzpUaOWQMRxhFM2NlkmyaYXqc0MBz968SlApkdKKevUMKF3Hbfz2'
client = Client(api_key, api_secret)


#blue_chips={'BTCBUSD', 'ETHBUSD', 'SOLBUSD', 'XRPBUSD', 'ADABUSD', 'MATICBUSD', 'LTCBUSD'}
blue_chips={'BTCBUSD'}

def MAX(values, n):
    return pd.Series(values).rolling(n).max()
def MIN(values, n):
    return pd.Series(values).rolling(n).min()

class SmaCross(Strategy):
    n = 90

    def init(self):
        self.max= self.I(MAX, self.data.Close, self.n)
        self.min= self.I(MIN, self.data.Close, self.n)
    def next(self):
        if crossover(self.data.Close,  self.max):
            self.position.close()
            self.buy()
        if self.data.Close < self.max:
            self.position.close()

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