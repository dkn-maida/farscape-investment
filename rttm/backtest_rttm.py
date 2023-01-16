
import pandas as pd
from binance.client import Client
import datetime as dt
import matplotlib.pyplot as plt
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest
import pandas_ta as ta
from backtesting.test import EURUSD


# client configuration
api_key = 'dFGaAHxaE3hRu1bpUooBjBS1oXlqjYTVdLF7dVxssb6mJCRq1BBp43YtXwtegIOC' 
api_secret = 'NdT69wFUGhCUdzpUaOWQMRxhFM2NlkmyaYXqc0MBz968SlApkdKKevUMKF3Hbfz2'
client = Client(api_key, api_secret)


blue_chips={'BTCBUSD', 'ETHBUSD', 'SOLBUSD', 'XRPBUSD', 'ADABUSD', 'MATICBUSD', 'LTCBUSD'}

for symbol in blue_chips:

    print('symbol -> {}'.format(symbol))
    interval='5m'
    print('interval -> {}'.format(interval))
    Client.KLINE_INTERVAL_1DAY 
    klines = client.get_historical_klines(symbol, interval, "15 Jan,2022")
    data = pd.DataFrame(klines)
    # create colums name
    data.columns = ['open_time','Open', 'High', 'Low', 'Close', 'Volume','close_time', 'qav','num_trades','taker_base_vol', 'taker_quote_vol', 'ignore']
    # change the timestamp
    data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]

    print('symbol -> {}'.format(symbol))
    data.to_csv(symbol + '.csv', index = None, header=True)

    #convert data to float and plot
    df=pd.read_csv(symbol+'.csv')


def BBANDS(data, n, n_std):
 bbands=ta.bbands(close=data.Close.s, length=n, std=n_std)
 return bbands.to_numpy().T[0:3]


class Rttm(Strategy):
    # Define the bb parameters as  *class variables*
    # for later optimization
    n=20
    std=2
    
    def init(self):
        # Precomputupper, self.lower=e the bollinger bands
        self.bbands= self.I(BBANDS, self.data, self.n, self.std)

    def next(self):
        lower_band = self.bbands[0]
        upper_band = self.bbands[2]

        if self.data.Close[-1] <= upper_band[-1] or self.data.Close[-1] >= lower_band[-1]:
            self.position.close()
        if self.data.Close[-1] > upper_band[-1]:
            self.sell()
        elif self.data.Close[-1] < lower_band[-1]:
            self.buy()
                


df=pd.read_csv('BTCBUSD.csv')

bt = Backtest(df, Rttm, cash=100_000, commission=0)
stats = bt.run()
print(stats)



bt.plot()