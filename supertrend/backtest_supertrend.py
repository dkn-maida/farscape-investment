import pandas as pd
from binance.client import Client
from backtesting import Strategy
import numpy as np
import pandas_ta as pdta
import matplotlib.pyplot as plt
from backtesting import Backtest

client = Client()

klinesT = client.get_historical_klines("BTCBUSD", Client.KLINE_INTERVAL_1HOUR, "01 january 2023")

df = pd.DataFrame(klinesT, columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
df['Close'] = pd.to_numeric(df['Close'])
df['High'] = pd.to_numeric(df['High'])
df['Low'] = pd.to_numeric(df['Low'])
df['Open'] = pd.to_numeric(df['Open'])

del df['ignore']
del df['close_time']
del df['quote_av']
del df['trades']
del df['tb_base_av']
del df['tb_quote_av']

df = df.set_index(df['timestamp'])
df.index = pd.to_datetime(df.index, unit='ms')
del df['timestamp']

df.drop(df.columns.difference(['Open','High','Low','Close','Volume']), 1, inplace=True)

def EMA(data, n):
    return pdta.ema(close=data.Close.s, length=n).to_numpy()
def SRSI(data):
    return pdta.stochrsi(close=data.Close.s).to_numpy()
def SUPERTREND(data, m, l):
    return pdta.supertrend(data['High'].s, data['Low'].s, data['Close'].s,  multiplier=m , length=l).to_numpy()


class Supertrend(Strategy):
    # Define  *class variables*
    # for later optimization
    st_l = 20
    st_m = 3
    
    def init(self):
        # Precompute indicators
        self.ema = self.I(EMA, self.data, 90)
        self.rsi = self.I(SRSI, self.data)
        self.supertrend = self.I(SUPERTREND, self.data, self.st_m, self.st_l)
       
    def next(self):
        pass

bt = Backtest(df, Supertrend, cash=100_000, commission=0)
stats = bt.run()
print(stats)
bt.plot(filename='BTCBUSD_backtest.html')
