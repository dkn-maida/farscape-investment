import pandas as pd
from backtesting import Strategy
from backtesting import Backtest
from backtesting.lib import resample_apply

#blue_chips={'BTCBUSD', 'ETHBUSD', 'SOLBUSD', 'XRPBUSD', 'ADABUSD', 'MATICBUSD', 'LTCBUSD'}
blue_chips={'BTCBUSD'}

class Minmax(Strategy):

    def MAX(self):
        pass

    def init(self):
        super().init()
        print(self.data)

    def next(self):
        pass

def run_backtest(interval):
    for symbol in blue_chips:
        #convert data to float and plot
        df=pd.read_csv('../data/' + symbol + interval + '.csv', index_col='open_time')
        del df['qav']
        del df['num_trades']
        del df['taker_base_vol']
        del df['taker_quote_vol']
        del df['ignore']
    
        print(df)
        bt = Backtest(df, Minmax, cash=100_000, commission=.002)
        stats = bt.run()
        print(stats)
        bt.plot(filename=symbol+'_backtest.html')

run_backtest('1d')