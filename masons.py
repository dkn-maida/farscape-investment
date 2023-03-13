import pandas as pd
from backtesting import Backtest
from backtesting.lib import SignalStrategy


class Masons(SignalStrategy):
    
    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()
        # Precompute the two moving averages
        self.masons = self.I(self.MASONS)
        signal = self.masons
        # Use 95% of available liquidity (at the time) on each order.
        # (Leaving a value of 1. would instead buy a single share.)
        entry_size = signal * .95
        # Set order entry sizes using the method provided by 
        # `SignalStrategy`. See the docs.
        self.set_signal(entry_size=entry_size)
        
    def MASONS(self):
        pass
        
        
df=pd.read_csv('data/data/BTCBUSD1h.csv')
df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

bt = Backtest(df, Masons, commission=.002)
bt.run()
bt.plot()