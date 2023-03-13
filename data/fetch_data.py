import pandas as pd
from binance.client import Client
import datetime as dt

# client configuration
api_key = 'dFGaAHxaE3hRu1bpUooBjBS1oXlqjYTVdLF7dVxssb6mJCRq1BBp43YtXwtegIOC' 
api_secret = 'NdT69wFUGhCUdzpUaOWQMRxhFM2NlkmyaYXqc0MBz968SlApkdKKevUMKF3Hbfz2'
client = Client(api_key, api_secret)

blue_chips={'BTCBUSD', 'ETHBUSD', 'SOLBUSD', 'XRPBUSD', 'ADABUSD', 'MATICBUSD', 'LTCBUSD'}

def fetch(interval):
    for symbol in blue_chips:
        print('symbol -> {}'.format(symbol))
        print('interval -> {}'.format(interval))
        Client.KLINE_INTERVAL_1HOUR 
        klines = client.get_historical_klines(symbol, interval, "1 Jan,2022")
        data = pd.DataFrame(klines)
        # create colums name
        data.columns = ['timestamp','Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore']
        # delete not used columns
        del data["qav"]
        del data["num_trades"]
        del data["taker_base_vol"]
        del data["taker_quote_vol"]
        del data["ignore"]
        del data["close_time"] 
        
        data["timestamp"]=pd.to_datetime(data["timestamp"], unit='ms')
        
        print('symbol -> {}'.format(symbol))
        data.to_csv('data/'+symbol + interval +'.csv', index = None, header=True)

fetch('1h')
fetch('1m')