import pandas as pd

class DataManager:

    def __init__(self, csv_path, date_col):

        self.data = pd.read_csv(csv_path, parse_dates=[date_col],
                                index_col=date_col)
        self.data['t_plus'] = self.data.open.shift(-1)
        self.data.dropna(inplace=True)

        self.df = self.data.copy()

        self.timeframe = '1min'


    def change_resolution(self, new_timeframe):

        resample_dict = {'volume': 'sum', 'open': 'first',
                         'low': 'min', 'high': 'max',
                         'close': 'last',
                         't_plus': 'last'}

        self.df = self.data.resample(new_timeframe).agg(resample_dict)

        self.timeframe = new_timeframe