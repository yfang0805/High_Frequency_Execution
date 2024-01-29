import os
import pickle
import numpy as np

from config import DefaultConfig


class Data(object):
    price_5level_features = [
        'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
        'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5', 
    ]
    other_price_features = [
        'high_price', 'low_price', 'open_price', 'close_price', 'vwap',
    ]
    price_delta_features = [
        'ask_bid_spread', 'trend', 'immediate_market_order_cost_ask', 
        'immediate_market_order_cost_bid', 'volatility', 'high_low_price_diff',
    ]
    volume_5level_features = [
        'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',  
        'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5', 
    ]
    other_volume_features = [
        'volume', 'ab_volume_misbalance', 'transaction_net_volume', 
        'VOLR', 'BSP',
    ]
    backtest_lo_features = [
        'max_last_price', 'min_last_price', 'ask1_deal_volume', 'bid1_deal_volume',
    ]

    def __init__(self, config):
        self.config = config
        self.data = None
        self.backtest_data = None

    
    def data_exists(self, code="000001", date="20231207"):
        return os.path.isfile(os.path.join(self.config.path_pkl_data, code, date + '.pkl'))

    def obtain_data(self, code, date, start_index=None, do_normalization=True):
        with open(os.path.join(self.config.path_pkl_data, code, date + '.pkl'), 'rb') as f:
            self.data = pickle.load(f)
        assert self.data.shape[0] == 239, \
            'The data should be of the shape (239, 42), instead of {}'.format(self.data.shape)
        if start_index is None:
            start_index = self._random_valid_start_index()
            self._set_horizon(start_index)
        else:
            self._set_horizon(start_index)
            assert self._sanity_check(), "code={} date={} with start_index={} is invalid".format(code, date, start_index)
        self._maintain_backtest_data()
        if do_normalization:
            self._normalization()
    
    def step(self):
        self.current_index += 1
    
    def obtain_features(self, do_flatten=True):
        features = self.data.loc[self.current_index - self.config.simulation_loockback_horizon + 1: self.current_index, 
            self.config.simulation_features][::-1].values
        if do_flatten:
            return features.flatten()
        else:
            return features
    
    def obtain_feature(self, feature):
        return self.data.loc[self.current_index, feature]

    def obtain_future_features(self, features):
        return self.data.loc[self.current_index:self.end_index, features]

    def obtain_level(self, name, level=''):
        return self.backtest_data.loc[self.current_index, '{}{}'.format(name, level)]

    def _maintain_backtest_data(self):

        self.backtest_data =\
            self.data[self.price_5level_features + self.volume_5level_features + self.backtest_lo_features].copy()
        self.backtest_data['latest_price'] = \
            (self.backtest_data['askPrice1'] + self.backtest_data['bidPrice1']) / 2
        
    def _normalization(self):

        # Keep normalization units
        self.basis_price = self.backtest_data.loc[self.start_index, "latest_price"]
        self.basis_volume = self.data['basis_volume'].values[0]

        # Approximation: Average price change 2% * 50 = 1.0
        self.data[self.price_5level_features] = \
            (self.data[self.price_5level_features] - self.basis_price) / self.basis_price * 50
        self.data[self.other_price_features] = \
            (self.data[self.other_price_features] - self.basis_price) / self.basis_price * 50
        self.data[self.price_delta_features] = \
            self.data[self.price_delta_features] / self.basis_price * 10

        # Such that the volumes are equally distributed in the range [-1, 1]
        self.data[self.volume_5level_features] = \
            self.data[self.volume_5level_features] / self.basis_volume * 100
        self.data[self.other_volume_features] = \
            self.data[self.other_volume_features] / self.basis_volume * 100

    def _set_horizon(self, start_index):
        self.start_index = start_index
        self.current_index = self.start_index
        self.end_index = self.start_index + self.config.simulation_planning_horizon

    def _random_valid_start_index(self):
        cols = ['bidPrice1', 'bidVolume1', 'askPrice1', 'askVolume1']

        tmp = (self.data[cols] > 0).all(axis=1)
        tmp1 = tmp.rolling(self.config.simulation_loockback_horizon).apply(lambda x: x.all())
        tmp2 = tmp[::-1].rolling(self.config.simulation_planning_horizon + 1).apply(lambda x: x.all())[::-1]
        available_indx = tmp1.loc[(tmp1 > 0) & (tmp2  > 0)].index.tolist()
        assert len(available_indx) > 0, "The data is invalid"
        return np.random.choice(available_indx)
    
    def _sanity_check(self):
        """ When the price reaches daily limit, the price and volume"""
        cols = ['bidPrice1', 'bidVolume1', 'askPrice1', 'askVolume1']
        if (self.data.loc[self.start_index:self.end_index, cols] == 0).any(axis=None):
            return False
        else:
            return True

def run_Data():
    
    config = DefaultConfig()
    data = Data(config)
    data.obtain_data(code="000001", date="20231207")
    print("Total Length:", len(data.data))
    print("Index:", data.current_index)
    data.step()
    print("Index:", data.current_index)

    print("Obtain Features:")
    print(np.shape(data.obtain_features(do_flatten=False)))
    print("Future Features:")
    print(np.shape(data.obtain_future_features(data.price_5level_features)))
        
if __name__ == "__main__":
    run_Data()