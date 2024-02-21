import os
import pdb
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from pathos.multiprocessing import ProcessingPool as Pool

import warnings
warnings.filterwarnings('ignore')
NUM_CORES = 40



class DataPrepare(object):
    """
    For data preparation: 
        Parse raw csv files to pickle files required by the simulated environment 
        I.e., we transform time-tick-level csv data into bar level pkl data
    """
    def __init__(self, config):

        self.config = config

        if not os.path.isdir(self.config.path_raw_data):
            print(self.config)
            self.download_raw_data()

        os.makedirs(self.config.path_pkl_data, exist_ok=True)
        file_paths = self.obtain_file_paths()
        
        # pool = Pool(NUM_CORES)
        # res = pool.map(self.process_file, file_paths)
        info = []
        for csv_path, pkl_path in tqdm(file_paths):
            if os.path.exists(pkl_path):
                continue
            res = self.process_file(csv_path, pkl_path)
            info.append(res)
        pd.DataFrame(info).to_csv('data_generation_report.csv')

    def download_raw_data(self):

        raise NotImplementedError
    

    def process_file(self, csv_path, pkl_path, debug=True):
        '''
        Input:
            The csv needs to contain
            [tradeDate, dataTime, volume_dt, value_dt, lastPrice
            askPrice1 to askPrice5, askVolume1 to askVolume5, bidPrice1 to bidPrice5, bidVolume1 to bidVolume5]

        Return:
            pkl_file with
            
            ['bidPrice1', 'bidVolume1', 'bidPrice2', 'bidVolume2', 'bidPrice3', 'bidVolume3', 'bidPrice4', 'bidVolume4', 'bidPrice5', 'bidVolume5', 
            'askPrice1', 'askVolume1', 'askPrice2', 'askVolume2', 'askPrice3', 'askVolume3', 'askPrice4', 'askVolume4', 'askPrice5', 'askVolume5']

            ['max_last_price', 'min_last_price', 'ask1_deal_volume', 'bid1_deal_volume']
            
            # Normalization constant
            ['basis_price', 'basis_volume']

            # Bar information
            ['high_price','low_price', 'high_low_price_diff', 'open_price', 'close_price', 'volume', 'vwap']

            # LOB features
            ['ask_bid_spread','ab_volume_misbalance', 'transaction_net_volume', 'volatility', 'trend', 'immediate_market_order_cost_ask', 'immediate_market_order_cost_bid']

            # new LOB features
            ['VOLR', 'PCTN_1min', 'MidMove_1min', 'BSP', 'weighted_price', 'order_imblance', 'trend_strength', 'time', 'time_diff']

        '''

        # Step 1: Read data
        data = pd.read_csv(csv_path)
        data["value"] = data["value_dt"].cumsum()
        data["volume"] = data["volume_dt"].cumsum()

        csv_shape0, csv_shape1 = data.shape

        # Filter out abnormal files (e.g., the stock is not traded on this day)
        if csv_shape0 == 1:
            return dict(csv_path=csv_path, pkl_path=pkl_path, status='EMPTY')
        if data['volume'].max() <= 0:
            return dict(csv_path=csv_path, pkl_path=pkl_path, status='NO_VOL')
        if data['lastPrice'][data['lastPrice'] > 0].mean() >= 1.09 * data['lastPrice'][data['lastPrice'] > 0].values[0]:
            return dict(csv_path=csv_path, pkl_path=pkl_path, status='LIMIT_UP')
        if data['lastPrice'][data['lastPrice'] > 0].mean() <= 0.91 * data['lastPrice'][data['lastPrice'] > 0].values[0]:
            return dict(csv_path=csv_path, pkl_path=pkl_path, status='LIMIT_DO')

        if debug:
            print('Current process: {} {} Shape: {}'.format(csv_path, pkl_path, data.shape))
            assert csv_shape1 == 28

        # Step 2: Formatting the timeline
        data['tradeDate']= data['tradeDate'].apply(lambda x: str(x))
        trade_date = data.iloc[0]["tradeDate"]
        data['dataTime'] = data['dataTime'].apply(lambda x: str(x))
        data.index = pd.to_datetime(data['tradeDate'] + ' ' + data['dataTime'], format='%Y-%m-%d %H:%M:%S')
        data['time'] = data.index
        data = data.resample('3S', closed='right', label='right').last().fillna(method='ffill')

        # Exclude call auction
        data = data[data['time'].between(trade_date + ' 09:30:00', trade_date + ' 14:57:00')]
        data = data[~data['time'].between(trade_date + ' 11:30:01', trade_date + ' 12:59:59')]

        # Step 3: Backtest required bar-level information
        # Convert to 1min bar
        #   1) current snapshot (5 levels of ask/bid price/volume)
        #   2) the lowest/highest ask/bid price that yields partial execution
        ask1_deal_volume_tick = ((data['value_dt'] - data['volume_dt'] * data['bidPrice1']) \
            / (data['askPrice1'] - data['bidPrice1'])).clip(upper=data['volume_dt'], lower=0)
        bid1_deal_volume_tick = ((data['volume_dt'] * data['askPrice1'] - data['value_dt']) \
            / (data['askPrice1'] - data['bidPrice1'])).clip(upper=data['volume_dt'], lower=0)

        max_last_price = data['lastPrice'].resample('T').max().reindex(data.index).fillna(method='ffill')
        min_last_price = data['lastPrice'].resample('T').min().reindex(data.index).fillna(method='ffill')

        ask1_deal_volume = ((data['askPrice1'] == max_last_price) * ask1_deal_volume_tick).resample('T').sum()
        bid1_deal_volume = ((data['bidPrice1'] == min_last_price) * bid1_deal_volume_tick).resample('T').sum()
        max_last_price = data['askPrice1'].resample('T').max()
        min_last_price = data['bidPrice1'].resample('T').min()

        # Current 5-level ask/bid price/volume (for modeling temporary market impact of MOs)
        level_infos = ['bidPrice1', 'bidVolume1', 'bidPrice2', 'bidVolume2', 'bidPrice3', 'bidVolume3', 'bidPrice4',
            'bidVolume4', 'bidPrice5', 'bidVolume5', 'askPrice1', 'askVolume1', 'askPrice2', 'askVolume2', 'askPrice3', 
            'askVolume3', 'askPrice4', 'askVolume4', 'askPrice5', 'askVolume5']
        bar_data = data[level_infos].resample('T').first()

        bar_data.iloc[-1].replace(0.0, np.nan, inplace=True)
        bar_data.fillna(method='ffill', inplace=True)

        # Lowest ask/bid executable price and volume till the next bar (for modeling temporary market impact of LOs)
        bar_data['max_last_price'] = max_last_price
        bar_data['min_last_price'] = min_last_price
        bar_data['ask1_deal_volume'] = ask1_deal_volume
        bar_data['bid1_deal_volume'] = bid1_deal_volume

        # Step 4: Generate state features
        # Normalization constant
        bar_data['basis_price'] = data['lastPrice'].values[0]
        bar_data['basis_volume'] = data['volume'].values[-1]         # TODO: change this to total volume of the last day instead of the current day

        # Bar information
        bar_data['high_price'] = data['lastPrice'].resample('T', closed='right', label='right').max()
        bar_data['low_price'] = data['lastPrice'].resample('T', closed='right', label='right').min()
        bar_data['high_low_price_diff'] = bar_data['high_price'] - bar_data['low_price']
        bar_data['open_price'] = data['lastPrice'].resample('T', closed='right', label='right').first()
        bar_data['close_price'] = data['lastPrice'].resample('T', closed='right', label='right').last()
        bar_data['volume'] = data['volume_dt'].resample('T', closed='right', label='right').sum()
        bar_data['vwap'] = data['value_dt'].resample('T', closed='right', label='right').sum() / bar_data['volume']
        bar_data['vwap'] = bar_data['vwap'].fillna(bar_data['close_price'])

        # LOB features
        bar_data['ask_bid_spread'] = bar_data['askPrice1'] - bar_data['bidPrice1']
        bar_data['ab_volume_misbalance'] = \
            (bar_data['askVolume1'] + bar_data['askVolume2'] + bar_data['askVolume3'] + bar_data['askVolume4'] + bar_data['askVolume5']) \
            - (bar_data['bidVolume1'] + bar_data['bidVolume2'] + bar_data['bidVolume3'] + bar_data['bidVolume4'] + bar_data['bidVolume5']) 
        bar_data['transaction_net_volume'] = (ask1_deal_volume_tick - bid1_deal_volume_tick).resample('T', closed='right', label='right').sum()
        bar_data['volatility'] = data['lastPrice'].rolling(20, min_periods=1).std().fillna(0).resample('T', closed='right', label='right').last()
        bar_data['trend'] = (data['lastPrice'] - data['lastPrice'].shift(20)).fillna(0).resample('T', closed='right', label='right').last()
        bar_data['immediate_market_order_cost_ask'] = self._calculate_immediate_market_order_cost(bar_data, 'ask')
        bar_data['immediate_market_order_cost_bid'] = self._calculate_immediate_market_order_cost(bar_data, 'bid')

        # new LOB features
        bar_data['VOLR'] = self._VOLR(data).fillna(0).resample('T', closed='right', label='right').last()
        bar_data['PCTN_1min'] = self._PCTN(data, n=20).fillna(0).resample('T', closed='right', label='right').last()
        bar_data['MidMove_1min'] = self._MidMove(data, n=20).fillna(0).resample('T', closed='right', label='right').last()
        bar_data['BSP'] = self._BSP(data).fillna(0).resample('T', closed='right', label='right').last()
        bar_data['weighted_price'] = self._weighted_price(data).fillna(0).resample('T', closed='right', label='right').last()
        bar_data['order_imblance'] = self._order_imblance(data).fillna(0).resample('T', closed='right', label='right').last()
        bar_data['trend_strength'] = self._trend_strength(data, n=20).fillna(0).resample('T', closed='right', label='right').last()

        bar_data['time'] = bar_data.index
        bar_data = bar_data[bar_data['time'].between(trade_date + ' 09:30:00', trade_date + ' 14:57:00')]
        bar_data = bar_data[~bar_data['time'].between(trade_date + ' 11:30:01', trade_date + ' 12:59:59')]
        bar_data['time_diff'] = (bar_data['time'] - bar_data['time'].values[0]) / np.timedelta64(1, 'm') / 330
        bar_data = bar_data.reset_index(drop=True)

        # Step 5: Save to pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(bar_data, f, pickle.HIGHEST_PROTOCOL)
            print(f"Success process Data and save at {pkl_path}")

        return dict(csv_path=csv_path, pkl_path=pkl_path, 
            csv_shape0=csv_shape0, csv_shape1=csv_shape1, 
            res_shape0=bar_data.shape[0], res_shape1=bar_data.shape[1])
    

    @staticmethod
    def _calculate_immediate_market_order_cost(bar_data, direction='ask'):

        # Assume the market order quantity is 1/500 of the basis volume
        remaining_quantity = (bar_data['basis_volume'] / 500).copy()
        total_fee = pd.Series(0, index=bar_data.index)
        for i in range(1, 6):
            total_fee = total_fee \
                + bar_data['{}Price{}'.format(direction, i)] \
                * np.minimum(bar_data['{}Volume{}'.format(direction, i)], remaining_quantity)
            remaining_quantity = (remaining_quantity - bar_data['{}Volume{}'.format(direction, i)]).clip(lower=0)
        if direction == 'ask':
            return total_fee / (bar_data['basis_volume'] / 500) - bar_data['askPrice1']
        elif direction == 'bid':
            return bar_data['bidPrice1'] - total_fee / (bar_data['basis_volume'] / 500)

    def obtain_file_paths(self):

        file_paths = []
        tickers = os.listdir(self.config.path_raw_data)
        for ticker in tickers:
            dates = os.listdir(os.path.join(self.config.path_raw_data, ticker))
            file_paths.extend([
                (os.path.join(self.config.path_raw_data, ticker, date), 
                 os.path.join(self.config.path_pkl_data, ticker, date.split('.')[0] + '.pkl')) for date in dates])
            os.makedirs(os.path.join(self.config.path_pkl_data, ticker), exist_ok=True)
        return file_paths

    @staticmethod
    def _VOLR(df, beta1=0.551, beta2=0.778, beta3=0.699):
        """
        Volume Ratio: 
            reflects the supply and demand of investment behavior.
        Unit: Volume
        """

        volr = beta1 * (df['bidVolume1'] - df['askVolume1']) / (df['bidVolume1'] + df['askVolume1']) + \
            beta2 * (df['bidVolume2'] - df['askVolume2']) / (df['bidVolume2'] + df['askVolume2']) + \
            beta3 * (df['bidVolume3'] - df['askVolume3']) / (df['bidVolume3'] + df['askVolume3'])

        return volr

    @staticmethod
    def _PCTN(df, n):
        """
        Price Percentage Change: 
            a simple mathematical concept that represents the degree of change over time,
            it is used for many purposes in finance, often to represent the price change of a security.
        Unit: One
        """

        mid = (df['askPrice1'] + df['bidPrice1']) / 2
        pctn = (mid - mid.shift(n)) / mid

        return pctn

    @staticmethod
    def _MidMove(df, n):
        """
        Middle Price Move: 
            indicates the movement of middle price, which can simply be defined as the average of 
            the current bid and ask prices being quoted.
        Unit: One
        """

        mid = (df['askPrice1'] + df['bidPrice1']) / 2
        mean = mid.rolling(n).mean()
        mid_move = (mid - mean) / mean
        return mid_move

    @staticmethod
    def _BSP(df):
        """
        Buy-Sell Pressure: 
            the distribution of chips in the buying and selling direction.
        Unit: Volume
        """

        EPS = 1e-5
        mid = (df['askPrice1'] + df['bidPrice1']) / 2

        w_buy_list = []
        w_sell_list = []

        for level in range(1, 6):
            w_buy_level = mid / (df['bidPrice{}'.format(level)] - mid - EPS)
            w_sell_level = mid / (df['askPrice{}'.format(level)] - mid + EPS)

            w_buy_list.append(w_buy_level)
            w_sell_list.append(w_sell_level)

        sum_buy = pd.concat(w_buy_list, axis=1).sum(axis=1)
        sum_sell = pd.concat(w_sell_list, axis=1).sum(axis=1)

        p_buy_list = []
        p_sell_list = []
        for w_buy_level, w_sell_level in zip(w_buy_list, w_sell_list):
            p_buy_list.append((df['bidVolume{}'.format(level)] * w_buy_level) / sum_buy)
            p_sell_list.append((df['askVolume{}'.format(level)] * w_sell_level) / sum_sell)

        p_buy = pd.concat(p_buy_list, axis=1).sum(axis=1)
        p_sell = pd.concat(p_sell_list, axis=1).sum(axis=1)
        p = np.log((p_sell + EPS) / (p_buy + EPS))

        return p

    @staticmethod
    def _weighted_price(df):
        """
        Weighted price: The average price of ask and bid weighted 
            by corresponding volumn (divided by last price).
        Unit: One
        """

        price_list = []
        for level in range(1, 6):

            price_level = (df['bidPrice{}'.format(level)] * df['bidVolume{}'.format(level)] + \
                           df['askPrice{}'.format(level)] * df['askVolume{}'.format(level)]) / \
                          (df['bidVolume{}'.format(level)] + df['askVolume{}'.format(level)])

            price_list.append(price_level)

        weighted_price = pd.concat(price_list, axis=1).mean(axis=1)
        weighted_price = weighted_price / (df['lastPrice'] + 1e-5)
        return weighted_price

    @staticmethod
    def _order_imblance(df):
        """
        Order imbalance: 
            a situation resulting from an excess of buy or sell orders 
            for a specific security on a trading exchange, 
            making it impossible to match the orders of buyers and sellers.
        Unit: One
        """

        oi_list = []
        for level in range(1, 6):

            oi_level = (df['bidVolume{}'.format(level)] - df['askVolume{}'.format(level)]) / \
                (df['bidVolume{}'.format(level)] + df['askVolume{}'.format(level)])

            oi_list.append(oi_level)

        oi = pd.concat(oi_list, axis=1).mean(axis=1)

        return oi

    @staticmethod
    def _trend_strength(df, n):
        """
        Trend strength: describes the strength of the short-term trend.
        Unit: One
        """

        mid = (df['askPrice1'] + df['bidPrice1']) / 2
        diff_mid = mid - mid.shift(1)
        sum1 = diff_mid.rolling(n).sum()
        sum2 = diff_mid.abs().rolling(n).sum()
        TS = sum1 / sum2

        return TS


def run_prepare_data():
    class DefaultConfig(object):
        path_raw_data = './data/toy_data/tick_data_csv'
        path_pkl_data = './data/toy_data/bar_data_pkl'

        code_list = ["000001"]
        date_list = ["20231207"]
    
    config = DefaultConfig()
    dataprepare = DataPrepare(config)


def run_prepare_data_CSI300():
    from constants import CODE_LIST_CSI300, DATE_LIST_202312, DATE_LIST_202312_Validation
    class DefaultConfig(object):
        path_raw_data = './data/CSI300/tick_data_csv'
        path_pkl_data = './data/CSI300/bar_data_pkl'

        code_list = CODE_LIST_CSI300
        date_list = DATE_LIST_202312 + DATE_LIST_202312_Validation

    config = DefaultConfig()
    dataprepare = DataPrepare(config)

if __name__ == "__main__":
    run_prepare_data_CSI300()