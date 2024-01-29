import numpy as np

CODE_LIST = ["000001"]
DATE_LIST = ["20231207"]


# Private_Variable_Information 
FEATURE_SET_LOB = [
    'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
    'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',  
    'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5', 
    'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5', 
    'high_low_price_diff', 'close_price', 'volume', 'vwap', 'time_diff'
]

# Market_Variable_Information
FEATURE_SET_FULL = FEATURE_SET_LOB + [
    'ask_bid_spread', 'ab_volume_misbalance', 'transaction_net_volume', 
    'volatility', 'trend', 'immediate_market_order_cost_bid', 
    'VOLR', 'PCTN_1min', 'MidMove_1min', 'weighted_price', 'order_imblance', 
    'trend_strength'
]


class DefaultConfig(object):
    path_raw_data = './data/tick_data_csv'
    path_pkl_data = './data/bar_data_pkl'
    result_path = 'results/exp_env'

    code_list = CODE_LIST
    date_list = DATE_LIST

    # ############################### Trade Setting Parameters ###############################
    # Planning horizon is 30mins
    simulation_planning_horizon = 30
    # Order volume = total volume / simulation_num_shares
    simulation_num_shares = 10
    # Total volume to trade w.r.t. the basis volume
    simulation_volume_ratio = 0.005
    # Encourage a uniform liquidation strategy
    simulation_linear_reg_coeff = 0.1
    # Features used for the market variable
    simulation_features = FEATURE_SET_FULL
    # Stack the features of the previous x bars
    simulation_loockback_horizon = 5
    # Whether return flattened or stacked features of the past x bars
    simulation_do_feature_flatten = True
    # A liquidation task
    simulation_direction = 'sell'
    # If the quantity is not fully filled at the last time step, 
    #   we place an MO to fully liquidate and further plus a penalty (unit: bp)
    simulation_not_filled_penalty_bp = 2.0
    # Use discrete actions (unit: relative bp)
    simulation_discrete_actions = \
        np.concatenate([[-50, -40, -30, -25, -20, -15], np.linspace(-10, 10, 21), [15, 20, 25, 30, 40, 50]])
    # Scale the price delta if we use continuous actions
    simulation_continuous_action_scale = 10
    # Use 'discrete' or 'continuous' action space?
    simulation_action_type = 'discrete'
    # ############################### END ###############################