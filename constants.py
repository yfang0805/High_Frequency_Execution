import numpy as np

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
    path_raw_data = './data/toy_data/tick_data_csv'
    path_pkl_data = './data/toy_data/bar_data_pkl'
    result_path = 'results/exp_env'

    code_list = ["000001"]
    date_list = ["20231207"]

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


# CSI300 Instrument (2024.1.31)
CODE_LIST_CSI300 = ['000999', '001289', '001979', '002001', '002007', '002027', '002049', '002050', '002074', '002129', '002142', '002179', '002180', '002202', '002230', '002236', '002241', '002252', '002271', '002304', '002311', '002352', '002371', '002410', '002415',\
                '300450', '300454', '300496', '300498', '300628', '300661', '300750', '300751', '300759', '300760', '300763', '300782', '300896', '300919', '300957', '300979', '300999', '301269', '600000', '600009', '600010', '600011', '600015', '600016', '600018',\
                '600460', '600489', '600515', '600519', '600547', '600570', '600584', '600585', '600588', '600600', '600606', '600660', '600674', '600690', '600732', '600741', '600745', '600754', '600760', '600795', '600803', '600809', '600837', '600845', '600875',\
                '601319', '601328', '601336', '601360', '601377', '601390', '601398', '601600', '601601', '601607', '601615', '601618', '601628', '601633', '601658', '601668', '601669', '601688', '601689', '601698', '601699', '601728', '601766', '601788', '601799',\
                '603799', '603806', '603833', '603899', '603986', '603993', '605117', '605499', '688008', '688012', '688036', '688041', '688065', '688111', '688126', '688187', '688223', '688256', '688271', '688303', '688363', '688396', '688561', '688599', '688981'
            ]


DATE_LIST_202312 = ['20231201', '20231204', '20231205', '20231206', '20231207', '20231208', '20231211', '20231212', '20231213', \
                    '20231214', '20231215', '20231218', '20231219', '20231220', '20231221', '20231222', '20231225', '20231226', ]

DATE_LIST_202312_Validation = ['20231227', '20231228', '20231229']


class CSI300Config(object):
    path_raw_data = './data/CSI300/tick_data_csv'
    path_pkl_data = './data/CSI300/bar_data_pkl'
    result_path = 'results/exp_env'

    code_list = CODE_LIST_CSI300
    date_list = DATE_LIST_202312

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