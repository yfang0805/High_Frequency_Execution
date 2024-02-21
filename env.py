import os
import pickle
import pandas as pd
import numpy as np

from constants import DefaultConfig
from data_wrapper import Data
class ExecutionEnv(object):
    
    def __init__(self, config):
        self.config = config
        self.current_code = None
        self.current_date = None
        self.data = Data(config)
        self.cash = 0
        self.total_quantity = 0
        self.quantity = 0
        self.valid_code_date_list = self.get_valid_code_date_list()
    
    def get_valid_code_date_list(self):
        code_date_list = []
        for code in self.config.code_list:
            for date in self.config.date_list:
                if self.data.data_exists(code, date):
                    
                    code_date_list.append((code, date))
        return code_date_list
    
    def reset(self, code=None, date=None, start_index=None):

        count = 0
        
        while True:
            if code is None and date is None:
                # Uniformly randomly select a code and date
                ind = np.random.choice(len(self.valid_code_date_list))
                self.current_code, self.current_date = self.valid_code_date_list[ind]
            else:
                self.current_code, self.current_date = code, date
            
            try:
                # if self.data.data_exists(self.current_code, self.current_date):
                self.data.obtain_data(self.current_code, self.current_date, start_index)
                # print(f"Obtain: code={self.current_code} date={self.current_date}")
                break
            except:
                count += 1
                print('Invalid: code={} date={}'.format(self.current_code, self.current_date))
                if count > 100:
                    raise ValueError("code={} date={} is invalid".format(code, date))
            
        self.cash = 0
        self.total_quantity = self.config.simulation_volume_ratio * self.data.basis_volume
        self.quantity = self.total_quantity
        self.latest_price = self.data.obtain_level('latest_price')

        # Notice that we do not use the first time step of the day (with zero volume)
        market_state = self.data.obtain_features(do_flatten=self.config.simulation_do_feature_flatten)
        private_state = self._generate_private_state()

        return market_state, private_state
    
    def step(self, action=dict(price=18.88, quantity=300)):
        if self.config.simulation_direction == "sell":
            return self._step_sell(action)
        else:
            raise NotImplementedError
        
    def _step_sell(self, action=dict(price=18.88, quantity=300)):
        """
        We only consider limit orders.
        If the price is no better than the market order, 
            it will be transformed to market order automatically.
        """

        info = dict(
            code=self.current_code, 
            date=self.current_date, 
            start_index=self.data.start_index, 
            end_index=self.data.end_index,
            current_index=self.data.current_index
        )
        order_quantity = action['quantity']
        pre_quantity = self.quantity
        pre_cash = self.cash
        price_penalty = 0.0

        done = (self.data.current_index + 1 >= self.data.end_index)
        if done:
            action['price'] = 0.0
            action['quantity'] = float('inf')
            price_penalty = self.config.simulation_not_filled_penalty_bp / 10000 * self.data.basis_price

        # Step 1: If can be executed immediately
        for level in range(1, 6):
            if action['quantity'] > 0 and action['price'] <= self.data.obtain_level('bidPrice', level):
                executed_volume = min(self.data.obtain_level('bidVolume', level), action['quantity'], self.quantity)
                self.cash += executed_volume * (self.data.obtain_level('bidPrice', level) - price_penalty)
                self.quantity -= executed_volume
                action['quantity'] -= executed_volume

        # Liquidate all the remaining inventory on the last step
        if done:
            executed_volume = self.quantity
            self.cash += executed_volume * (self.data.obtain_level('bidPrice', 5) - price_penalty)
            self.quantity = 0
            action['quantity'] = 0

        # Step 2: If can be executed until the next bar
        if action['price'] < self.data.obtain_level('max_last_price'):
            executed_volume = min(self.quantity, action['quantity'])
            self.cash += executed_volume * action['price']
            self.quantity -= executed_volume
            action['quantity'] -= executed_volume
        elif action['price'] == self.data.obtain_level('max_last_price'):
            executed_volume = min(self.quantity, action['quantity'], self.data.obtain_level('ask1_deal_volume'))
            self.cash += executed_volume * action['price']
            self.quantity -= executed_volume
            action['quantity'] -= executed_volume

        if action['quantity'] == order_quantity:
            info['status'] = 'NOT_FILLED'
        elif action['quantity'] == 0:
            info['status'] = 'FILLED'
        else:
            info['status'] = 'PARTIAL_FILLED'

        # Step 3: Reward/Done calculation
        if not done:
            self.data.step()

        reward = self._calculate_reward_v1(pre_cash)

        market_state = self.data.obtain_features(do_flatten=self.config.simulation_do_feature_flatten)
        private_state = self._generate_private_state()

        return market_state, private_state, reward, done, info

    
    def get_future(self, features, padding=None):
        future = self.data.obtain_future_features(features)
        if padding is None:
            return future
        else:
            padding_width = padding - future.shape[0]
            future = np.pad(future, ((0, padding_width), (0, 0), 'edge'))
            return future
        
    def _generate_private_state(self):
        elapsed_time = (self.data.current_index - self.data.start_index) / self.config.simulation_planning_horizon
        remaining_quantity = self.quantity / self.total_quantity
        return np.array([elapsed_time, remaining_quantity])

    def _calculate_reward_v1(self, pre_cash):
        _recommand_quantity = self.total_quantity * (self.data.end_index - self.data.current_index) \
            / self.config.simulation_planning_horizon
        basic_reward = (self.cash - pre_cash) / self.data.basis_price / \
            (self.data.basis_volume * self.config.simulation_volume_ratio / self.config.simulation_planning_horizon)
        linear_reg = abs(self.quantity - _recommand_quantity) / \
            (self.data.basis_volume * self.config.simulation_volume_ratio / self.config.simulation_planning_horizon)
        return basic_reward - self.config.simulation_linear_reg_coeff * linear_reg
    
    @property
    def observation_dim(self):
        return len(self.config.simulation_features) * self.config.simulation_loockback_horizon

    def get_metric(self, mtype='IS'):
        # IS: implementation shortfall
        if mtype == 'IS': 
            return self.data.basis_price * (self.total_quantity - self.quantity) - self.cash
        # BP: bp over mid price TWAP
        if mtype == 'BP':
            if self.total_quantity == self.quantity:
                return 0
            avg_price = self.cash / (self.total_quantity - self.quantity)
            TWAP_mid = self.data.backtest_data.loc[self.data.start_index:self.data.end_index, 'latest_price'].mean()
            bp = (avg_price - TWAP_mid) / self.data.basis_price * 10000
            return bp
                
class BaseWrapper(object):
    def __init__(self, env):
        self.env = env 

    def reset(self, code=None, date=None, start_index=None):
        return self.env.reset(code, date, start_index)

    def step(self, action):
        return self.env.step(action)

    @property
    def quantity(self):
        return self.env.quantity

    @property
    def total_quantity(self):
        return self.env.total_quantity

    @property
    def cash(self):
        return self.env.cash

    @property
    def config(self):
        return self.env.config

    @property
    def data(self):
        return self.env.data

    @property
    def observation_dim(self):
        return self.env.observation_dim

    def get_metric(self, mtype='IS'):
        return self.env.get_metric(mtype)

    def get_future(self, features, padding=None):
        return self.env.get_future(features, padding=padding)


class DiscreteActionBaseWrapper(BaseWrapper):
    def __init__(self, env):
        super(DiscreteActionBaseWrapper, self).__init__(env)

    @property
    def action_sample_func(self):
        return lambda: np.random.randint(len(self.discrete_actions))

    @property
    def action_dim(self):
        return len(self.discrete_actions)
        

class DiscretePriceQuantityWrapper(DiscreteActionBaseWrapper):
    def __init__(self, env):
        super(DiscretePriceQuantityWrapper, self).__init__(env)
        self.discrete_actions = self.config.simulation_discrete_actions
        self.simulation_discrete_quantities = self.config.simulation_discrete_quantities
        self.base_quantity_ratio = self.config.simulation_volume_ratio \
            / self.config.simulation_num_shares / self.simulation_discrete_quantities

    def step(self, action):
        price, quantity = self.discrete_actions[action]
        price = np.round((1 + price / 10000) * self.data.obtain_level('askPrice', 1) * 100) / 100
        quantity = self.data.basis_volume * self.base_quantity_ratio * quantity
        return self.env.step(dict(price=price, quantity=quantity))


class DiscreteQuantityNingWrapper(DiscreteActionBaseWrapper):
    """
    Follows [Ning et al 2020]
    Divide the remaining quantity into several parts and trade using MO
    """
    def __init__(self, env):
        super(DiscreteQuantityNingWrapper, self).__init__(env)
        self.discrete_actions = self.config.simulation_discrete_actions

    def step(self, action):
        quantity = self.discrete_actions[action]
        # This ensures that this can be an MO
        price = -50 
        price = np.round((1 + price / 10000) * self.data.obtain_level('askPrice', 1) * 100) / 100
        quantity = self.discrete_actions[action] / (len(self.discrete_actions) - 1) * self.quantity
        return self.env.step(dict(price=price, quantity=quantity))


class DiscreteQuantityWrapper(DiscreteActionBaseWrapper):
    """
    Specify the quantity and trade using MO
    """
    def __init__(self, env):
        super(DiscreteQuantityWrapper, self).__init__(env)
        self.discrete_actions = self.config.simulation_discrete_actions

    def step(self, action):
        quantity = self.discrete_actions[action]
        # This ensures that this can be an MO
        price = -50 
        price = np.round((1 + price / 10000) * self.data.obtain_level('askPrice', 1) * 100) / 100
        quantity = self.discrete_actions[action] / (len(self.discrete_actions) - 1) * self.total_quantity
        return self.env.step(dict(price=price, quantity=quantity))


class DiscretePriceWrapper(DiscreteActionBaseWrapper):
    """
    The quantity is fixed and equals to total_quantity 
    """
    def __init__(self, env):
        super(DiscretePriceWrapper, self).__init__(env)
        self.discrete_actions = self.config.simulation_discrete_actions
        self.num_shares = self.config.simulation_num_shares

    def step(self, action):
        price = self.discrete_actions[action]
        price = np.round((1 + price / 10000) * self.data.obtain_level('askPrice', 1) * 100) / 100
        quantity = self.total_quantity / self.num_shares
        return self.env.step(dict(price=price, quantity=quantity))


class ContinuousActionWrapper(BaseWrapper):
    def __init__(self, env):
        super(ContinuousActionWrapper, self).__init__(env)
        self.fixed_quantity_ratio = self.config.simulation_volume_ratio / self.config.simulation_num_shares
        self.num_shares = self.config.simulation_num_shares

    def step(self, action):
        price = self.continuous_action_scale * action
        price = np.round((1 + price / 10000) * self.data.obtain_level('askPrice', 1) * 100) / 100
        quantity = self.total_quantity / self.num_shares
        return self.env.step(dict(price=price, quantity=quantity))


def make_env(config):
    if config.simulation_action_type == 'discrete_p':
        return DiscretePriceWrapper(ExecutionEnv(config))
    elif config.simulation_action_type == 'continuous':
        return ContinuousActionWrapper(ExecutionEnv(config))
    elif config.simulation_action_type == 'discrete_pq':
        return DiscretePriceQuantityWrapper(ExecutionEnv(config))
    elif config.simulation_action_type == 'discrete_q_ning':
        return DiscreteQuantityNingWrapper(ExecutionEnv(config))
    elif config.simulation_action_type == 'discrete_q':
        return DiscreteQuantityWrapper(ExecutionEnv(config))
    
def run_env_test():

    config = DefaultConfig()
    config.simulation_action_type = "discrete_p"
    env = make_env(config)
    
    

    market_state, private_state = env.reset()
    print('market_state_shape = {}'.format(np.shape(market_state)))
    print('private_state = {}'.format(private_state))
    print('price: {}'.format(env.data.obtain_level('max_last_price')))
    print('snapshot = ')
    print(env.data.backtest_data.loc[env.data.current_index])

    market_state, private_state, reward, done, info = env.step(16)
    print('market_state_shape = {}'.format(np.shape(market_state)))
    print('private_state = {}'.format(private_state))
    print('price: {}'.format(env.data.obtain_level('max_last_price')))
    print('reward = {}'.format(reward))
    print('done = {}'.format(done))
    print('info = {}'.format(info)) 
    
    # print('snapshot = ')
    # print(env.data.backtest_data.loc[env.data.current_index])
    

    market_state, private_state, reward, done, info = env.step(17)
    market_state, private_state, reward, done, info = env.step(18)
    market_state, private_state, reward, done, info = env.step(19)
    market_state, private_state, reward, done, info = env.step(20)
    print('market_state_shape = {}'.format(np.shape(market_state)))
    print('private_state = {}'.format(private_state))
    print('price: {}'.format(env.data.obtain_level('max_last_price')))
    print('reward = {}'.format(reward))
    print('done = {}'.format(done))
    print('info = {}'.format(info))

    print("Total Index: {}".format(env.data.data.shape[0]))

if __name__ == "__main__":
    run_env_test()