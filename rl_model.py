from math import inf
from re import S
import torch
import random
import pandas as pd
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import osqp
import numpy as np
import scipy.sparse as sp

class energy_env(gym.Env):
    def __init__(self, params):
        super(energy_env, self).__init__()

        # Initialize environment parameters
        self.current_hour = 0
        self.min_demand = params['min_load']
        self.max_demand = params['max_load']
        self.min_price = params['min_price']
        self.max_price = params['max_price']
        self.demand = params['demand_forecast']
        self.price = params['price_forecast']
        
        self.current_imbalance = np.zeros((24,), dtype=np.float32)
        self.total_cost = np.zeros((24,), dtype=np.float32)
        self.generation = np.zeros((24, 4), dtype=np.float32)  # Generation amounts for solar, wind, battery, conv for each hour
        self.imbalance_ratio = np.zeros((24,), dtype=np.float32)
        self.best_bound_ratio = np.zeros((24,), dtype=np.float32)
        self.prev_imbalance_ratio = np.zeros((24,), dtype=np.float32)
        self.imbal_vio = np.zeros((24,), dtype=np.float32)

        self.perturbed_demand = np.copy(self.demand)
        self.perturbed_price = np.copy(self.price)
        self.max_solar = params['max_solar']
        self.solar_profile = params['solar_profile']
        self.max_wind = params['max_wind']
        self.wind_profile = params['wind_profile']
        self.battery_capacity = params['battery_capacity']
        self.conv_profile = params['conv_profile']
        self.battery_bal = np.zeros((24,), dtype=np.float32)  # Battery balance for each hour
        
        self.best_bounds = self.compute_best_bounds() # Pre-compute best bounds for all hours
       
        self.current_step = 0
        self.max_steps = 24
        
        self.imbalance_rel_gap = 0.25
        self.best_bound_rel_gap = 0.25

        # Action Space: Make action space continuous for each agent
        self.action_space = spaces.Box(
                            low=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),   # [solar_gen, wind_gen, battery, conv_gen]
                            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                            shape=(4,),
                            dtype=np.float32)
        
        # Observation space: 1 hour cyclic representation, solar,  wind, current imbalance, battery SoC, current demand, current price
        obs_low = np.concatenate([np.array([-1.0], dtype=np.float32), # 1 hour cyclic representation
                                  np.array([-1.0], dtype=np.float32), # 1 hour cyclic representation
                                  np.array([0.0], dtype=np.float32),  # solar
                                  np.array([0.0], dtype=np.float32),  # wind
                                  np.array([0.0], dtype=np.float32), #imbalance
                                  np.array([0.0], dtype=np.float32), # battery SoC
                                  np.array([0.0], dtype=np.float32), # demand
                                  np.array([0.0], dtype=np.float32), # price
                              ])
        
        max_demand = self.max_demand + self.battery_capacity   # Total demand including battery capacity
        obs_high = np.concatenate([np.array([1.0], dtype=np.float32), # 1 hour cyclic representation
                                   np.array([1.0], dtype=np.float32), # 1 hour cyclic representation
                                   np.array([self.max_solar], dtype=np.float32), # solar
                                   np.array([self.max_wind], dtype=np.float32), # wind
                                   np.array([max_demand], dtype=np.float32), #imbalance
                                   np.array([self.battery_capacity], dtype=np.float32), # battery SoC
                                   np.array([max_demand], dtype=np.float32), # demand
                                   np.array([self.max_price], dtype=np.float32), # price
                              ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(8,), dtype=np.float32)
        
        self.done = False
        self.penalty = params['max_price']  # Penalty

        # Set the parameters for the experiment
        self.reward_mode = params['reward_mode']
        self.use_curriculum = params['use_curriculum']

        # Set the weights for the reward function
        self.imbal_weight = params['imbal_weight']
        self.cost_weight = params['cost_weight']
    
    def set_rel_gap(self, imbalance_rel_gap, best_bound_rel_gap):
        self.imbalance_rel_gap = imbalance_rel_gap
        self.best_bound_rel_gap = best_bound_rel_gap

    def initialize(self, imbalance_rel_gap, best_bound_rel_gap):
        self.set_rel_gap(imbalance_rel_gap, best_bound_rel_gap)
        self.best_bounds = self.compute_best_bounds()
        self.imbal_vio = np.zeros((24,), dtype=np.float32)

    def seed(self, seed=None):
        """Set the seed for numpy and random global state."""
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    # For realistic simulation, check https://www.ercot.com/gridmktinfo/dashboards/
    def reset(self):
        self.current_hour = 0
        self.current_imbalance = np.zeros((24,), dtype=np.float32)
        self.total_cost = np.zeros((24,), dtype=np.float32)
        self.generation = np.zeros((24, 4), dtype=np.float32)  # Generation amounts for solar, wind, battery, conv for each hour
        self.imbalance_ratio = np.zeros((24,), dtype=np.float32)
        self.best_bound_ratio = np.zeros((24,), dtype=np.float32)
        self.done = False

        # Initialize current step for each new episode
        self.current_step = 0

        # Reset battery balance for the new episode
        self.battery_bal = np.zeros_like(self.battery_bal)

        # Perturb demand and price forecasts with some noise to simulate real-world uncertainty, 
        # and clip them to stay within the min and max bounds
        demand_noise = np.random.normal(0, 500, size=self.demand.shape)  # std dev = 500 MW
        price_noise = np.random.normal(0, 1, size=self.price.shape)  # std dev = 1 $/MWh
        self.perturbed_demand = np.clip(self.demand + demand_noise, self.min_demand, self.max_demand)
        self.perturbed_price = np.clip(self.price + price_noise, self.min_price, self.max_price)

        self.best_bounds = self.compute_best_bounds() # Pre-compute best bounds for all hours

        return self.build_obs()

    def build_obs(self):
        h = self.current_hour
        # Current hour cyclic representation
        hour_cos_t = np.cos(2 * np.pi * h / 24.0)  # Cosine representation of the hour
        hour_sin_t = np.sin(2 * np.pi * h / 24.0)  # Sine representation of the hour
        available_solar = self.solar_profile[h]  # Solar generation for the current hour
        available_wind = self.wind_profile[h]    # Wind generation for the current hour
        imbalance_t = self.current_imbalance[h]
        soc_t = self.battery_bal[h - 1] if h >= 1 else 0.0  # Use the pre-calculated best bound for the hour
        demand_t = self.perturbed_demand[h]  # Demand forecast for the current hour
        price_t = self.perturbed_price[h]  # Price forecast for the current hour
        obs = np.concatenate([np.array([hour_cos_t], dtype=np.float32),
                              np.array([hour_sin_t], dtype=np.float32),
                              np.array([available_solar], dtype=np.float32),
                              np.array([available_wind], dtype=np.float32),
                              np.array([imbalance_t], dtype=np.float32),
                              np.array([soc_t], dtype=np.float32),
                              np.array([demand_t], dtype=np.float32),
                              np.array([price_t], dtype=np.float32),
                              ])
        return obs

    def step(self, action):
        # Update the current step
        self.current_step += 1

        # Compute reward
        reward = self.compute_reward(action)

        # Check if the episode should end. Each episode can last for a maximum of 24 steps (hours).
        self.done = self.current_step >= self.max_steps

        h = self.current_hour
        obs = self.build_obs()
        info = {'Imbalance Ratio':self.imbalance_ratio[h], 
                'Best Bound Ratio': self.best_bound_ratio[h], 
                'Total Cost': self.total_cost[h], 
                'Best Bound':self.best_bounds[h], 
                'Settlement Price':self.perturbed_price, 
                'Generation Amounts':self.generation[h]}
        
        self.current_hour += 1  # Move to the next hour for the next step

        # Return the state, reward, done flag, and additional info
        return obs, reward, self.done, info

    def compute_reward(self, action):
        # Current hour
        h = self.current_hour

        # Unpack the action
        solar_usage, wind_usage, battery_usage, conv_usage = action  # Normalized [0, 1]
        
        # Calculate the generation
        solar_gen = solar_usage * self.solar_profile[h]
        wind_gen = wind_usage * self.wind_profile[h]
        battery_gen = battery_usage * self.battery_capacity  # Can be negative for discharge
        conv_gen = conv_usage * self.conv_profile[h]

        supply = solar_gen + wind_gen + conv_gen
        demand = self.perturbed_demand[h]

        # Clip the battery action so that it is within the battery capacity limits
        battery_violation = 0.0
        start_battery_bal = self.battery_bal[h-1] if h >= 1 else 0.0  # Previous hour's battery balance
        if (battery_gen < 0.0):
            # Discharging the battery below min balance (0.0)
            if (start_battery_bal + battery_gen < 0.0):
                battery_violation = abs(start_battery_bal + battery_gen)
            # Clip the battery action to the battery balance limits
            battery_gen = max(battery_gen, -start_battery_bal)  # Cannot discharge more than the current battery balance
        elif (battery_gen > 0.0):
            # Charging the battery more than the available generation.
            if (battery_gen > supply):
                battery_violation = (battery_gen - supply)
            
            # Charging the battery over the max capacity
            if (start_battery_bal + battery_gen > self.battery_capacity):
                battery_violation += (start_battery_bal + battery_gen - self.battery_capacity)
            
            # Clip the battery action to stay within the available generation limits
            battery_gen = min(battery_gen, supply)
            
            # Clip the battery action to stay within the battery capacity limits
            battery_gen = min(battery_gen, self.battery_capacity - start_battery_bal)  # Cannot charge more than the maximum capacity

        if (battery_gen < 0.0):
            supply += abs(battery_gen)
        elif (battery_gen > 0.0):
            demand += battery_gen

        # Update the battery balance for the current hour
        self.battery_bal[h] = start_battery_bal + battery_gen

        # Current imbalance
        self.current_imbalance[h] = abs(demand - supply)

        # In reward calculation, check for invalid or harmful actions first
        reward = 0.0
        if (battery_violation > 0.0):
            # Invalid action. So return a large negative reward
            reward =  -(battery_violation / self.battery_capacity) * 1000.0

        # Give a huge negative reward if no available capacity but action is taken
        if (solar_usage > 0.0 and self.solar_profile[h] <= 0.0):
            reward += -1000.0

        if (wind_usage > 0.0 and self.wind_profile[h] <= 0.0):
            reward += -1000.0

        # Calculate the generation and battery operation costs
        solar_cost = 0.0
        wind_cost = 0.0

        battery_cost = 0.0
        if (battery_gen < 0.0):
            battery_cost = abs(battery_gen) * min(self.perturbed_price)  # Battery discharge cost

        conv_cost = conv_gen * self.perturbed_price[h]
        total_cost = solar_cost + wind_cost + battery_cost + conv_cost

        # Check if the imbalance and cost performance meets the criteria for success 
        # and provide additional reward
        met_imbal_flag, imbalance_ratio = self.check_if_met_imbalance_criterion(supply, demand)
        best_bound = self.best_bounds[h]
        met_best_bound_flag, best_bound_ratio = self.check_if_met_best_bound_criterion(total_cost, best_bound)
        
        if (imbalance_ratio >= 0.0):
            if (self.reward_mode == 1):
                reward += self.calc_gated_reward(total_cost, supply, imbalance_ratio, met_imbal_flag, h)
            else:
                reward += self.calc_weighted_reward(total_cost, supply, imbalance_ratio, h)
        else:
            # Penalize heavily for over-supply
            reward += -self.imbal_weight

        # Save the outputs. These can be used for analysis and plotting after the episode ends
        self.imbalance_ratio[h] = imbalance_ratio
        self.best_bound_ratio[h] = best_bound_ratio
        self.total_cost[h] = total_cost
        self.generation[h] = [solar_gen, wind_gen, battery_gen, conv_gen]
        self.prev_imbalance_ratio[h] = imbalance_ratio

        return reward

    def calc_gated_reward(self, total_cost, supply, imbalance_ratio, met_imbal_flag, h):
        gated_reward = 0.0

        if (imbalance_ratio > self.imbalance_rel_gap):
            gated_reward += -self.imbal_weight * (imbalance_ratio - self.imbalance_rel_gap)
        else:
            gated_reward += 0.01 * self.imbal_weight * (self.imbalance_rel_gap - imbalance_ratio)

        if met_imbal_flag:
            avg_best_price = self.perturbed_price[h]
            avg_price = total_cost / supply if supply > 0 else self.max_price
            avg_price_ratio = (avg_price - avg_best_price) / avg_best_price
            gated_reward += -self.cost_weight * (avg_price_ratio)
        else:
            # Only apply progress penalty if there is a prior episode to compare against
            progress = (self.prev_imbalance_ratio[h] - imbalance_ratio)
            if progress <= 0.01:
                vio_count = int(self.imbal_vio[h]) + 1
                gated_reward += -0.01 * self.imbal_weight * vio_count
                self.imbal_vio[h] = vio_count

        return gated_reward

    def calc_weighted_reward(self, total_cost, supply, imbalance_ratio, h):
        weighted_reward = 0.0

        # Imbalance signal: always present
        weighted_reward += -self.imbal_weight * (imbalance_ratio - self.imbalance_rel_gap)

        # Cost signal: always present simultaneously
        avg_best_price = self.perturbed_price[h]
        avg_price = total_cost / supply if supply > 0 else self.max_price
        avg_price_ratio = (avg_price - avg_best_price) / avg_best_price
        weighted_reward += -self.cost_weight * avg_price_ratio

        return weighted_reward

    def calc_arbitrage_bonus(self, action):
        # Calculate the arbitrage bonus based on future prices and current action
        arbitrage_bonus = 0.0
        h = self.current_hour
        future_window = 6
        current_price = self.perturbed_price[h]
        current_demand = self.perturbed_demand[h]
        future_price_vec = self.perturbed_price[h+1:h+1+future_window]
        future_demand_vec = self.perturbed_demand[h+1:h+1+future_window]

        bonus = 0.0
        max_bonus = 0.0
        for i in range(len(future_price_vec)):
            # If future prices are not available, skip
            if future_price_vec[i] <= 0 or future_demand_vec[i] <= 0:
                continue
            # Find the price and demand in the lookahead window
            future_price = future_price_vec[i]
            future_demand = future_demand_vec[i]

            # If current price is low and future price is high, reward charging
            if current_price < future_price and action > 0:  # battery_usage > 0 (charging)
                bonus = (future_price - current_price) * min(action, future_demand)
            # If current price is high and future price is low, reward discharging
            elif current_price > future_price and action < 0:  # battery_usage < 0 (discharging)
                bonus = (current_price - future_price) * min(abs(action), current_demand)
            if bonus > max_bonus:
                max_bonus = bonus
        return max_bonus

    def check_if_met_best_bound_criterion(self, total_cost, best_bound):
        denom = best_bound
        rel_gap = self.best_bound_rel_gap
        best_bound_ratio = 1.0
        if denom > 0:
            best_bound_ratio = (total_cost - best_bound) / denom
            if best_bound_ratio <= rel_gap:
                return True, best_bound_ratio
        return False, best_bound_ratio

    def check_if_met_imbalance_criterion(self, supply, demand):
        # Calculate the raw supply and demand for the current hour
        h = self.current_hour
        
        denom = demand
        rel_gap = self.imbalance_rel_gap
        imbalance_ratio = 1.0
        if denom > 0:
            imbalance_ratio = (demand - supply) / denom
            if imbalance_ratio <= rel_gap:
                return True, imbalance_ratio
        return False, imbalance_ratio

    def compute_best_bounds(self):
        # Solve the optimization problem to find the best bound for each hour and set it as the initial best bound for the environment
        n = 24*6  # Number of action variables for 24 hours with 6 resources with 3 variables reserved for battery charge/discharge/SoC
        P = sp.csc_matrix((n, n))  # Quadratic Cost is 0.0
        q = np.zeros(n)  # Linear cost for solar, wind, battery, conv
        
        constraints = []
        lb = []
        ub = []
        i = 0
        min_price = min(self.perturbed_price)
        for h in range(24):
            q[i] = 0.0  # Solar cost
            q[i+1] = 0.0  # Wind cost
            q[i+2] = 0.0  # Battery charge cost
            q[i+3] = min_price  # Battery discharge cost
            q[i+4] = self.perturbed_price[h]  # Conv cost
            q[i+5] = 0.0  # Battery SoC variable has no cost

            bound_row_solar = np.zeros(n)
            bound_row_solar[i] = 1.0
            constraints.append(bound_row_solar)
            lb.append(0.0)
            ub.append(self.solar_profile[h])

            bound_row_wind = np.zeros(n)
            bound_row_wind[i+1] = 1.0
            constraints.append(bound_row_wind)
            lb.append(0.0)
            ub.append(self.wind_profile[h])

            bound_row_battery_charge = np.zeros(n)
            bound_row_battery_charge[i+2] = 1.0
            constraints.append(bound_row_battery_charge)
            lb.append(0.0)
            ub.append(self.battery_capacity)

            bound_row_battery_discharge = np.zeros(n)
            bound_row_battery_discharge[i+3] = 1.0
            constraints.append(bound_row_battery_discharge)
            lb.append(0.0)
            ub.append(self.battery_capacity)

            bound_row_conv = np.zeros(n)
            bound_row_conv[i+4] = 1.0
            constraints.append(bound_row_conv)
            lb.append(0.0)
            ub.append(self.conv_profile[h])

            bound_row_battery_soc = np.zeros(n)
            bound_row_battery_soc[i+5] = 1.0
            constraints.append(bound_row_battery_soc)
            lb.append(0.0)
            ub.append(self.battery_capacity)
            
            # Add energy balance constraints for each hour
            energy_balance_row = np.zeros(n)
            energy_balance_row[i] = 1.0  # Solar
            energy_balance_row[i+1] = 1.0  # Wind
            energy_balance_row[i+2] = -1.0  # Battery charge
            energy_balance_row[i+3] = 1.0  # Battery discharge
            energy_balance_row[i+4] = 1.0  # Conv
            lb.append(self.perturbed_demand[h])
            ub.append(self.perturbed_demand[h])
            constraints.append(energy_balance_row)

            # Add battery balance constraints for each hour
            # Initial battery balance constraint: battery SoC at hour 0 is 0
            battery_balance_row = np.zeros(n)
            if h >= 1:
                battery_balance_row[i-1] = 1.0  # Previous hour battery SoC
            battery_balance_row[i+2] = 1.0  # Current hour battery charge
            battery_balance_row[i+3] = -1.0  # Current hour battery discharge
            battery_balance_row[i+5] = -1.0  # Current hour battery SoC
            lb.append(0.0)
            ub.append(0.0)
            constraints.append(battery_balance_row)

            i += 6  # Move to the next hour's variables

        # Solve the QP to find the best dispatch for each hour
        A = sp.csc_matrix(constraints)
        l = np.array(lb)
        u = np.array(ub)

        prob = osqp.OSQP()
        prob.setup(
            P=P, q=q, A=A, l=l, u=u,
            verbose=False, # Critical: no printing
            eps_abs=1e-8, # tight tolerance
            eps_rel=1e-8,  
            max_iter=2000, # Maximum number of iterations
            polish=False, # Do not perform polishing
            warm_start=True, # Start from the previous solution (if available
        )
        sol = prob.solve()
        
        # Calculate the best bound for each hour based on the optimal dispatch
        i = 0
        best_bounds = []
        for h in range(24):
            best_bound = q[i] * sol.x[i] + q[i+1] * sol.x[i+1] + q[i+2] * sol.x[i+2] + q[i+3] * sol.x[i+3] + q[i+4] * sol.x[i+4]
            best_bounds.append(best_bound)
            i += 6

        return best_bounds

class safe_energy_env(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # Optimize action using QP or other methods to ensure safety
        modified_action = self.optimize_action(action)

        # Calculate penalty for unsafe actions
        deviation = np.linalg.norm(action - modified_action)
        penalty = -10 * deviation**2 if deviation > 0.01 else 0
        
        # Call the parent step function
        obs, reward, done, info = self.env.step(modified_action)
        reward += penalty
    
        return obs, reward, done, info

    def optimize_action(self,  action):
        # Set objective function for QP
        n = 4  # Number of action variables
        P = sp.diags([2.0, 2.0, 2.0, 2.0], format='csc')  # Quadratic cost for deviation from original action   
        q = -2.0 * np.array(action)
       
        h = self.env.current_hour

        # Contraction Factor: Imbalance/total cost must shrink to at least 95% of previous value
        rho = 0.95
        
        demand = self.env.perturbed_demand[h]
        battery_bal = self.env.battery_bal[h-1] if h >=1 else 0.0

        solar_profile = self.env.solar_profile[h]
        wind_profile = self.env.wind_profile[h]
        battery_cap = self.env.battery_capacity
        conv_profile = self.env.conv_profile[h]
        best_bound = self.env.best_bounds[h]

        constraints = []
        l_bounds = []
        u_bounds = []

        # Safety constraint to ensure that the supply meets the demand: solar + wind + battery + conv = demand
        lp_row = [solar_profile, wind_profile, battery_cap, conv_profile]
        lp_row_lb = demand
        lp_row_ub = demand
        constraints.append(lp_row)
        l_bounds.append(lp_row_lb)
        u_bounds.append(lp_row_ub)

        # Add the battery limits as CBF constraints for safe battery operation
        # 0 <= previous balance + battery charge /discharge <= battery_capacity
        bl_row = [0, 0, battery_cap, 0.0]
        bl_row_lb = 0.0 - battery_bal
        bl_row_ub = battery_cap - battery_bal
        constraints.append(bl_row)
        l_bounds.append(bl_row_lb)
        u_bounds.append(bl_row_ub)

        # Add the constraint to ensure solution stays as close to best bound as possible
        # min_price = min(self.env.perturbed_price[h])
        # if (best_bound > 0.0):
        #     cd_row_1 = [0.0, 0.0, 0.0, conv_profile * self.env.perturbed_price[h]]
        #     cd_row_1_lb = 0.0
        #     cd_row_1_ub = rho * best_bound
        #     constraints.append(cd_row_1)
        #     l_bounds.append(cd_row_1_lb)
        #     u_bounds.append(cd_row_1_ub)

        #     cd_row_2 = [0.0, 0.0, -battery_cap * self.env.min_price, conv_profile * self.env.perturbed_price[h]]
        #     cd_row_2_lb = 0.0
        #     cd_row_2_ub = rho * best_bound
        #     constraints.append(cd_row_2)
        #     l_bounds.append(cd_row_2_lb)
        #     u_bounds.append(cd_row_2_ub)

        # Set the variable bounds for the modified action variables
        bound_row_solar = [1.0, 0.0, 0.0, 0.0]
        lb_solar = 0.0
        ub_solar = 1.0
        constraints.append(bound_row_solar)
        l_bounds.append(lb_solar)
        u_bounds.append(ub_solar)

        bound_row_wind = [0.0, 1.0, 0.0, 0.0]
        lb_wind = 0.0
        ub_wind = 1.0
        constraints.append(bound_row_wind)
        l_bounds.append(lb_wind)
        u_bounds.append(ub_wind)

        bound_row_battery = [0.0, 0.0, 1.0, 0.0]
        lb_battery = -1.0
        ub_battery = 1.0
        constraints.append(bound_row_battery)
        l_bounds.append(lb_battery)
        u_bounds.append(ub_battery)

        bound_row_conv = [0.0, 0.0, 0.0, 1.0]
        lb_conv = 0.0
        ub_conv = 1.0
        constraints.append(bound_row_conv)
        l_bounds.append(lb_conv)
        u_bounds.append(ub_conv)

        A = sp.csc_matrix(constraints)
        l = np.array(l_bounds)
        u = np.array(u_bounds)

        prob = osqp.OSQP()
        prob.setup(
            P=P, q=q, A=A, l=l, u=u,
            verbose=False, # Critical: no printing
            eps_abs=1e-3, # Slightly loose tolerance
            eps_rel=1e-3,  
            max_iter=2000, # Maximum number of iterations
            polish=False, # Do not perform polishing
            warm_start=True, # Start from the previous solution (if available
        )

        sol = prob.solve()

        if sol.info.status != 'solved':
            print(f"QP not solved. Status: {sol.info.status}. Returning original action.")
            # If the optimization problem is not solved, return the original action
            return np.clip(action, [0.0, 0.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        return (sol.x[0:n])

class rl_model:
    def __init__(self, initial_seed=42, reward_mode=1, use_curriculum=1, imbal_weight=1000.0, cost_weight=10.0):
        # Fill the demand and price forecasts with some dummy data
        parms = self.set_parms(reward_mode, use_curriculum, imbal_weight, cost_weight)
        # Create the environment by wrapping the energy environment
        env = lambda: energy_env(parms)
        env = DummyVecEnv([env])
        env.seed(initial_seed)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        env.seed(initial_seed)
        self.env = env
        self.env.reset()
        self.policy = PPO("MlpPolicy", self.env, verbose=1, seed=initial_seed, device='auto')
        print(torch.__version__)

    def set_parms(self, reward_mode, use_curriculum, imbal_weight, cost_weight):
        params ={'min_load': 50803,  # Minimum load in MW,'
                'max_load': 74391,  # Maximum load in MW,
                'min_price': 14.43,  # Minimum price in $/MWh,
                'max_price': 65.44,    # Maximum price in $/MWh,
                'max_solar': 24748,   # Maximum solar generation in MW,
                'max_wind': 19734,    # Maximum wind generation in MW,
                'battery_capacity': 4000, # Battery capacity in MW,
                'max_conv': 50215, # Maximum conventional generation in MW,
                'demand_forecast': np.zeros(48, dtype=np.float32), # 48 hours demand forecast
                'price_forecast': np.zeros(48, dtype=np.float32), # 48 hours price forecast
                }

        # Read ERCOT data 
        df = pd.read_csv('data/ERCOT Data.csv')
        params['demand_forecast'] = df['Demand'].values
        params['price_forecast'] = df['Price'].values
        params['solar_profile'] = df['Solar'].values[0:24]
        params['wind_profile'] = df['Wind'].values[0:24]
        max_conv = df['Capacity'].values - (df['Solar'].values + df['Wind'].values)
        params['conv_profile'] = max_conv[0:24]

        # Set parameters for the experiment
        params['reward_mode'] = reward_mode
        params['use_curriculum'] = use_curriculum
        params['imbal_weight'] = imbal_weight   
        params['cost_weight'] = cost_weight

        return params

    def train(self):
        if self.env.envs[0].use_curriculum == 1:
            curriculum = [
                (0.80, 0.0, 120000),
                (0.70, 0.0, 120000),
                (0.60, 0.0, 120000),
                (0.50, 0.0, 120000),
                (0.40, 0.0, 150000),
                (0.30, 0.0, 150000),
                (0.20, 0.0, 200000),
                (0.10, 0.0, 150000),
                (0.05, 0.0, 120000)
            ]
            for level, (imbal_rel_gap, best_bound_rel_gap, timesteps) in enumerate(curriculum):
                self.env.envs[0].set_rel_gap(imbal_rel_gap, best_bound_rel_gap)
                self.env.envs[0].prev_imbalance_ratio = np.full((24,), imbal_rel_gap, dtype=np.float32)
                self.env.envs[0].imbal_vio = np.zeros((24,), dtype=np.float32)
                self.policy.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        else:
            self.env.envs[0].set_rel_gap(0.05, 0.0)
            self.env.envs[0].prev_imbalance_ratio = np.full((24,), 0.05, dtype=np.float32)
            self.env.envs[0].imbal_vio = np.zeros((24,), dtype=np.float32)
            self.policy.learn(total_timesteps=1250000)

    def predict(self, output, seed=None):
        imbalance_gap = []
        best_bound_gap = []
        battery_bal = []
        demand = []
        price = []
        actions = []
        rewards = []
        total_costs = []
        best_bounds = []
        settlement_prices = []
        time_steps = []
        gen_amts = []

        if (seed is not None):
            self.env.seed(seed)

        obs = self.env.reset()
        if (obs.ndim == 1):
            obs = obs.reshape((1, -1))

        # Initialize the environment with the tightest criteria for success to evaluate the 
        # trained policy under the most challenging conditions
        self.env.envs[0].initialize(0.02, 0.05)

        # Simulate for each hour of the day
        done = False
        for hour in range(24):
            time_steps.append(hour)
            action, hidden_state = self.policy.predict(obs)
            obs, reward, done, info = self.env.step(action)

            # Store the results
            actions.append(action)
            imbalance_gap.append(100*info[0]['Imbalance Ratio'])
            battery_bal.append(self.env.envs[0].battery_bal[hour])
            demand.append(self.env.envs[0].perturbed_demand[hour])
            price.append(self.env.envs[0].perturbed_price[hour])
            best_bound_gap.append(100*info[0]['Best Bound Ratio'])
            total_costs.append(info[0]['Total Cost'])
            best_bounds.append(info[0]['Best Bound'])
            settlement_prices.append(info[0]['Settlement Price'])
            gen_amts.append(info[0]['Generation Amounts'])
            rewards.append(reward)

        # Compute the renewable percentage as one number for the entire day.
        total_renewable_gen = sum([gen[0] + gen[1] for gen in gen_amts])
        total_renewable_available = sum([self.env.envs[0].solar_profile[h] + self.env.envs[0].wind_profile[h] for h in range(24)])
        reur = 100 * total_renewable_gen / total_renewable_available

        # Plot the results over steps for each hour
        if output:
            plt.figure(figsize=(3.5, 2.5))  # single-column IEEE size
            plt.plot(time_steps, imbalance_gap, marker='o')
            plt.title('Imbalance Gap')
            plt.xlabel('Hour')
            plt.ylabel('Imbalance Gap (%)')
            plt.tight_layout()
            plt.savefig(f"imbalance_gap.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close()

            plt.figure(figsize=(3.5, 2.5))
            plt.plot(time_steps, best_bound_gap, marker='o')
            plt.title('Best Bound Gap')
            plt.xlabel('Hour')
            plt.ylabel('Best Bound Gap (%)')
            plt.tight_layout()
            plt.savefig(f"best_bound_gap.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close()

            plt.figure(figsize=(3.5, 2.5))
            plt.plot(time_steps, self.env.envs[0].battery_bal, marker='o')
            plt.title('Battery Balance')
            plt.xlabel('Hour')
            plt.ylabel('Battery Balance (MW)')
            plt.tight_layout()
            plt.savefig(f"battery_balance.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close()

            plt.figure(figsize=(3.5, 2.5))
            plt.plot(time_steps, self.env.envs[0].perturbed_demand[0:24], marker='o')
            plt.title('Demand')
            plt.xlabel('Hour')
            plt.ylabel('Demand (MW)')
            plt.tight_layout()
            plt.savefig(f"demand.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close()

            plt.figure(figsize=(3.5, 2.5))
            plt.plot(time_steps, self.env.envs[0].perturbed_price[0:24], marker='o')
            plt.title('Price')
            plt.xlabel('Hour')
            plt.ylabel('Price ($/MWh)')
            plt.tight_layout()
            plt.savefig(f"price.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close()

            plt.figure(figsize=(3.5, 2.5))
            plt.plot(time_steps, total_costs, label='Total Cost', marker='o')
            plt.plot(time_steps, best_bounds, label='Best Bound', marker='x')
            plt.title('Total Cost vs Best Bound')
            plt.xlabel('Hour')
            plt.ylabel('Cost ($)')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(f"cost.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close()

            plt.figure(figsize=(3.5, 2.5))
            plt.plot(time_steps, self.env.envs[0].solar_profile, label='Solar', marker='o')
            plt.plot(time_steps, self.env.envs[0].wind_profile, label='Wind', marker='x')
            plt.title('Renewable Profiles')
            plt.xlabel('Hour')
            plt.ylabel('Capacity (MW)')
            plt.tight_layout()
            plt.legend(loc='best')
            plt.savefig(f"renewable_profile.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close()

            solar_gen = [gen_amt[0] for gen_amt in gen_amts]
            wind_gen = [gen_amt[1] for gen_amt in gen_amts]

            plt.figure(figsize=(3.5, 2.5))
            plt.plot(time_steps, solar_gen, label='Solar', marker='o')
            plt.plot(time_steps, wind_gen, label='Wind', marker='x')
            plt.title('Renewable Generation')
            plt.xlabel('Hour')
            plt.ylabel('Generation (MW)')
            plt.tight_layout()
            plt.legend(loc='best')
            plt.savefig(f"renewable_gen.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
            plt.close()

        # Set the output dictionary
        output_dict = {
            'Imbalance Gap': imbalance_gap,
            'Best Bound Gap': best_bound_gap,
            'Battery Balance': battery_bal,
            'Demand': demand,
            'Price': price,
            'Total Costs': total_costs,
            'Best Bounds': best_bounds,
            'Solar Generation': [gen_amt[0] for gen_amt in gen_amts],
            'Wind Generation': [gen_amt[1] for gen_amt in gen_amts],
            'Settlement Prices': settlement_prices,
            'Generation Amounts': gen_amts,
            'Renewable Utilization Ratio': reur
        }
        return output_dict
