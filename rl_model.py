from pickle import FALSE
import random
from winreg import REG_NOTIFY_CHANGE_LAST_SET
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# TO DO: Add more realistic data for offers and bids
# TO DO: Add designated agents: solar, wind, battery, traditional and 1 consumer agent.
# Each agent takes action for each hour based on the available capacity to reduce the current future imbalance at the lowest cost.
# TO DO: Add more realistic reward function based on future arbitrage opportunities
# TO DO: solar and wind are free but limited, battery is expensive but can be used to store energy for future arbitrage opportunities
# TO DO: Traditional generators are expensive and can be used to balance the grid in case of imbalance
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
        self.perturbed_demand = np.copy(self.demand)
        self.perturbed_price = np.copy(self.price)
        self.current_imbalance = 0.0
        self.max_solar = params['max_solar']
        self.solar_profile = params['solar_profile']
        self.max_wind = params['max_wind']
        self.wind_profile = params['wind_profile']
        self.battery_capacity = params['battery_capacity']
        self.max_coal = params['max_coal']
        self.battery_bal = np.zeros((24,), dtype=np.float32)  # Battery balance for each hour
        self.best_bound = 0.0
        self.prev_imbalance = 0.0
        self.prev_total_cost = 0.0

        # Action Space: Make action space continuous for each agent
        self.action_space = spaces.Box(
                            low=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),   # [solar_gen, wind_gen, battery, coal_gen]
                            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                            shape=(4,),
                            dtype=np.float32)
        
        # Observation space: 1 hour cyclic representation + solar + wind + current imbalance+ best_bound + current demand + current price + 6 hr demand forecast + 6 hr price forecast
        obs_low = np.concatenate([np.array([-1.0], dtype=np.float32),
                              np.array([-1.0], dtype=np.float32),
                              np.array([0.0], dtype=np.float32),
                              np.array([0.0], dtype=np.float32),
                              np.array([self.min_demand], dtype=np.float32),
                              np.array([0.0], dtype=np.float32),
                              np.array([self.min_demand], dtype=np.float32),
                              np.array([0.0], dtype=np.float32),
                              np.ones(6, dtype=np.float32) * self.min_demand,
                              np.ones(6, dtype=np.float32) * self.min_price,
                              ])
        
        max_demand = params['max_load'] + params['battery_capacity']   # Total demand including battery capacity
        obs_high = np.concatenate([np.array([1.0], dtype=np.float32),
                                   np.array([1.0], dtype=np.float32),
                                   np.array([self.max_solar], dtype=np.float32),
                                   np.array([self.max_wind], dtype=np.float32),
                                   np.array([max_demand], dtype=np.float32),
                                   np.array([max_demand*self.max_price], dtype=np.float32),
                                   np.array([max_demand], dtype=np.float32),
                                   np.array([self.max_price], dtype=np.float32),
                                   np.ones(6, dtype=np.float32) * max_demand,
                                   np.ones(6, dtype=np.float32) * self.max_price,
                              ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(20,), dtype=np.float32)
        
        self.done = False
        self.penalty = params['max_price']  # Penalty
    
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    # Set the hour, imbalance and reset the battery balance for the new hour
    def set_state(self, h):
        self.current_hour = h
        self.current_imbalance = 0.0
        # Reset battery balance for the new hour as the battery balance is carried over from the previous hour
        self.battery_bal[h] = self.battery_bal[h-1] if h >= 1 else 0.0

    # For realistic simulation, check https://www.ercot.com/gridmktinfo/dashboards/
    # TO DO: slightly perturb the hour, demand and price forecasts to simulate realistic scenarios
    def reset(self):
        self.current_hour = np.random.randint(0,24)  # Randomly select an hour to start the episode
        self.current_imbalance = 0.0
        self.best_bound = 0.0
        self.done = False
        demand_noise = np.random.normal(0, 2, size=self.demand.shape)  # std dev = 2 MW
        price_noise = np.random.normal(0, 0.1, size=self.price.shape)  # std dev = 0.1 $/MWh
        self.perturbed_demand = self.demand + demand_noise
        self.perturbed_price = self.price + price_noise

        # Clip to min/max bounds
        self.perturbed_demand = np.clip(self.perturbed_demand, self.min_demand, self.max_demand)
        self.perturbed_price = np.clip(self.perturbed_price, self.min_price, self.max_price)
        
        # Set prev variables
        self.prev_imbalance = 0.0
        self.prev_total_cost = 0.0
        return self.build_obs()

    def build_obs(self):
        h = self.current_hour
        # Current hour cyclic representation
        hour_cos_t = np.cos(2 * np.pi * h / 24.0)  # Cosine representation of the hour
        hour_sin_t = np.sin(2 * np.pi * h / 24.0)  # Sine representation of the hour
        available_solar = self.solar_profile[h]  # Solar generation for the current hour
        available_wind = self.wind_profile[h]    # Wind generation for the current hour
        imbalance_t = self.current_imbalance
        best_bound_t = self.best_bound
        demand_t = self.perturbed_demand[h]  # Demand forecast for the current hour
        price_t = self.perturbed_price[h]  # Price forecast for the current hour
        demand_forecast = self.demand[h:h+6]  # Future demand forecast from current hour to end of day
        price_forecast = self.price[h:h+6]    # Future price forecast from current hour to end of day
        obs = np.concatenate([np.array([hour_cos_t], dtype=np.float32),
                              np.array([hour_sin_t], dtype=np.float32),
                              np.array([available_solar], dtype=np.float32),
                              np.array([available_wind], dtype=np.float32),
                              np.array([imbalance_t], dtype=np.float32),
                              np.array([best_bound_t], dtype=np.float32),
                              np.array([demand_t], dtype=np.float32),
                              np.array([price_t], dtype=np.float32),
                              np.array(demand_forecast, dtype=np.float32),
                              np.array(price_forecast, dtype=np.float32),
                              ])
        return obs

    def step(self, action):
        reward, total_cost = self.compute_reward(action)
        
        # Return the state, reward, done flag, and additional info
        return self.build_obs(), reward, self.done, {'hour': self.current_hour, 'Imbalance':self.current_imbalance, 'Total Cost': total_cost, 'Best Bound':self.best_bound}
    
    def compute_reward(self, action):
        # Current hour
        h = self.current_hour

        # Unpack the action
        solar_usage, wind_usage, battery_usage, coal_usage = action  # Normalized [0, 1]
        
        # Calculate the generation
        solar_gen = solar_usage * self.solar_profile[h]
        wind_gen = wind_usage * self.wind_profile[h]
        battery_action = battery_usage * self.battery_capacity  # Can be negative for discharge
        coal_gen = coal_usage * self.max_coal

        supply = solar_gen + wind_gen + coal_gen
        demand = self.perturbed_demand[h]

        # Clip the battery action so that it is within the battery capacity limits
        battery_violation = 0.0
        start_battery_bal = self.battery_bal[h-1] if h >= 1 else 0.0  # Previous hour's battery balance
        if (battery_action < 0.0):
            # Discharging the battery below min balance (0.0)
            if (start_battery_bal + battery_action < 0.0):
                battery_violation = abs(start_battery_bal + battery_action)
            # Clip the battery action to the battery balance limits
            battery_action = max(battery_action, -start_battery_bal)  # Cannot discharge more than the current battery balance
        elif (battery_action > 0.0):
            # Charging the battery more than the available generation.
            if (battery_action > supply):
                battery_violation = (battery_action - supply)
            
            # Charging the battery over the max capacity
            if (start_battery_bal + battery_action > self.battery_capacity):
                battery_violation += (start_battery_bal + battery_action - self.battery_capacity)
            
            # Clip the battery action to stay within the available generation limits
            battery_action = min(battery_action, supply)
            
            # Clip the battery action to stay within the battery capacity limits
            battery_action = min(battery_action, self.battery_capacity - start_battery_bal)  # Cannot charge more than the maximum capacity

        if (battery_action < 0.0):
            supply += abs(battery_action)
        elif (battery_action > 0.0):
            demand += battery_action

        # Update the battery balance for the current hour
        self.battery_bal[h] = start_battery_bal + battery_action

        # Imbalance cost
        self.current_imbalance = abs(demand - supply)
        #imbalance_cost = -self.penalty * self.current_imbalance  # Cost of imbalance based on current price

        # In reward calculation, check for invalid or harmful actions first
        reward = 0.0
        if (battery_violation > 0.0):
            # Invalid action. So return a large negative reward
            reward =  -self.penalty * battery_violation * 1000

        # Give a huge negative reward if no available capacity but action is taken
        if (solar_usage > 0.0 and self.solar_profile[h] <= 0.0):
            reward += -self.penalty * solar_usage * 1000

        if (wind_usage > 0.0 and self.wind_profile[h] <= 0.0):
            reward += -self.penalty * wind_usage * 1000

        # Reward imbalance reduction and penalize imbalance increase
        reward += self.penalty * (self.prev_imbalance - self.current_imbalance)


        # if (imbalance_cost < 0.0):
        #     # Discourage imbalance
        #     reward += imbalance_cost

        # Calculate the generation and battery operation costs
        solar_cost = 0.0
        wind_cost = 0.0
        battery_cost = abs(battery_action) * 0.2
        coal_cost = coal_gen * self.perturbed_price[h]
        total_cost = solar_cost + wind_cost + battery_cost + coal_cost
        reward -= total_cost # cost of generation

        # Extra reward for cost performance if imbalance also reduced
        if (self.current_imbalance < self.prev_imbalance and total_cost < self.prev_total_cost):
            reward += self.penalty * (self.prev_total_cost - total_cost)

        # --- future arbitrage reward ---
        profit = self.calc_arbitrage_bonus(battery_action)
        reward += profit
        # --- end of future arbitrage reward ---

        self.best_bound = self.calc_best_bound(demand)
        self.best_bound += battery_cost

        # Check if the episode should end
        self.done = False
        if (self.check_if_met_imbalance(supply, demand)):
            self.done = self.check_if_met_best_bound(total_cost, self.best_bound)
        
        # Update previous imbalance and total cost
        self.prev_imbalance = self.current_imbalance
        self.prev_total_cost = total_cost

        return reward, total_cost

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
                bonus = (future_price - self.price[h]) * min(action, future_demand)
            # If current price is high and future price is low, reward discharging
            elif current_price > future_price and action < 0:  # battery_usage < 0 (discharging)
                bonus = (current_price - future_price) * min(abs(action), current_demand)
            if bonus > max_bonus:
                max_bonus = bonus
        return max_bonus

    def check_if_met_best_bound(self, total_cost, best_bound):
        if (total_cost <= best_bound or total_cost <= 1.0):
            return True

        denom = total_cost
        rel_gap = 0.20
        if denom > 0:
            best_bound_ratio = (total_cost - best_bound) / denom
            if best_bound_ratio <= rel_gap:
                return True
        return False

    def check_if_met_imbalance(self, supply, demand):
        # Calculate the raw supply and demand for the current hour
        h = self.current_hour
        max_supply = self.solar_profile[h] + self.wind_profile[h] + self.max_coal
        
        denom = min(demand, max_supply)
        rel_gap = 0.20
        if denom > 0:
            imbalance_ratio = self.current_imbalance / denom
            if imbalance_ratio <= rel_gap:
                return True
        return False

    # Merit order calculation for the best bound for the cost.
    # This is used to calculate the lowest possible cost to meet the demand
    def calc_best_bound(self, demand):
        # calculate the lowest possible cost to meet the demand
        h = self.current_hour
        price = self.perturbed_price[h]
        # Calculate the best bound using resource merit order
        best_bound = 0.0
        supplies = [self.solar_profile[h], self.wind_profile[h], self.max_coal]
        prices = [0.0, 0.0, self.perturbed_price[h]]
        rem_demand = demand
        for k in range(len(supplies)):
            if (supplies[k] < rem_demand):
                best_bound += supplies[k] * prices[k]
                rem_demand -= supplies[k]
            else:
                best_bound += rem_demand * prices[k]
                break
        return best_bound

class rl_model:
    def __init__(self):
        # Initialize the seed for reproducibility

        SEED = 97
        # Fill the demand and price forecasts with some dummy data
        # TO DO: Add ERCOT demand and price forecasts
        parms = self.set_parms()

        env = lambda: energy_env(parms)
        env = DummyVecEnv([env])
        env.seed(SEED)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        env.seed(SEED)
        self.env = env
        self.env.reset()
        self.policy = PPO("MlpPolicy", self.env, verbose=1, ent_coef=0.08, seed=SEED)

    def set_parms(self):
        params ={'min_load': 10,  # Minimum load in MW,'
                'max_load': 100,  # Maximum load in MW,
                'min_price': 0.1,  # Minimum price in $/MWh,
                'max_price': 5,    # Maximum price in $/MWh,
                'max_solar': 10,   # Maximum solar generation in MW,
                'max_wind': 10,    # Maximum wind generation in MW,
                'battery_capacity': 10, # Battery capacity in MW,
                'max_coal': 100, # Maximum coal generation in MW,
                'demand_forecast': np.zeros(48, dtype=np.float32), # 48 hours demand forecast
                'price_forecast': np.zeros(48, dtype=np.float32), # 48 hours price forecast
                }
        # Fill the demand and price forecasts with some cyclical demand data
        hours = np.arange(0, 48)
        base_demand = (params['min_load'] + params['max_load']) / 2.0
        peak_demand = (params['max_load'] - params['min_load']) / 2.0
        params['demand_forecast'] = base_demand + peak_demand * np.sin(2 * np.pi * hours / 24)  # Cyclical demand
        
        base_price = (params['min_price'] + params['max_price']) / 2.0
        peak_price = (params['max_price'] - params['min_price']) / 2.0
        params['price_forecast'] = base_price + peak_price * np.sin(2 * np.pi * hours / 24)  # Cyclical price

        # 24 hr Solar and wind profiles
        # Solar: bell curve peaking at solar_peak_hour, zero at night
        hours = np.arange(0, 24)
        max_solar = params['max_solar']
        solar_peak_hour = 12  # Solar peak at noon
        solar_profile = max_solar * np.exp(-0.5 * ((hours - solar_peak_hour) / 3)**2)
        #solar_profile = np.ones(24, dtype=np.float32) * max_solar  # Initialize solar profile
        solar_profile[(hours < 6) | (hours > 18)] = 0  # No solar before 6 AM or after 6 PM
        params['solar_profile'] = solar_profile

        # Wind: more random, with a slight pattern (e.g., stronger at night/morning)
        max_wind = params['max_wind']
        base_wind = 0.5 * (np.sin((hours + 6) * np.pi / 12) + 1)  # Patterned component
        noise = 0.2 * np.random.randn(24)  # Random noise
        wind_profile = np.clip(max_wind * (base_wind + noise), 0, max_wind)
        #wind_profile = np.ones(24, dtype=np.float32) * max_wind  # Initialize wind profile
        params['wind_profile'] = wind_profile
        return params

    def train(self, timesteps):
        self.policy.learn(total_timesteps=timesteps)

    def predict(self, output):
        imbalances = []
        actions = []
        rewards = []
        total_costs = []
        best_bounds = []
        time_steps = []
        self.env.reset()

        # Simulate for each hour of the day
        for hour in range(24):
            time_steps.append(hour)
            done = False
            # Reset the environment for each hour & build observations
            self.env.envs[0].set_state(hour)
            obs = self.env.envs[0].build_obs()
            while (True):
                if (obs.ndim == 1):
                    obs = obs.reshape((1, -1))
                action, hidden_state = self.policy.predict(obs)
                obs, reward, done, info = self.env.step(action)

                # Terminate the episode if done
                if (done):
                    # Store the results
                    actions.append(action)
                    imbalances.append(info[0]['Imbalance'])
                    total_costs.append(info[0]['Total Cost'])
                    best_bounds.append(info[0]['Best Bound'])
                    rewards.append(reward)
                    break

            # Update the environment for the next hour

        # Plot the results over steps for each hour
        if output:
            plt.figure(figsize=(10, 5))
            plt.subplot(3, 2, 1)
            plt.plot(time_steps, imbalances, marker='o')
            plt.title('Imbalance')
            plt.xlabel('Hour')
            plt.ylabel('Imbalance')

            plt.subplot(3, 2, 2)
            plt.plot(time_steps, self.env.envs[0].battery_bal, marker='o')
            plt.title('Battery Balance')
            plt.xlabel('Hour')
            plt.ylabel('Battery Balance')

            plt.subplot(3, 2, 3)
            plt.plot(time_steps, self.env.envs[0].perturbed_demand[0:24], marker='o')
            plt.title('Demand')
            plt.xlabel('Hour')
            plt.ylabel('Demand')

            plt.subplot(3, 2, 4)
            plt.plot(time_steps, self.env.envs[0].perturbed_price[0:24], marker='o')
            plt.title('Price')
            plt.xlabel('Hour')
            plt.ylabel('Price')

            plt.subplot(3, 2, 5)
            plt.plot(time_steps, total_costs, label='Total Cost', marker='o')
            plt.plot(time_steps, best_bounds, label='Best Bound', marker='x')
            plt.title('Total Cost vs Best Bound')
            plt.xlabel('Hour')
            plt.ylabel('Cost')
            plt.legend()

            plt.subplot(3, 2, 6)
            plt.plot(time_steps, self.env.envs[0].solar_profile, label='Solar Profile', marker='o')
            plt.plot(time_steps, self.env.envs[0].wind_profile, label='Wind Profile', marker='x')
            plt.title('Renewable Profiles')
            plt.xlabel('Hour')
            plt.ylabel('Capacity')

            plt.legend()
            plt.tight_layout()
            plt.show()

        return None

# Test code
model = rl_model()
model.train(200000)
info = model.predict(True)