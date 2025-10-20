import torch
import random
import pandas as pd
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
        self.conv_profile = params['conv_profile']
        self.battery_bal = np.zeros((24,), dtype=np.float32)  # Battery balance for each hour
        self.best_bound = 0.0
        self.prev_imbalance = 0.0
        self.prev_total_cost = 0.0
        self.current_step = 0
        self.max_steps = 20000
        self.imbalance_rel_gap = 0.25
        self.best_bound_rel_gap = 0.25

        # Action Space: Make action space continuous for each agent
        self.action_space = spaces.Box(
                            low=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),   # [solar_gen, wind_gen, battery, conv_gen]
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
        self.current_step = 0
        
        # Reset battery balance for the new hour as the battery balance is carried over from the previous hour
        self.battery_bal[h] = self.battery_bal[h-1] if h >= 1 else 0.0

    def set_rel_gap(self, imbalance_rel_gap, best_bound_rel_gap):
        self.imbalance_rel_gap = imbalance_rel_gap
        self.best_bound_rel_gap = best_bound_rel_gap

    # For realistic simulation, check https://www.ercot.com/gridmktinfo/dashboards/
    # TO DO: slightly perturb the hour, demand and price forecasts to simulate realistic scenarios
    def reset(self):
        self.current_hour = np.random.randint(0,24)  # Randomly select an hour to start the episode
        self.current_imbalance = 0.0
        self.best_bound = 0.0
        self.done = False
        demand_noise = np.random.normal(0, 500, size=self.demand.shape)  # std dev = 500 MW
        price_noise = np.random.normal(0, 1, size=self.price.shape)  # std dev = 1 $/MWh
        self.perturbed_demand = np.clip(self.demand + demand_noise, self.min_demand, self.max_demand)
        self.perturbed_price = np.clip(self.price + price_noise, self.min_price, self.max_price)

        # Clip to min/max bounds
        self.perturbed_demand = np.clip(self.perturbed_demand, self.min_demand, self.max_demand)
        self.perturbed_price = np.clip(self.perturbed_price, self.min_price, self.max_price)
        
        # Set prev variables
        self.prev_imbalance = 0.0
        self.prev_total_cost = 0.0

        # Initialize current step for each new episode
        self.current_step = 0

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
        # Update the current step
        self.current_step += 1

        # Compute reward
        reward, supply, demand, total_cost, gen_amts = self.compute_reward(action)

        # Check if the episode should end. We allow for at most 1000
        # steps so that imbalance and cost drop to an acceptable value.
        best_bound_ratio = 1.0
        met_imbal_flag, imbalance_ratio = self.check_if_met_imbalance_criterion(supply, demand)
        if (self.current_step >= self.max_steps):
            self.done = True
        elif (met_imbal_flag):
            self.done, best_bound_ratio = self.check_if_met_best_bound_criterion(total_cost, self.best_bound)
        
        # Return the state, reward, done flag, and additional info
        return self.build_obs(), reward, self.done, {'hour': self.current_hour, 'Imbalance Ratio':imbalance_ratio, 'Best Bound Ratio': best_bound_ratio, 'Total Cost': total_cost, 'Best Bound':self.best_bound, 'Settlement Price':self.perturbed_price[self.current_hour], 'Generation Amounts':gen_amts}
    
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

        # Imbalance cost
        self.current_imbalance = abs(demand - supply)
        #imbalance_cost = -self.penalty * self.current_imbalance  # Cost of imbalance based on current price

        # In reward calculation, check for invalid or harmful actions first
        reward = 0.0
        if (battery_violation > 0.0):
            # Invalid action. So return a large negative reward
            reward =  -self.penalty * self.max_demand * 1000

        # Give a huge negative reward if no available capacity but action is taken
        if (solar_usage > 0.0 and self.solar_profile[h] <= 0.0):
            reward += -self.penalty * self.max_demand * 1000

        if (wind_usage > 0.0 and self.wind_profile[h] <= 0.0):
            reward += -self.penalty * self.max_demand * 1000

        # Reward imbalance reduction and penalize imbalance increase
        reward += self.penalty * (self.prev_imbalance - self.current_imbalance)


        # if (imbalance_cost < 0.0):
        #     # Discourage imbalance
        #     reward += imbalance_cost

        # Calculate the generation and battery operation costs
        solar_cost = 0.0
        wind_cost = 0.0
        battery_cost = abs(battery_gen) * self.min_price
        conv_cost = conv_gen * self.perturbed_price[h]
        total_cost = solar_cost + wind_cost + battery_cost + conv_cost
        reward -= total_cost # cost of generation

        # Extra reward for cost performance if imbalance also reduced
        if (self.current_imbalance < self.prev_imbalance and total_cost < self.prev_total_cost):
            reward += self.penalty * (self.prev_total_cost - total_cost)

        # --- future arbitrage reward ---
        profit = self.calc_arbitrage_bonus(battery_gen)
        reward += profit
        # --- end of future arbitrage reward ---

        self.best_bound = self.calc_best_bound(demand)
        self.best_bound += battery_cost

        # Update previous imbalance and total cost
        self.prev_imbalance = self.current_imbalance
        self.prev_total_cost = total_cost

        return reward, supply, demand, total_cost, [solar_gen, wind_gen, battery_gen, conv_gen]

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

    def check_if_met_best_bound_criterion(self, total_cost, best_bound):
        if (total_cost <= best_bound or total_cost <= 1.0):
            return True, 0.0

        denom = total_cost
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
        max_supply = self.solar_profile[h] + self.wind_profile[h] + self.conv_profile[h]
        
        denom = min(demand, max_supply)
        rel_gap = self.imbalance_rel_gap
        imbalance_ratio = 1.0
        if denom > 0:
            imbalance_ratio = self.current_imbalance / denom
            if imbalance_ratio <= rel_gap:
                return True, imbalance_ratio
        return False, imbalance_ratio

    # Merit order calculation for the best bound for the cost.
    # This is used to calculate the lowest possible cost to meet the demand
    def calc_best_bound(self, demand):
        # calculate the lowest possible cost to meet the demand
        h = self.current_hour
        price = self.perturbed_price[h]
        # Calculate the best bound using resource merit order
        best_bound = 0.0
        supplies = [self.solar_profile[h], self.wind_profile[h], self.conv_profile[h]]
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
    def __init__(self, SEED=42):
        # Fill the demand and price forecasts with some dummy data
        parms = self.set_parms()

        env = lambda: energy_env(parms)
        env = DummyVecEnv([env])
        env.seed(SEED)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        env.seed(SEED)
        self.env = env
        self.env.reset()
        self.policy = PPO("MlpPolicy", self.env, verbose=1, seed=SEED, device='auto')
        print(torch.__version__)

    def set_parms(self):
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
        return params

    def train(self):
        curriculum = [(0.40, 0.40, 40000), (0.20, 0.30, 50000), (0.10, 0.20, 60000), (0.05, 0.10, 80000), (0.02, 0.10, 100000)]
        for level, (imbal_rel_gap, best_bound_rel_gap, timesteps) in enumerate(curriculum):
            self.env.envs[0].set_rel_gap(imbal_rel_gap, best_bound_rel_gap)
            self.policy.learn(total_timesteps=timesteps, reset_num_timesteps=False)

    def predict(self, output):
        imbalance_gap = []
        best_bound_gap = []
        actions = []
        rewards = []
        total_costs = []
        best_bounds = []
        settlement_prices = []
        time_steps = []
        gen_amts = []
        self.env.reset()

        # Simulate for each hour of the day
        for hour in range(24):
            time_steps.append(hour)
            done = False
            # Reset the environment for each hour & build observations
            self.env.envs[0].set_state(hour)
            self.env.envs[0].set_rel_gap(0.02, 0.10)
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
                    imbalance_gap.append(100*info[0]['Imbalance Ratio'])
                    best_bound_gap.append(100*info[0]['Best Bound Ratio'])
                    total_costs.append(info[0]['Total Cost'])
                    best_bounds.append(info[0]['Best Bound'])
                    settlement_prices.append(info[0]['Settlement Price'])
                    gen_amts.append(info[0]['Generation Amounts'])
                    rewards.append(reward)
                    break

            # Update the environment for the next hour

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
            'Battery Balance': self.env.envs[0].battery_bal.tolist(),
            'Demand': self.env.envs[0].perturbed_demand[0:24].tolist(),
            'Price': self.env.envs[0].perturbed_price[0:24].tolist(),
            'Total Costs': total_costs,
            'Best Bounds': best_bounds,
            'Solar Generation': [gen_amt[0] for gen_amt in gen_amts],
            'Wind Generation': [gen_amt[1] for gen_amt in gen_amts],
            'Settlement Prices': settlement_prices,
            'Generation Amounts': gen_amts
        }
        return output_dict
