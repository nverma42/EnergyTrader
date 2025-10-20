from asyncio import windows_events
from encodings.cp437 import decoding_map
from sqlite3 import SQLITE_OK_LOAD_PERMANENTLY
from rl_model import rl_model
import numpy as np
import matplotlib.pyplot as plt

def plot_series1(time_steps, mean, std, minval, maxval, title, xlabel, ylabel, filename):
    plt.plot(time_steps, mean, label=title)
    plt.fill_between(range(len(mean)), np.maximum(minval, mean - std), np.minimum(maxval, mean + std), alpha=0.3, label="1 SD")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()
    plt.close()

def plot_series2(time_steps, label1, mean1, std1, minval1, maxval1, label2, mean2, std2, minval2, maxval2, title, xlabel, ylabel, filename):
    plt.plot(time_steps, mean1, label=label1, marker='x')
    plt.fill_between(range(len(mean1)), np.maximum(minval1, mean1 - std1), np.minimum(maxval1, mean1 + std1), alpha=0.3, label="1 SD")
    plt.plot(time_steps, mean2, label=label2, marker='o')
    plt.fill_between(range(len(mean2)), np.maximum(minval2, mean2 - std2), np.minimum(maxval2, mean2 + std2), alpha=0.3, label="1 SD")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()
    plt.close()

def plot_series3(time_steps, label1, mean1, std1, minval1, maxval1, label2, mean2, std2, minval2, maxval2, title, xlabel, ylabel, filename):
    plt.plot(time_steps, mean1, label=label1, marker='x', color="tab:blue")
    plt.fill_between(range(len(mean1)), np.maximum(minval1, mean1 - std1), np.minimum(maxval1, mean1 + std1), alpha=0.3, color="tab:blue", label="1 SD")
    plt.xlabel(xlabel)
    plt.ylabel(label1, color='tab:blue')
    plt.legend(loc='lower left')

    ax2 = plt.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(time_steps, mean2, label=label2, marker='o', color="tab:orange")
    ax2.fill_between(range(len(mean2)), np.maximum(minval2, mean2 - std2), np.minimum(maxval2, mean2 + std2), alpha=0.3, color="tab:orange", label="1 SD")
    ax2.set_ylabel(label2, color='tab:orange')
    ax2.legend(loc='upper right')
        
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()
    plt.close()
import random

# Fix the random generator so sampling is reproducible
random.seed(0)  

# Sample 10 seeds from a large range (e.g., 0 to 9999)
num_seeds = 10
seeds = random.sample(range(0, 10000), num_seeds)

imbalance_gap = [[] for _ in range(num_seeds)]
best_bound_gap = [[] for _ in range(num_seeds)]
battery_bal = [[] for _ in range(num_seeds)]
demand = [[] for _ in range(num_seeds)]
price = [[] for _ in range(num_seeds)]
total_costs = [[] for _ in range(num_seeds)]
best_bounds = [[] for _ in range(num_seeds)]
solar_gen = [[] for _ in range(num_seeds)]
wind_gen = [[] for _ in range(num_seeds)]
i = 0
for seed in seeds:
    model = rl_model(seed)
    outputs = model.predict(False)
    imbalance_gap[i] = outputs['Imbalance Gap']
    best_bound_gap[i] = outputs['Best Bound Gap']
    battery_bal[i] = outputs['Battery Balance']
    demand[i] = outputs['Demand']
    price[i] = outputs['Price']
    total_costs[i] = outputs['Total Costs']
    best_bounds[i] = outputs['Best Bounds']
    solar_gen[i] = outputs['Solar Generation']
    wind_gen[i] = outputs['Wind Generation']

    i += 1

time_steps = np.arange(len(imbalance_gap[0]))
avg_imbalance_gap = np.mean(imbalance_gap, axis=0)
avg_best_bound_gap = np.mean(best_bound_gap, axis=0)
avg_battery_bal = np.mean(battery_bal, axis=0)
avg_demand = np.mean(demand, axis=0)
avg_price = np.mean(price, axis=0)
avg_total_costs = np.mean(total_costs, axis=0)
avg_best_bounds = np.mean(best_bounds, axis=0)
avg_solar_gen = np.mean(solar_gen, axis=0)
avg_wind_gen = np.mean(wind_gen, axis=0)

std_imbalance_gap = np.std(imbalance_gap, axis=0)
std_best_bound_gap = np.std(best_bound_gap, axis=0)
std_battery_bal = np.std(battery_bal, axis=0)
std_demand = np.std(demand, axis=0)
std_price = np.std(price, axis=0)
std_total_costs = np.std(total_costs, axis=0)
std_best_bounds = np.std(best_bounds, axis=0)
std_solar_gen = np.std(solar_gen, axis=0)
std_wind_gen = np.std(wind_gen, axis=0)

min_imbalance_gap = np.min(imbalance_gap, axis=0)
min_best_bound_gap = np.min(best_bound_gap, axis=0)
min_battery_bal = np.min(battery_bal, axis=0)
min_demand = np.min(demand, axis=0)
min_price = np.min(price, axis=0)
min_total_costs = np.min(total_costs, axis=0)
min_best_bounds = np.min(best_bounds, axis=0)
min_solar_gen = np.min(solar_gen, axis=0)
min_wind_gen = np.min(wind_gen, axis=0)

max_imbalance_gap = np.max(imbalance_gap, axis=0)
max_best_bound_gap = np.max(best_bound_gap, axis=0)
max_battery_bal = np.max(battery_bal, axis=0)
max_demand = np.max(demand, axis=0)
max_price = np.max(price, axis=0)
max_total_costs = np.max(total_costs, axis=0)
max_best_bounds = np.max(best_bounds, axis=0)
max_solar_gen = np.max(solar_gen, axis=0)
max_wind_gen = np.max(wind_gen, axis=0)

plot_series1(time_steps, avg_imbalance_gap, std_imbalance_gap, min_imbalance_gap, max_imbalance_gap, "Avg Imbalance Gap", "Hour", "Imbalance Gap (%)", "avg_imbalance_gap.pdf")
plot_series1(time_steps, avg_best_bound_gap, std_best_bound_gap, min_best_bound_gap, max_best_bound_gap, "Avg Best Bound Gap", "Hour", "Best Bound Gap (%)", "avg_best_bound_gap.pdf")
plot_series1(time_steps, avg_battery_bal, std_battery_bal, min_battery_bal, max_battery_bal, "Avg Battery Balance", "Hour", "Battery Balance (MW)", "avg_battery_bal.pdf")
plot_series1(time_steps, avg_demand, std_demand, min_demand, max_demand, "Avg Demand", "Hour", "Demand (MW)", "avg_demand.pdf")
plot_series1(time_steps, avg_price, std_price, min_price, max_price, "Avg Price", "Hour", "Price ($/MWh)", "avg_price.pdf")
plot_series2(time_steps, 'Avg Total Cost', avg_total_costs, std_total_costs, min_total_costs, max_total_costs, "Avg Best Bound", avg_best_bounds, std_best_bounds, min_best_bounds, max_best_bounds, "Avg Total Cost vs Avg Best Bound", "Hour", "Cost ($)", "avg_cost.pdf")
plot_series2(time_steps, 'Avg Solar', avg_solar_gen, std_solar_gen, min_solar_gen, max_solar_gen, "Avg Wind", avg_wind_gen, std_wind_gen, min_wind_gen, max_wind_gen, "Renewable Generation", "Hour", "Generation (MW)", "avg_renewable_gen.pdf")
plot_series3(time_steps, 'Avg Price', avg_price, std_price, min_price, max_price, 'Avg Battery Balance', avg_battery_bal, std_battery_bal, min_battery_bal, max_battery_bal, "Avg Price vs Battery Balance", "Hour", "Price ($/MWh)", "avg_price_battery_bal.pdf")


