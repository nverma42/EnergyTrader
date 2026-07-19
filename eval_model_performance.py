from rl_model import rl_model
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import time

def plot_series1(time_steps, mean, std, minval, maxval, title, xlabel, ylabel, filename):
    plt.plot(time_steps, mean, label=title)
    plt.fill_between(range(len(mean)), np.maximum(minval, mean - std), np.minimum(maxval, mean + std), alpha=0.3, label="1 SD")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
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
    plt.close()

def plot_runs(time_steps, run_params, time_series, xlabel, ylabel, filename):
    fig, ax = plt.subplots()
    for i in range(len(run_params)):
        avg = np.mean(time_series[i], axis=0)
        ax.plot(time_steps, avg, label=run_params[i]['name'], marker='x')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()

def plot_runs_bar(run_params, vals, ylabel, filename):
    fig, ax = plt.subplots()
    num_runs = len(run_params)
    names = [run_params[i]['name'] for i in range(num_runs)]
    colors = plt.cm.tab10.colors[:num_runs]
    hatches = ['/', '\\', 'x', 'o']
    bars = ax.bar(range(num_runs), vals, color=colors, hatch=[hatches[i] for i in range(num_runs)], edgecolor='black')

    for bar, name in zip(bars, names):
        bar.set_label(name)

    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()

def evaluate_seed(i, j, model, imbalance_gap, best_bound_gap, battery_bal, demand, price, total_costs, best_bounds, solar_gen, wind_gen, reur):
    print(f"Evaluating on run {i+1}, seed {j+1}: {seeds[j]}")
    outputs = model.predict(False, seed=seeds[j]    )
    imbalance_gap[i][j] = outputs['Imbalance Gap']
    best_bound_gap[i][j] = outputs['Best Bound Gap']
    battery_bal[i][j] = outputs['Battery Balance']
    demand[i][j] = outputs['Demand']
    price[i][j] = outputs['Price']
    total_costs[i][j] = outputs['Total Costs']
    best_bounds[i][j] = outputs['Best Bounds']
    solar_gen[i][j] = outputs['Solar Generation']
    wind_gen[i][j] = outputs['Wind Generation']
    reur[i][j] = outputs['Renewable Utilization Ratio']
    print(f"Completed run {i+1}, seed {j+1}: {seeds[j]}")

# Initialize the timer to measure total evaluation time
start_time = time.time()

# Fix the random generator so sampling is reproducible
random.seed(0)
np.random.seed(0)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate RL model performance with or without safety layer')
parser.add_argument('--seed', type=int, default=42, help='Training seed (default: 42)')
parser.add_argument('--num_seeds', type=int, default=1, help='Number of evaluation seeds (default: 10)')
parser.add_argument('--reward_mode', type=int, default=1, help='Reward Mode (1: Gated, 2: Weighted, default: 1)')
parser.add_argument('--use_curriculum', type=int, default=1, help='Use Curriculum (1: Yes, 2: No, default: 1)')
parser.add_argument('--imbal_weight', type=float, default=1000.0, help='Imbalance Weight (default: 1000.0)')
parser.add_argument('--cost_weight', type=float, default=10.0, help='Cost Weight (default: 10.0)')

args = parser.parse_args()

INITIAL_SEED = args.seed
NUM_SEEDS = args.num_seeds

# Sample seeds from a large range (e.g., 0 to 9999)
seeds = random.sample(range(0, 10000), NUM_SEEDS)

NUM_HOURS = 24
NUM_RUNS = 4

run_params = {0: {'name': 'Gated with Curriculum', 'reward_mode': 1, 'use_curriculum': 1, 'imbal_weight': 1000.0, 'cost_weight': 10.0},
              1: {'name': 'Gated without Curriculum', 'reward_mode': 1, 'use_curriculum': 0, 'imbal_weight': 1000.0, 'cost_weight': 10.0},
              2: {'name': 'Weighted with Curriculum', 'reward_mode': 2, 'use_curriculum': 1, 'imbal_weight': 1000.0, 'cost_weight': 10.0},
              3: {'name': 'Weighted without Curriculum', 'reward_mode': 2, 'use_curriculum': 0, 'imbal_weight': 1000.0, 'cost_weight': 10.0}}

imbalance_gap  = np.zeros((NUM_RUNS, NUM_SEEDS, NUM_HOURS))
best_bound_gap = np.zeros((NUM_RUNS, NUM_SEEDS, NUM_HOURS))
battery_bal    = np.zeros((NUM_RUNS, NUM_SEEDS, NUM_HOURS))
demand         = np.zeros((NUM_RUNS, NUM_SEEDS, NUM_HOURS))
price          = np.zeros((NUM_RUNS, NUM_SEEDS, NUM_HOURS))
total_costs    = np.zeros((NUM_RUNS, NUM_SEEDS, NUM_HOURS))
best_bounds    = np.zeros((NUM_RUNS, NUM_SEEDS, NUM_HOURS))
solar_gen      = np.zeros((NUM_RUNS, NUM_SEEDS, NUM_HOURS))
wind_gen       = np.zeros((NUM_RUNS, NUM_SEEDS, NUM_HOURS))
reur           = np.zeros((NUM_RUNS, NUM_SEEDS))

for i in range(NUM_RUNS):
    REWARD_MODE = run_params[i]['reward_mode']
    USE_CURRICULUM = run_params[i]['use_curriculum']
    IMBAL_WEIGHT = run_params[i]['imbal_weight']
    COST_WEIGHT = run_params[i]['cost_weight']

    # Train model ONCE on a single seed
    print("Training model...")
    model = rl_model(initial_seed=INITIAL_SEED, reward_mode=REWARD_MODE, use_curriculum=USE_CURRICULUM, imbal_weight=IMBAL_WEIGHT, cost_weight=COST_WEIGHT)
    model.train()
    print("Training complete!") 

    # Evaluate model on multiple seeds
    for j, seed in enumerate(seeds):
        evaluate_seed(i, j, model, imbalance_gap, best_bound_gap, battery_bal, demand, price, total_costs, best_bounds, solar_gen, wind_gen, reur)

print("Computing statistics and generating plots for all runs...")
time_steps = np.arange(NUM_HOURS)

for i in range(NUM_RUNS):
    run_name = run_params[i]['name'].replace(' ', '_').lower()
    avg_imbalance_gap  = np.mean(imbalance_gap[i],  axis=0)
    avg_best_bound_gap = np.mean(best_bound_gap[i], axis=0)
    avg_battery_bal    = np.mean(battery_bal[i],    axis=0)
    avg_demand         = np.mean(demand[i],         axis=0)
    avg_price          = np.mean(price[i],          axis=0)
    avg_total_costs    = np.mean(total_costs[i],    axis=0)
    avg_best_bounds    = np.mean(best_bounds[i],    axis=0)
    avg_solar_gen      = np.mean(solar_gen[i],      axis=0)
    avg_wind_gen       = np.mean(wind_gen[i],       axis=0)

    std_imbalance_gap  = np.std(imbalance_gap[i],  axis=0)
    std_best_bound_gap = np.std(best_bound_gap[i], axis=0)
    std_battery_bal    = np.std(battery_bal[i],    axis=0)
    std_demand         = np.std(demand[i],         axis=0)
    std_price          = np.std(price[i],          axis=0)
    std_total_costs    = np.std(total_costs[i],    axis=0)
    std_best_bounds    = np.std(best_bounds[i],    axis=0)
    std_solar_gen      = np.std(solar_gen[i],      axis=0)
    std_wind_gen       = np.std(wind_gen[i],       axis=0)

    min_imbalance_gap  = np.min(imbalance_gap[i],  axis=0)
    min_best_bound_gap = np.min(best_bound_gap[i], axis=0)
    min_battery_bal    = np.min(battery_bal[i],    axis=0)
    min_demand         = np.min(demand[i],         axis=0)
    min_price          = np.min(price[i],          axis=0)
    min_total_costs    = np.min(total_costs[i],    axis=0)
    min_best_bounds    = np.min(best_bounds[i],    axis=0)
    min_solar_gen      = np.min(solar_gen[i],      axis=0)
    min_wind_gen       = np.min(wind_gen[i],       axis=0)

    max_imbalance_gap  = np.max(imbalance_gap[i],  axis=0)
    max_best_bound_gap = np.max(best_bound_gap[i], axis=0)
    max_battery_bal    = np.max(battery_bal[i],    axis=0)
    max_demand         = np.max(demand[i],         axis=0)
    max_price          = np.max(price[i],          axis=0)
    max_total_costs    = np.max(total_costs[i],    axis=0)
    max_best_bounds    = np.max(best_bounds[i],    axis=0)
    max_solar_gen      = np.max(solar_gen[i],      axis=0)
    max_wind_gen       = np.max(wind_gen[i],       axis=0)

    plot_series1(time_steps, avg_imbalance_gap, std_imbalance_gap, min_imbalance_gap, max_imbalance_gap, "Avg Imbalance Gap", "Hour", "Imbalance Gap (%)", f"{run_name}_avg_imbalance_gap.pdf")
    plot_series1(time_steps, avg_best_bound_gap, std_best_bound_gap, min_best_bound_gap, max_best_bound_gap, "Avg Best Bound Gap", "Hour", "Best Bound Gap (%)", f"{run_name}_avg_best_bound_gap.pdf")
    plot_series1(time_steps, avg_battery_bal, std_battery_bal, min_battery_bal, max_battery_bal, "Avg Battery Balance", "Hour", "Battery Balance (MW)", f"{run_name}_avg_battery_bal.pdf")
    plot_series1(time_steps, avg_demand, std_demand, min_demand, max_demand, "Avg Demand", "Hour", "Demand (MW)", f"{run_name}_avg_demand.pdf")
    plot_series1(time_steps, avg_price, std_price, min_price, max_price, "Avg Price", "Hour", "Price ($/MWh)", f"{run_name}_avg_price.pdf")
    plot_series2(time_steps, 'Avg Total Cost', avg_total_costs, std_total_costs, min_total_costs, max_total_costs, "Avg Best Bound", avg_best_bounds, std_best_bounds, min_best_bounds, max_best_bounds, "Avg Total Cost vs Avg Best Bound", "Hour", "Cost ($)", f"{run_name}_avg_cost.pdf")
    plot_series2(time_steps, 'Avg Solar', avg_solar_gen, std_solar_gen, min_solar_gen, max_solar_gen, "Avg Wind", avg_wind_gen, std_wind_gen, min_wind_gen, max_wind_gen, "Renewable Generation", "Hour", "Generation (MW)", f"{run_name}_avg_renewable_gen.pdf")
    plot_series3(time_steps, 'Avg Price', avg_price, std_price, min_price, max_price, 'Avg Battery Balance', avg_battery_bal, std_battery_bal, min_battery_bal, max_battery_bal, "Avg Price vs Battery Balance", "Hour", "Price ($/MWh)", f"{run_name}_avg_price_battery_bal.pdf")

print("Generating comparison plots across all runs...")
plot_runs(time_steps, run_params, imbalance_gap, "Hour", "Avg Imbalance Gap (%)", f"avg_imbalance_gap_runs.pdf")
plot_runs(time_steps, run_params, best_bound_gap, "Hour", "Avg Best Bound Gap (%)", f"avg_best_bound_gap_runs.pdf")
plot_runs(time_steps, run_params, total_costs, "Hour", "Avg Total Cost ($)", f"avg_total_costs_runs.pdf")
plot_runs(time_steps, run_params, battery_bal, "Hour", "Avg Battery Balance (MW)", f"avg_battery_bal_runs.pdf")

avg_reur = [np.mean(reur[i]) for i in range(NUM_RUNS)]
plot_runs_bar(run_params, avg_reur, "Avg Renewable Utilization Ratio (%)", "avg_reur_runs_bar.pdf")

end_time = time.time()
print(f"Done! Total evaluation time: {end_time - start_time:.2f} seconds")