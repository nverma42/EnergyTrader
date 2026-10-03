import multiprocessing as mp
from rl_model import rl_model
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import json
from pathlib import Path
from dataclasses import dataclass
import datetime
import sys
import torch

@dataclass
class Experiment:
    name: str
    reward_mode: int
    use_curriculum: int
    imbal_weight: float
    cost_weight: float

def load_experiments(json_path):
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    experiments = []
    for i, row in enumerate(data):
        experiments.append(
            Experiment(
                name=row['name'],
                reward_mode=int(row['reward_mode']),
                use_curriculum=int(row['use_curriculum']),
                imbal_weight=float(row['imbal_weight']),
                cost_weight=float(row['cost_weight'])
            )
        )
    return experiments

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

    if len(vals) != num_runs:
        raise ValueError(f"plot_runs_bar expected {num_runs} values, got {len(vals)}")

    cmap = plt.get_cmap('tab20', num_runs)
    colors = [cmap(i) for i in range(num_runs)]
    bars = ax.bar(range(num_runs), vals, color=colors,edgecolor='black')

    for bar, name in zip(bars, names):
        bar.set_label(name)

    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()

def evaluate_seed(i, j, seeds, model, imbalance_gap, best_bound_gap, battery_bal, demand, price, total_costs, best_bounds, solar_gen, wind_gen, reur, best_cost, best_reur):
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
    best_cost[i][j] = outputs['Best Cost']
    best_reur[i][j] = outputs['Best Renewable Utilization Ratio']
    print(f"Completed run {i+1}, seed {j+1}: {seeds[j]}")

def train_and_eval(training_seed):
    # Initialize the timer to measure total evaluation time
    start_time = time.time()

    # Fix the random generator so sampling is reproducible
    random.seed(0)
    np.random.seed(0)

    INITIAL_SEED = training_seed
    NUM_EVAL_SEEDS = 10

    # Sample seeds from a large range (e.g., 0 to 9999)
    seeds = random.sample(range(0, 10000), NUM_EVAL_SEEDS)

    NUM_HOURS = 24

    EXPERIMENTS_FILE = "data/experiments_gated_weighted.json"
    experiments = load_experiments(EXPERIMENTS_FILE)
    NUM_RUNS = len(experiments)

    experiments_stem = Path(EXPERIMENTS_FILE).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = Path("results") / f"{experiments_stem}_{training_seed}_{timestamp}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUTPUT_DIR / "train_log.txt"
    sys.stdout = open(log_path, "w", encoding="utf-8", buffering=1)

    def out(filename):
        return str(OUTPUT_DIR / filename)

    run_params = [{} for _ in range(NUM_RUNS)]
    for i, exp in enumerate(experiments):
        run_params[i] = {
                'name': exp.name,
                'reward_mode': exp.reward_mode,
                'use_curriculum': exp.use_curriculum,
                'imbal_weight': exp.imbal_weight,
                'cost_weight': exp.cost_weight
            }

    imbalance_gap  = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS, NUM_HOURS))
    best_bound_gap = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS, NUM_HOURS))
    battery_bal    = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS, NUM_HOURS))
    demand         = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS, NUM_HOURS))
    price          = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS, NUM_HOURS))
    total_costs    = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS , NUM_HOURS))
    best_bounds    = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS, NUM_HOURS))
    solar_gen      = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS, NUM_HOURS))
    wind_gen       = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS, NUM_HOURS))
    reur           = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS))
    best_cost      = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS))
    best_reur     = np.zeros((NUM_RUNS, NUM_EVAL_SEEDS))

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
            evaluate_seed(i, j, seeds, model, imbalance_gap, best_bound_gap, battery_bal, demand, price, total_costs, best_bounds, solar_gen, wind_gen, reur, best_cost, best_reur)
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

        plot_series1(time_steps, avg_imbalance_gap, std_imbalance_gap, min_imbalance_gap, max_imbalance_gap, "Avg Imbalance Gap", "Hour", "Imbalance Gap (%)", out(f"{run_name}_avg_imbalance_gap.pdf"))
        plot_series1(time_steps, avg_best_bound_gap, std_best_bound_gap, min_best_bound_gap, max_best_bound_gap, "Avg Best Bound Gap", "Hour", "Best Bound Gap (%)", out(f"{run_name}_avg_best_bound_gap.pdf"))
        plot_series1(time_steps, avg_battery_bal, std_battery_bal, min_battery_bal, max_battery_bal, "Avg Battery Balance", "Hour", "Battery Balance (MW)", out(f"{run_name}_avg_battery_bal.pdf"))
        plot_series1(time_steps, avg_demand, std_demand, min_demand, max_demand, "Avg Demand", "Hour", "Demand (MW)", out(f"{run_name}_avg_demand.pdf"))
        plot_series1(time_steps, avg_price, std_price, min_price, max_price, "Avg Price", "Hour", "Price ($/MWh)", out(f"{run_name}_avg_price.pdf"))
        plot_series2(time_steps, 'Avg Total Cost', avg_total_costs, std_total_costs, min_total_costs, max_total_costs, "Avg Best Bound", avg_best_bounds, std_best_bounds, min_best_bounds, max_best_bounds, "Avg Total Cost vs Avg Best Bound", "Hour", "Cost ($)", out(f"{run_name}_avg_cost.pdf"))
        plot_series2(time_steps, 'Avg Solar', avg_solar_gen, std_solar_gen, min_solar_gen, max_solar_gen, "Avg Wind", avg_wind_gen, std_wind_gen, min_wind_gen, max_wind_gen, "Renewable Generation", "Hour", "Generation (MW)", out(f"{run_name}_avg_renewable_gen.pdf"))
        plot_series3(time_steps, 'Avg Price', avg_price, std_price, min_price, max_price, 'Avg Battery Balance', avg_battery_bal, std_battery_bal, min_battery_bal, max_battery_bal, "Avg Price vs Battery Balance", "Hour", "Price ($/MWh)", out(f"{run_name}_avg_price_battery_bal.pdf"))

    print("Generating comparison plots across all runs...")
    plot_runs(time_steps, run_params, imbalance_gap, "Hour", "Avg Imbalance Gap (%)", out("avg_imbalance_gap_runs.pdf"))
    plot_runs(time_steps, run_params, best_bound_gap, "Hour", "Avg Best Bound Gap (%)", out("avg_best_bound_gap_runs.pdf"))
    plot_runs(time_steps, run_params, total_costs, "Hour", "Avg Total Cost ($)", out("avg_total_costs_runs.pdf"))
    plot_runs(time_steps, run_params, battery_bal, "Hour", "Avg Battery Balance (MW)", out("avg_battery_bal_runs.pdf"))

    # Get the daily average and std imbalance gap, total costs, and renewable utilization ratio
    avg_imbalance_gap_runs  = [np.mean(np.mean(imbalance_gap[i],  axis=1)) for i in range(NUM_RUNS)]
    std_imbalance_gap_runs  = [np.std(np.mean(imbalance_gap[i],  axis=1)) for i in range(NUM_RUNS)]
    plot_runs_bar(run_params, avg_imbalance_gap_runs, "Avg Imbalance Gap (%)", out("avg_imbalance_gap_runs_bar.pdf"))

    avg_total_costs_runs  = [np.mean(np.mean(total_costs[i],  axis=1)) for i in range(NUM_RUNS)]
    std_total_costs_runs  = [np.std(np.mean(total_costs[i],  axis=1)) for i in range(NUM_RUNS)]
    plot_runs_bar(run_params, avg_total_costs_runs, "Avg Total Cost ($)", out("avg_total_costs_runs_bar.pdf"))

    avg_reur_runs  = [np.mean(reur[i]) for i in range(NUM_RUNS)]
    std_reur_runs  = [np.std(reur[i]) for i in range(NUM_RUNS)]
    plot_runs_bar(run_params, avg_reur_runs, "Avg Renewable Utilization Ratio (%)", out("avg_reur_runs_bar.pdf"))

    avg_best_cost  = np.mean([np.mean(best_cost[i]) for i in range(NUM_RUNS)])
    std_best_cost  = np.std([np.std(best_cost[i]) for i in range(NUM_RUNS)])

    avg_best_reur  = np.mean([np.mean(best_reur[i]) for i in range(NUM_RUNS)])
    std_best_reur  = np.std([np.std(best_reur[i]) for i in range(NUM_RUNS)])

    # Print a table of the average and std values for each run
    TAB_FILE = out("run_summary.csv")
    with open(TAB_FILE, "w", encoding="utf-8") as f:
        f.write("Run, Avg Imbalance Gap (%), SD Imbalance Gap (%), Avg Total Cost ($), SD Total Cost ($), Avg Renewable Utilization Ratio (%), SD Renewable Utilization Ratio (%)\n")
        for i in range(NUM_RUNS):
            f.write(f"{run_params[i]['name']},{avg_imbalance_gap_runs[i]:.2f},{std_imbalance_gap_runs[i]:.2f},{avg_total_costs_runs[i]:.2f},{std_total_costs_runs[i]:.2f},{avg_reur_runs[i]:.2f},{std_reur_runs[i]:.2f}\n")
        f.write(f"LP Benchmark, -, -, {avg_best_cost:.2f},{std_best_cost:.2f},{avg_best_reur:.2f},{std_best_reur:.2f}\n")
        end_time = time.time()
        f.write(f"Done! Total evaluation time: {end_time - start_time:.2f} seconds")

def _init_worker(num_threads):
    torch.set_num_threads(num_threads)

if __name__ == "__main__":
    # Set the training seed for reproducibility
    training_seeds = [42, 108, 456, 789, 101112]  # Example seeds for training
    n_workers = min(len(training_seeds), mp.cpu_count())
    threads_per_worker = max(1, mp.cpu_count() // n_workers)

    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(threads_per_worker,)) as pool:
        pool.map(train_and_eval, training_seeds)