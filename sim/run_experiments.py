
import pandas as pd
import numpy as np
import json
import pathlib
from typing import List, Dict, Type

from sim.models import Context, Provider, Payload
from sim.mechanisms import AuctionMechanism, ThompsonSamplingSecondPrice, UCBSecondPrice, ProbabilisticDSIC
from sim.metrics import calculate_metrics

def load_simulation_data(data_dir: str) -> Dict:
    """Loads all necessary CSVs from the data directory."""
    try:
        return {
            "contexts": pd.read_csv(pathlib.Path(data_dir) / "contexts.csv"),
            "providers": pd.read_csv(pathlib.Path(data_dir) / "providers.csv"),
            "payloads_catalog": pd.read_csv(pathlib.Path(data_dir) / "payloads_catalog.csv"),
        }
    except FileNotFoundError as e:
        print(f"Error loading simulation data: {e}")
        return None

def run_simulation(
    num_auctions: int,
    policy: str,
    data_dir: str = "outputs",
    output_dir: str = "outputs",
    seed: int = 42,
    num_runs: int = 1, # New parameter for multiple runs
    ts_alpha: float = 1.0, ts_beta: float = 1.0,
    ucb_c: float = 2.0,
    pl_tau: float = 0.1
) -> None:
    """Main simulation orchestrator."""
    
    all_runs_metrics = []

    for run_idx in range(num_runs):
        current_seed = seed + run_idx # Use different seed for each run
        rng = np.random.default_rng(current_seed)
        
        # 1. Load data
        sim_data = load_simulation_data(data_dir)
        if not sim_data:
            return
        
        providers_df = sim_data["providers"]
        payloads_catalog = sim_data["payloads_catalog"]
        contexts_df = sim_data["contexts"]

        # 2. Initialize Providers and Latent Qualities
        providers_list = [Provider(**row) for _, row in providers_df.iterrows()]
        true_qualities = {p.name: rng.beta(p.prior_alpha, p.prior_beta) for p in providers_list}
        # print(f"True qualities generated for run {run_idx}: {true_qualities}") # Commented for cleaner output

        # 3. Select Mechanism based on Policy
        mechanism_class: Type[AuctionMechanism]
        if policy == "TS-SP":
            mechanism_class = ThompsonSamplingSecondPrice
        elif policy == "UCB-SP":
            mechanism_class = UCBSecondPrice
        elif policy == "PL-DSIC":
            mechanism_class = ProbabilisticDSIC
        else:
            raise ValueError(f"Unknown policy: {policy}")
        
        auction_mechanism = mechanism_class(providers_list, seed=current_seed)

        # 4. Simulation Loop
        results = []
        for i in range(num_auctions):
            # Sample a context for the auction
            context_row = contexts_df.sample(1, weights="demand_rate", random_state=rng).iloc[0]
            context_dict = context_row.to_dict()
            context_dict['orbit_mix'] = json.loads(context_dict['orbit_mix'])
            context = Context(**context_dict)

            # Sample payloads for the auction
            num_payloads = int(rng.poisson(context.avg_payloads_per_launch))
            if num_payloads == 0: continue
            
            payload_options = payloads_catalog[payloads_catalog["year"] <= context.year]
            if payload_options.empty:
                continue
            payload_samples = payload_options.sample(num_payloads, random_state=rng, replace=True)
            payloads = [
                Payload(
                    payload_id=f"p_{i}_{j}",
                    mass_kg=row.payload_mass_kg,
                    orbit=row.orbit,
                    base_value_usd=0 # Will be calculated inside mechanism
                ) for j, (_, row) in enumerate(payload_samples.iterrows())]

            # Run auction
            auction_result = auction_mechanism.run_auction(payloads, context, context.capacity_mass_kg)
            
            # Observe outcome
            winner_name = auction_result["winner"]
            true_winner_quality = true_qualities[winner_name]
            success = rng.random() < true_winner_quality
            auction_mechanism.observe_outcome(winner_name, success)
            
            auction_result["auction_id"] = i
            auction_result["context_year"] = context.year
            auction_result["true_winner_quality"] = true_winner_quality
            auction_result["outcome_success"] = success
            results.append(auction_result)

        # 5. Calculate and Save Metrics for this run
        metrics = calculate_metrics(results, providers_df, true_qualities)
        metrics["run_idx"] = run_idx # Add run index
        all_runs_metrics.append(metrics)
        
        # Save raw results for each run (optional, can be commented out for large num_runs)
        # results_df = metrics.pop("results_df")
        # results_output_path = pathlib.Path(output_dir) / f"results_{policy}_{current_seed}.csv"
        # results_df.to_csv(results_output_path, index=False)
        # print(f"Saved raw results for run {run_idx} to {results_output_path}")

    # Aggregate and save all runs metrics
    aggregated_metrics_output_path = pathlib.Path(output_dir) / f"aggregated_metrics_{policy}_{seed}.json"
    with open(aggregated_metrics_output_path, 'w') as f:
        json.dump(all_runs_metrics, f, indent=4)
    print(f"Saved aggregated metrics for {num_runs} runs to {aggregated_metrics_output_path}")

if __name__ == '__main__':
    # Example of how to run from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_auctions", type=int, default=1000)
    parser.add_argument("--policy", type=str, default="TS-SP", choices=["TS-SP", "UCB-SP", "PL-DSIC"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=1) # New argument
    args = parser.parse_args()

    run_simulation(
        num_auctions=args.num_auctions,
        policy=args.policy,
        seed=args.seed,
        num_runs=args.num_runs
    )
