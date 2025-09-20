import pandas as pd
import numpy as np
import pathlib
import json

from .bea import process_bea_data
from .spacex import process_spacex_data
from .global_ds import process_global_data

def build_derived_tables(data_dir: str, output_dir: str, seed: int = 42) -> None:
    """
    Runs all ETL steps and builds the final derived tables for the simulation.

    Args:
        data_dir: Directory containing the source data files.
        output_dir: Directory to save all generated CSV files.
        seed: Random seed for reproducibility.
    """
    # Ensure output directory exists
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define paths
    bea_input = str(pathlib.Path(data_dir) / "Business.xlsx")
    spacex_input = str(pathlib.Path(data_dir) / "spacex_launch_data.csv")
    global_input = str(pathlib.Path(data_dir) / "Global_Space_Exploration_Dataset.csv")

    # Step 1: Run individual ETL processors
    print("--- Running BEA data processing ---")
    process_bea_data(bea_input, output_dir)
    
    print("\n--- Running SpaceX data processing ---")
    process_spacex_data(spacex_input, output_dir, seed)
    
    print("\n--- Running Global dataset processing ---")
    process_global_data(global_input, output_dir)

    # Step 2: Load the intermediate tables
    print("\n--- Building final contexts.csv ---")
    try:
        bea_contexts = pd.read_csv(pathlib.Path(output_dir) / "bea_contexts.csv")
        spacex_launches = pd.read_csv(pathlib.Path(output_dir) / "spacex_launches.csv")
        global_missions = pd.read_csv(pathlib.Path(output_dir) / "global_missions.csv")
    except FileNotFoundError as e:
        print(f"Error: Could not find intermediate CSV file. {e}")
        return

    # Step 3: Create the final contexts.csv
    # Get annual mission counts for demand_rate proxy
    mission_counts = global_missions.groupby("year").size().reset_index(name="mission_count")
    total_missions = mission_counts["mission_count"].sum()
    if total_missions > 0:
        mission_counts["demand_rate"] = mission_counts["mission_count"] / total_missions
    else:
        mission_counts["demand_rate"] = 0

    # Get annual SpaceX stats
    spacex_annual = spacex_launches.groupby("year").agg(
        avg_payloads_per_launch=("payload_count", "mean"),
        # Use median for capacity as it's more robust to outliers
        capacity_mass_kg=("payload_mass_total_kg", lambda x: x[spacex_launches.loc[x.index, 'is_rideshare']].median())
    ).reset_index()

    # Get orbit mix
    orbit_mix = spacex_launches.groupby(["year", "orbit"]).size().unstack(fill_value=0)
    orbit_mix_json = orbit_mix.apply(lambda x: x.to_dict(), axis=1).to_json(orient="index")
    orbit_mix_df = pd.read_json(orbit_mix_json, orient="index")
    orbit_mix_df.index.name = "year"
    orbit_mix_df = orbit_mix_df.reset_index()
    orbit_mix_df["orbit_mix"] = orbit_mix_df.apply(lambda row: json.dumps({col: int(row[col]) for col in orbit_mix.columns if col != 'year'}), axis=1)

    # Join tables
    contexts = pd.merge(mission_counts[["year", "demand_rate"]], bea_contexts[["year", "value_scale_usd"]], on="year", how="left")
    contexts = pd.merge(contexts, spacex_annual, on="year", how="left")
    contexts = pd.merge(contexts, orbit_mix_df[["year", "orbit_mix"]], on="year", how="left")

    # Fill forward and backward to handle missing data at the beginning
    contexts["value_scale_usd"] = contexts["value_scale_usd"].ffill().bfill()
    contexts["avg_payloads_per_launch"] = contexts["avg_payloads_per_launch"].ffill().bfill()
    contexts["capacity_mass_kg"] = contexts["capacity_mass_kg"].ffill().bfill()
    contexts["orbit_mix"] = contexts["orbit_mix"].ffill().bfill()

    # Final cleanup
    contexts = contexts.dropna(subset=["demand_rate"])
    contexts = contexts.sort_values("year").reset_index(drop=True)

    output_path = pathlib.Path(output_dir) / "contexts.csv"
    contexts.to_csv(output_path, index=False)
    print(f"Successfully created final contexts table at {output_path}")

if __name__ == "__main__":
    # Assumes the script is run from the project root
    build_derived_tables("data", "outputs")
