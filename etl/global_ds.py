
import pandas as pd
import numpy as np
import pathlib

PROVIDER_KEYWORDS = {
    "SpaceX": "SpaceX",
    "Arianespace": "Arianespace",
    "Rocket Lab": "Rocket Lab",
    "Roscosmos": "Roscosmos",
    "ULA": "ULA",
    "United Launch Alliance": "ULA",
    "Blue Origin": "Blue Origin",
    "Northrop Grumman": "Northrop Grumman",
    "CASC": "CASC"
}

def infer_provider(row) -> str:
    mission_name = str(row["Mission Name"]).lower()
    for keyword, provider in PROVIDER_KEYWORDS.items():
        if keyword.lower() in mission_name:
            return provider
    return str(row["Country"])

def process_global_data(input_path: str, output_dir: str, incumbent_threshold: int = 2009) -> None:
    """
    Processes the Global Space Exploration Dataset to create a list of missions and providers.

    Args:
        input_path: Path to the Global_Space_Exploration_Dataset.csv file.
        output_dir: Directory to save the output CSVs.
        incumbent_threshold: Year to define incumbents.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    df.columns = df.columns.str.strip()

    # Basic cleaning
    df["year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Infer provider
    df["provider"] = df.apply(infer_provider, axis=1)

    # Clean numeric columns
    df["budget_billion"] = pd.to_numeric(df["Budget (in Billion $)"], errors="coerce")
    df["success_rate"] = pd.to_numeric(df["Success Rate (%)"], errors="coerce") / 100.0
    df["success_rate"] = df["success_rate"].clip(0, 1)

    # Select and save global_missions.csv
    missions_df = df[["year", "provider", "Mission Type", "budget_billion", "success_rate", "Country"]]
    missions_df = missions_df.rename(columns={"Mission Type": "mission_type", "Country": "country"})
    
    output_path_missions = pathlib.Path(output_dir) / "global_missions.csv"
    output_path_missions.parent.mkdir(parents=True, exist_ok=True)
    missions_df.to_csv(output_path_missions, index=False)
    print(f"Successfully created {output_path_missions}")

    # Create providers.csv
    provider_info = df.groupby("provider").agg(
        first_year=("year", "min"),
        prior_mean_quality=("success_rate", "mean")
    ).reset_index()

    # Add SpaceX if not present
    if "SpaceX" not in provider_info["provider"].unique():
        spacex_row = pd.DataFrame([{"provider": "SpaceX", "first_year": 2010, "prior_mean_quality": 0.95}]) # Default values
        provider_info = pd.concat([provider_info, spacex_row], ignore_index=True)

    provider_info["prior_mean_quality"] = provider_info["prior_mean_quality"].fillna(0.7) # Default prior
    provider_info["incumbent_flag"] = provider_info["first_year"] <= incumbent_threshold

    # Assign prior strength based on incumbency
    provider_info["prior_strength"] = provider_info["incumbent_flag"].apply(lambda x: 10 if x else 5) # Alpha+Beta
    # Beta distribution parameters (alpha, beta)
    provider_info["prior_alpha"] = provider_info["prior_mean_quality"] * provider_info["prior_strength"]
    provider_info["prior_beta"] = (1 - provider_info["prior_mean_quality"]) * provider_info["prior_strength"]

    provider_info["notes"] = "Derived from Global Space Exploration Dataset"
    
    # Final selection for providers.csv
    providers_df = provider_info[[
        "provider", "first_year", "incumbent_flag", "prior_mean_quality", 
        "prior_alpha", "prior_beta", "prior_strength", "notes"
    ]]

    output_path_providers = pathlib.Path(output_dir) / "providers.csv"
    providers_df.to_csv(output_path_providers, index=False)
    print(f"Successfully created {output_path_providers}")

if __name__ == "__main__":
    process_global_data("data/Global_Space_Exploration_Dataset.csv", "outputs")
