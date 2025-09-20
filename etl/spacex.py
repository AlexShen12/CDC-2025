
import pandas as pd
import numpy as np
import pathlib

def normalize_orbit(orbit: str) -> str:
    """Normalizes orbit strings."""
    if not isinstance(orbit, str):
        return "Other"
    orbit_lower = orbit.lower()
    if "leo" in orbit_lower and "iss" in orbit_lower:
        return "ISS"
    if "leo" in orbit_lower:
        return "LEO"
    if "sso" in orbit_lower:
        return "SSO"
    if "gto" in orbit_lower:
        return "GTO"
    return "Other"

def process_spacex_data(input_path: str, output_dir: str, seed: int = 42) -> None:
    """
    Processes SpaceX launch data to create a cleaned launch list and a payload catalog.

    Args:
        input_path: Path to the spacex_launch_data.csv file.
        output_dir: Directory to save the output CSVs.
        seed: Random seed for any sampling.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # Clean column names
    df.columns = df.columns.str.strip()

    # Date processing
    df["date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year

    # Payload mass cleaning
    df["payload_mass_total_kg"] = df["Payload Mass (kg)"].astype(str).str.replace(r'\s|,', '', regex=True)
    df["payload_mass_total_kg"] = pd.to_numeric(df["payload_mass_total_kg"], errors="coerce")

    # Infer payload count and rideshare status
    df["payload_count"] = df["Payload"].astype(str).str.count(",") + 1
    df["is_rideshare"] = (df["payload_count"] > 1) | (df["Payload"].str.contains("Transporter", na=False))

    # Normalize orbit
    df["orbit"] = df["Orbit"].apply(normalize_orbit)

    # Normalize mission outcome
    df["mission_outcome"] = df["Mission Outcome"].apply(lambda x: "Success" if isinstance(x, str) and "Success" in x else "Failure")

    # Create launch_id
    df["launch_id"] = df["Flight Number"].astype(str)

    # Select and save spacex_launches.csv
    launches_df = df[[
        "launch_id", "date", "year", "orbit", "payload_count", 
        "payload_mass_total_kg", "mission_outcome", "is_rideshare"
    ]]
    
    output_path_launches = pathlib.Path(output_dir) / "spacex_launches.csv"
    output_path_launches.parent.mkdir(parents=True, exist_ok=True)
    launches_df.to_csv(output_path_launches, index=False)
    print(f"Successfully created {output_path_launches}")

    # Create and save payloads_catalog.csv
    # Filter for launches with known mass and at least one payload
    catalog_df = df[df["payload_mass_total_kg"].notna() & (df["payload_count"] > 0)].copy()
    catalog_df["payload_mass_kg"] = catalog_df["payload_mass_total_kg"] / catalog_df["payload_count"]

    payloads_catalog = catalog_df[["year", "orbit", "payload_mass_kg"]]
    
    output_path_catalog = pathlib.Path(output_dir) / "payloads_catalog.csv"
    payloads_catalog.to_csv(output_path_catalog, index=False)
    print(f"Successfully created {output_path_catalog}")

if __name__ == "__main__":
    process_spacex_data("data/spacex_launch_data.csv", "outputs")
