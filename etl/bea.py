import pandas as pd
import numpy as np
import pathlib

def process_bea_data(input_path: str, output_dir: str) -> None:
    """
    Processes the BEA Space Economy data to extract total space economy value.

    Args:
        input_path: Path to the BEA Excel file.
        output_dir: Directory to save the output CSV.
    """
    try:
        # Per the analysis, we need Table 2 for current dollar values.
        # The header is complex. We will skip initial rows and manually construct.
        df = pd.read_excel(input_path, sheet_name="Table 2", header=None)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except ValueError as e:
        print(f"Error reading excel file: {e}")
        return

    # From manual inspection, the year headers are on row index 5
    # and the data starts at row index 6.
    header_row_index = 5
    years = pd.to_numeric(df.iloc[header_row_index], errors='coerce').dropna().astype(int).tolist()
    
    # Find the row for "Space economy"
    # It has a footnote, so we search for the start of the string.
    space_economy_row = None
    for i, row in df.iterrows():
        # Check if the second column (index 1) contains the string
        if isinstance(row.iloc[1], str) and row.iloc[1].strip().startswith("Space economy"):
            space_economy_row = i
            break
            
    if space_economy_row is None:
        print("Could not find the 'Space economy' row in Table 2.")
        return

    # Extract the values for that row
    values = df.iloc[space_economy_row].tolist()
    
    # The values align with the years, but we need to find the starting column.
    # The first value is after the industry name.
    first_value_index = -1
    for i, val in enumerate(values):
        if pd.api.types.is_number(val):
            first_value_index = i
            break
    
    if first_value_index == -1:
        print("Could not find numeric values in the space economy row.")
        return
        
    # Create a clean dataframe
    data = {
        "year": years,
        "value_added": values[first_value_index:first_value_index + len(years)]
    }
    result_df = pd.DataFrame(data)

    result_df["value_scale_usd"] = pd.to_numeric(result_df["value_added"], errors='coerce') * 1_000_000 # Values are in millions
    result_df = result_df.dropna(subset=["value_scale_usd"])
    result_df["year"] = result_df["year"].astype(int)

    result_df["segment"] = "space_economy_total"
    
    # Calculate demand weight
    total_value_sum = result_df["value_scale_usd"].sum()
    if total_value_sum > 0:
        result_df["demand_weight"] = result_df["value_scale_usd"] / total_value_sum
    else:
        result_df["demand_weight"] = 0

    output_df = result_df[["year", "segment", "value_scale_usd", "demand_weight"]]
    
    output_path = pathlib.Path(output_dir) / "bea_contexts.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")

if __name__ == "__main__":
    process_bea_data("data/Business.xlsx", "outputs")