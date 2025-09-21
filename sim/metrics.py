
import numpy as np
import pandas as pd
from typing import List, Dict

def calculate_metrics(results: List[Dict], providers_df: pd.DataFrame, true_qualities: Dict[str, float]) -> Dict:
    """
    Calculates all specified metrics from a list of auction results.

    Args:
        results: A list of dictionaries, where each dict is the result of one auction.
        providers_df: DataFrame with provider information (for incumbent flag).
        true_qualities: A dictionary mapping provider names to their true (latent) quality.

    Returns:
        A dictionary of aggregated metrics.
    """
    if not results:
        return {}

    df = pd.DataFrame(results)
    
    # --- Revenue ---
    total_revenue = df["payment"].sum()

    # --- Allocative Efficiency ---
    # Oracle: who would have won with perfect information?
    def get_oracle_winner(row):
        # In this simplified model, the highest true quality provider is the oracle's choice.
        # A more complex model would involve payload values.
        return max(true_qualities, key=true_qualities.get)

    df["oracle_winner"] = df.apply(get_oracle_winner, axis=1)
    allocative_efficiency = (df["winner"] == df["oracle_winner"]).mean()

    # --- Market Thickness ---
    def get_gap(scores):
        sorted_scores = sorted(scores.values(), reverse=True)
        return sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0

    df["gap_top2"] = df["scores"].apply(get_gap)
    market_thickness_gap = df["gap_top2"].mean()

    def get_n_close(scores, delta=0.05):
        sorted_scores = sorted(scores.values(), reverse=True)
        if not sorted_scores:
            return 0
        top_score = sorted_scores[0]
        return sum(1 for s in sorted_scores if (top_score - s) / top_score <= delta)

    df["n_close"] = df["scores"].apply(get_n_close)
    market_thickness_n_close = df["n_close"].mean()

    # --- Regret ---
    # Cumulative difference vs oracle allocation
    def get_regret(row):
        winner_true_quality = true_qualities[row["winner"]]
        oracle_true_quality = true_qualities[row["oracle_winner"]]
        return oracle_true_quality - winner_true_quality

    df["regret"] = df.apply(get_regret, axis=1)
    cumulative_regret = df["regret"].sum()

    # --- Inclusion ---
    incumbent_threshold = 2009  # Align with ETL logic
    first_year_map = providers_df.set_index("name")["first_year"].to_dict()
    df["winner_first_year"] = df["winner"].map(first_year_map)
    df["winner_is_incumbent"] = df["winner_first_year"] <= incumbent_threshold
    
    entrant_wins = df[df["winner_is_incumbent"] == False].shape[0]
    total_wins = df.shape[0]
    inclusion_share_entrants = entrant_wins / total_wins if total_wins > 0 else 0

    # --- Learning Curves ---
    # The alpha/beta values are updated in the mechanism, not easily tracked here without more structure.
    # This would typically be handled in the main simulation loop.

    metrics = {
        "total_revenue": total_revenue,
        "allocative_efficiency": allocative_efficiency,
        "market_thickness_gap": market_thickness_gap,
        "market_thickness_n_close": market_thickness_n_close,
        "cumulative_regret": cumulative_regret,
        "inclusion_share_entrants": inclusion_share_entrants,
        "results_df": df # Return for more detailed plotting
    }
    
    return metrics
