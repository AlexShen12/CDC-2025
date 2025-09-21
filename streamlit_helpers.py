import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.io as pio

# --- Configuration ---
RUNS_DIR = Path("outputs/runs")

# --- Data Loading Utilities ---
@st.cache_data
def get_run_ids():
    """Enumerates all run_ids available in the outputs/runs directory."""
    return sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir()])

@st.cache_data
def load_optimal_arm_probs(run_id: str) -> pd.DataFrame:
    """Loads optimal_arm_probs.csv for a given run_id with schema validation."""
    file_path = RUNS_DIR / run_id / "optimal_arm_probs.csv"
    if not file_path.exists():
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    required_cols = ["run_id", "mechanism", "policy", "context_year", "t", "arm", "p_optimal"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Schema mismatch in {file_path}. Required columns: {required_cols}")
        return pd.DataFrame()
    
    return df

@st.cache_data
def load_arm_truths(run_id: str) -> pd.DataFrame:
    """Loads arm_truths.csv for a given run_id with schema validation."""
    file_path = RUNS_DIR / run_id / "arm_truths.csv"
    if not file_path.exists():
        st.warning(f"Arm truths file not found for run_id '{run_id}': {file_path}. Some features may be unavailable.")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    required_cols = ["run_id", "arm", "true_success_rate", "incumbent_flag"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Schema mismatch in {file_path}. Required columns: {required_cols}")
        return pd.DataFrame()
    
    return df

@st.cache_data
def load_config(run_id: str) -> dict:
    """Loads config.json for a given run_id."""
    file_path = RUNS_DIR / run_id / "config.json"
    if not file_path.exists():
        st.error(f"Config file not found: {file_path}")
        return {}
    
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

# --- Plot Builders ---
def build_optimal_prob_plot(df: pd.DataFrame, p_optimal_col: str, animation_frame_col: str = None):
    """Builds a Plotly line chart for optimal arm probabilities."""
    if df.empty:
        return px.line(title="No Data to Display")

    title_text = "Optimal Arm Probability Over Time"
    if animation_frame_col:
        title_text += " (Animated by Time)"

    fig = px.line(
        df,
        x="t",
        y=p_optimal_col,
        color="arm",
        line_group="arm",
        hover_name="arm",
        animation_frame=animation_frame_col,
        title=title_text,
        labels={
            "t": "Time Period (t)",
            p_optimal_col: "Optimal Arm Probability",
            "arm": "Arm"
        },
        height=500
    )
    fig.update_layout(hovermode="x unified")
    fig.update_traces(mode="lines+markers")
    return fig

def build_truth_bars(df: pd.DataFrame):
    """Builds a Plotly horizontal bar chart for true success rates."""
    if df.empty:
        return px.bar(title="No Data to Display")

    fig = px.bar(
        df.sort_values("true_success_rate", ascending=True),
        x="true_success_rate",
        y="arm",
        orientation="h",
        color="incumbent_flag",
        color_discrete_map={True: "#EF553B", False: "#636EFA"}, # Example colors
        title="True Success Rates by Arm",
        labels={
            "true_success_rate": "True Success Rate",
            "arm": "Arm",
            "incumbent_flag": "Incumbent"
        },
        height=400
    )
    fig.update_layout(showlegend=True)
    return fig
