import streamlit as st
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pathlib
import subprocess
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import plotly.express as px
import plotly.graph_objects as go
import streamlit_helpers

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(layout="wide")

st.title("Mechanism Design Simulation Platform")

st.markdown("""
## Introduction: Optimizing Smallsat Rideshare Launch Slot Allocation

This platform addresses a critical challenge in **mechanism design** – how to create rules for allocating scarce resources when participants (agents) have private information and act strategically. In the context of commercial space launches, a central authority (the launch platform) must allocate limited mass and volume capacity on rockets among many competing small satellite payloads. The problem is to design a mechanism that encourages truthful reporting of values, allocates slots efficiently, and balances revenue generation with fairness and innovation (e.g., giving new launch providers a chance to compete).

Our simulation platform tackles this by:
1.  **Defining the Problem:** We model scenarios where "payloads" (small satellites with a given mass and value) must be allocated to "providers" (launch companies such as SpaceX, Rocket Lab, Firefly, or new entrants). Each provider has an uncertain reliability (probability of success), and each payload has a value proportional to its mass and orbital target. Providers "bid" implicitly through their performance and cost profiles.
2.  **Exploring Solutions (Mechanisms):** We implement and compare various **mechanisms** – sets of rules that determine how payloads are allocated and what payments are made based on the submitted bids. Examples include:
    *   **TS-SP (Thompson Sampling + Second Price):** A sophisticated exploration strategy where the platform samples from its belief about each provider's quality. This allows for efficient learning and high revenue.
    *   **UCB-SP (Upper Confidence Bound + Second Price):** A more optimistic approach where the platform chooses providers based on their potential quality. This can lead to faster discovery of high-quality new entrants.
    *   **PL-DSIC (Probabilistic DSIC - Plackett-Luce):** A probabilistic mechanism that allocates slots based on scores, providing a balance between exploration and exploitation, designed to incentivize truthful bidding at the individual payload level.
3.  **Evaluating Performance:** The platform runs simulations using these mechanisms over multiple "seeds" (different random initial conditions) to assess their performance across key metrics such as:
    *   **Revenue:** The total payment collected by the central authority from allocated launch slots.
    *   **Allocative Efficiency:** How well launch slots are allocated to maximize overall value, ensuring payloads that value the slots most receive them.
    *   **Market Thickness:** The competitiveness of the market, measured by the gap between the top two bids, indicating a healthy supply-demand dynamic.
    *   **Inclusion:** The share of launch slots won by new or entrant providers, fostering a diverse and innovative market.
    *   **Regret:** The cumulative loss in quality compared to an oracle that always knows the best provider, indicating how much potential value was missed due to imperfect learning.

### Key Terms Explained:

*   **Payloads:** The small satellites or tasks requiring launch services, each with specific mass, volume, and value.
*   **Providers:** The launch companies (incumbents or new entrants) bidding for the opportunity to carry payloads, each with varying reliability and cost structures.
*   **Bids:** Offers submitted by providers, representing their private costs or valuations for carrying specific payloads. In our bandit models, these are often implicit through learned quality.
*   **Mechanisms:** The rules governing how launch slots are allocated to payloads and what payments are made by the winning providers.
*   **Revenue:** The total money collected from providers for allocated launch slots.
*   **Allocative Efficiency:** A measure of how effectively payloads are matched with providers to maximize overall value, ensuring the 'right' payload gets the 'right' slot.
*   **Market Thickness:** An indicator of market competitiveness. A smaller gap between top bids suggests a thicker, more competitive market.
*   **Inclusion:** The proportion of launch slots awarded to new or entrant providers, reflecting market openness and support for innovation.
*   **Regret:** A measure of the performance difference between the chosen mechanism and an ideal (oracle) mechanism that always makes the optimal choice.
*   **Seeds:** Different random starting points for simulations, used to ensure robustness and generalize results across various market conditions.
"""
)

st.sidebar.title("Configuration")


# --- Section 1: Data Explorer ---
with st.expander("Data Explorer", expanded=False):
    st.header("Data Check & Profiling")
    st.write("This section shows the status of the generated data files that feed the simulation.")

    output_files = [
        "bea_contexts.csv", "contexts.csv", "global_missions.csv", 
        "payloads_catalog.csv", "providers.csv", "spacex_launches.csv"
    ]

    for f in output_files:
        path = pathlib.Path("outputs") / f
        if path.exists():
            st.success(f"{f} is present.")
        else:
            st.error(f"{f} is missing.")

    if st.button("Re-run ETL"):
        with st.spinner("Running ETL..."):
            result = subprocess.run(["python", "etl/build_tables.py"], capture_output=True, text=True)
            st.code(result.stdout)
            st.code(result.stderr)
            st.success("ETL process finished.")

    st.subheader("Data Previews")
    selected_file = st.selectbox("Select a file to preview", output_files)
    if (pathlib.Path("outputs") / selected_file).exists():
        df = pd.read_csv(pathlib.Path("outputs") / selected_file)
        st.dataframe(df.head())
        st.write(f"Null counts:")
        st.write(df.isnull().sum())

# --- Section 2: Simulation Configuration ---
with st.expander("Simulation Configuration", expanded=False):
    st.header("Configuration")
    st.write("Adjust simulation parameters here.")

    # Value Scaling
    st.subheader("Value Scaling")
    price_per_kg = st.number_input("Price per kg", value=config['value_scaling']['price_per_kg'])
    sigma_v = st.number_input("Sigma V", value=config['value_scaling']['sigma_v'])

    # Bandit Knobs
    st.subheader("Bandit Parameters")
    ucb_c = st.number_input("UCB c", value=config['bandits']['ucb_c'])
    pl_tau = st.number_input("PL Tau", value=config['bandits']['pl_tau'])

    # Simulation
    st.subheader("Simulation Settings")
    num_auctions = st.number_input("Number of Auctions", value=config['simulation']['num_auctions'])
    num_runs = st.number_input("Number of Runs per Policy", value=config['simulation']['num_runs'])
    seed = st.number_input("Random Seed", value=config['simulation']['seed'])

    if st.button("Save Configuration"):
        # Logic to save to session state and yaml would go here
        st.session_state['num_runs'] = num_runs
        st.session_state['seed'] = seed
        st.success("Configuration saved.")

# --- Section 3: Run Simulation ---
st.header("Run Simulation")
st.write("Select an auction policy and run the simulation. Each policy represents a different strategy for the platform to learn about provider quality.")

policy_explanations = {
    "TS-SP": "**Thompson Sampling + Second Price:** A sophisticated exploration strategy where the platform samples from its belief about each provider's quality. This allows for efficient learning and high revenue.",
    "UCB-SP": "**Upper Confidence Bound + Second Price:** A more optimistic approach where the platform chooses providers based on their potential quality. This can lead to faster discovery of high-quality new entrants.",
    "PL-DSIC": "**Probabilistic DSIC (Plackett-Luce):** A probabilistic mechanism that allocates slots based on scores, providing a balance between exploration and exploitation."
}

policy_options = list(policy_explanations.keys())
policy_display_names = {short_name: explanation.split(':')[0].strip('* ') for short_name, explanation in policy_explanations.items()}
selected_policy_short_name = st.selectbox(
    "Select Policy",
    options=policy_options,
    format_func=lambda x: policy_display_names[x]
)
policy = selected_policy_short_name # Use the short name for internal logic
st.markdown(policy_explanations[policy])

if st.button("Run All Simulations"):
    current_num_runs = st.session_state.get('num_runs', config['simulation']['num_runs'])
    current_seed = st.session_state.get('seed', config['simulation']['seed'])
    for p in policy_explanations.keys():
        with st.spinner(f"Running simulation for {p} ({current_num_runs} runs)..."):
            cmd = ["python", "-m", "sim.run_experiments", "--policy", p, "--seed", str(current_seed), "--num_runs", str(current_num_runs)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"Simulation for {p} finished.")
            else:
                st.error(f"Simulation for {p} failed.")
                st.code(result.stderr)


# --- Section 4: Mathematical Foundations ---
st.header("Mathematical Foundations")
st.write("This section details the mathematical models and algorithms underpinning the simulation.")

st.subheader("Multi-Armed Bandits (MABs)")
st.markdown(
    r"""
    Multi-Armed Bandits (MABs) are a class of online learning problems where an agent repeatedly chooses between several actions (arms) with unknown reward distributions. The goal is to maximize the cumulative reward over a sequence of trials. In our simulation, each 'arm' represents a potential launch provider or a specific auction mechanism, and the 'reward' is related to metrics like revenue or efficiency.

    The core challenge in MABs is the **exploration-exploitation trade-off**:
    - **Exploration:** Trying new arms to learn more about their reward distributions.
    - **Exploitation:** Choosing the arm currently believed to have the highest expected reward.

    Different bandit algorithms offer various strategies to balance this trade-off.
    """
)

st.subheader("Thompson Sampling (TS)")
st.markdown(
    r"""
    Thompson Sampling is a probabilistic MAB algorithm that chooses arms based on their probability of being optimal. It maintains a belief (a probability distribution) over the reward parameters for each arm. In each round:
    1. A random sample is drawn from the belief distribution for each arm.
    2. The arm with the highest sampled value is chosen.
    3. The belief distribution for the chosen arm is updated based on the observed reward.

    Mathematically, for a Bernoulli bandit (where rewards are binary, e.g., success/failure), if we use a Beta distribution as the prior for the success probability $p$ of each arm:
    - Prior: $p_i \sim \text{Beta}(\alpha_i, \beta_i)$
    - If arm $i$ is chosen and yields a reward (success): $\alpha_i \leftarrow \alpha_i + 1$
    - If arm $i$ is chosen and yields no reward (failure): $\beta_i \leftarrow \beta_i + 1$$

    In our simulation, Thompson Sampling is applied to learn the quality of launch providers. The 'TS-SP' policy uses Thompson Sampling to decide which provider to select, aiming to identify high-quality providers efficiently. The learning curve (shown in the Results Dashboard) illustrates how the policy's choices improve over time as it gathers more evidence.
    """
)

st.subheader("Upper Confidence Bound (UCB)")
st.markdown(
    r"""
    The Upper Confidence Bound (UCB) algorithm is another popular MAB strategy. It selects the arm that maximizes an upper confidence bound on its expected reward. This bound is typically calculated as:

    $$\text{UCB}_i = \bar{x}_i + c \sqrt{\frac{\ln N}{n_i}}$$

    Where:
    - $\bar{x}_i$ is the empirical mean reward of arm $i$.
    - $n_i$ is the number of times arm $i$ has been played.
    - $N$ is the total number of rounds played.
    - $c$ is an exploration parameter (configured as `ucb_c` in the simulation) that balances exploration and exploitation. A higher $c$ encourages more exploration.

    The `UCB-SP` policy uses this principle to explore providers, giving preference to those with higher potential rewards, even if their current average reward is not the highest.
    """
)

st.subheader("Plackett-Luce Model")
st.markdown(
    r"""
    The Plackett-Luce model is a probabilistic model for rankings. In the context of the `PL-DSIC` (Probabilistic DSIC) policy, it\'s likely used to model the probability of different providers being chosen or ranked based on their estimated qualities or scores. While the exact implementation details are within the simulation module, the core idea is to assign probabilities to outcomes (e.g., a provider winning an auction) based on a set of underlying parameters (e.g., provider scores or utilities).

    If $s_1, s_2, \dots, s_K$ are the scores assigned to $K$ items (e.g., providers), the probability of item $i$ being ranked first (or chosen) is given by:

    $$P(\text{item } i \text{ is first}) = \frac{e^{s_i}}{\sum_{j=1}^K e^{s_j}}$$

    This model allows for a probabilistic allocation mechanism, where the `pl_tau` parameter might influence the sensitivity of these probabilities to the underlying scores, effectively controlling the exploration-exploitation balance within this policy.
    """
)

st.subheader("Modern Portfolio Theory (MPT) Concepts")
st.markdown(
    r"""
    The simulation results are analyzed using concepts from Modern Portfolio Theory (MPT), a framework for constructing investment portfolios to maximize expected return for a given level of market risk.

    - **Expected Revenue (Return):** The average revenue generated by a policy over multiple simulation runs.
    - **Revenue Volatility (Risk):** The standard deviation of the revenue generated by a policy, representing the variability or risk associated with that policy.
    - **Efficient Frontier:** A set of optimal portfolios that offer the highest expected return for a defined level of risk or the lowest risk for a given level of expected return. In our "Revenue-Efficiency Surface" plot, this represents the optimal trade-off between risk, revenue, and inclusion rate. The surface itself is convex, illustrating that diversification (or mixing policies) can lead to better risk-adjusted outcomes.
    - **Capital Allocation Line (CAL):** A line created on a graph of all possible portfolios, representing the risk-free asset and a risky portfolio. The slope of the CAL is the Sharpe ratio.
    - **Sharpe Ratio:** A measure of risk-adjusted return. It describes how much excess return you receive for the volatility you endure for holding a riskier asset.

    $$\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}$$

    Where:
    - $E(R_p)$ is the expected return of the portfolio (policy).
    - $R_f$ is the risk-free rate (benchmark revenue).
    - $\sigma_p$ is the standard deviation of the portfolio\'s (policy\'s) excess return.

    A higher Sharpe ratio indicates a better risk-adjusted performance.
    """
)


st.header("Results Dashboard")
st.write(
    "We translate the simulation outputs into the language of Modern Portfolio Theory (MPT) so that each policy can be assessed in terms of risk, expected revenue, and strategic depth for the smallsat marketplace."
)

REVENUE_SCALE = 1e6
REVENUE_UNIT_LABEL = "million USD"


def _coerce_boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False})


def load_policy_runs(policy_name: str, segments: int = 12):
    summaries = []
    surface_points = []
    metrics_files = sorted(pathlib.Path("outputs").glob(f"metrics_{policy_name}_*.json"))
    for metrics_file in metrics_files:
        run_id = metrics_file.stem.split("_")[-1]
        with open(metrics_file, "r") as f:
            metrics_payload = json.load(f)
        results_path = pathlib.Path("outputs") / f"results_{policy_name}_{run_id}.csv"
        if not results_path.exists():
            continue
        df = pd.read_csv(results_path)
        df["payment"] = df["payment"].astype(float)
        if "winner_is_incumbent" in df.columns:
            df["winner_is_incumbent"] = _coerce_boolean(df["winner_is_incumbent"]).fillna(True)
        else:
            df["winner_is_incumbent"] = True
        if "regret" in df.columns:
            df["regret"] = df["regret"].astype(float)
        else:
            df["regret"] = 0.0

        expected_revenue = df["payment"].mean() / REVENUE_SCALE
        revenue_volatility = df["payment"].std(ddof=0) / REVENUE_SCALE
        allocative_efficiency = (df["winner"] == df["oracle_winner"]).mean()
        thickness_gap = df["gap_top2"].mean() if "gap_top2" in df.columns else np.nan
        inclusion_rate = 1.0 - df["winner_is_incumbent"].mean()
        regret_total = df["regret"].sum()

        summaries.append(
            {
                "policy": policy_name,
                "run_id": run_id,
                "expected_revenue": expected_revenue,
                "revenue_volatility": revenue_volatility,
                "allocative_efficiency": allocative_efficiency,
                "market_thickness_gap": thickness_gap,
                "inclusion_rate": inclusion_rate,
                "regret": regret_total,
                "total_revenue_raw": metrics_payload.get("total_revenue"),
            }
        )

        split_segments = np.array_split(df, segments)
        for seg_index, seg_df in enumerate(split_segments):
            if len(seg_df) < 2:
                continue
            seg_payments = seg_df["payment"].astype(float)
            seg_incumbent = seg_df["winner_is_incumbent"].mean()
            seg_gap = seg_df["gap_top2"].mean() if "gap_top2" in seg_df else np.nan
            surface_points.append(
                {
                    "policy": policy_name,
                    "segment": seg_index,
                    "risk": seg_payments.std(ddof=0) / REVENUE_SCALE,
                    "expected_return": seg_payments.mean() / REVENUE_SCALE,
                    "inclusion_rate": 1.0 - seg_incumbent,
                    "market_thickness_gap": seg_gap,
                }
            )
    return summaries, surface_points


def load_learning_curve_data(policy_explanations):
    learning_curve_data = {}
    for policy_name in policy_explanations.keys():
        all_runs_dfs = []
        results_files = sorted(pathlib.Path("outputs").glob(f"results_{policy_name}_*.csv"))
        if not results_files:
            continue

        for results_file in results_files:
            df = pd.read_csv(results_file)
            df["payment"] = df["payment"].astype(float)
            df['cum_avg_revenue'] = df['payment'].expanding().mean() / REVENUE_SCALE
            all_runs_dfs.append(df[['auction_id', 'cum_avg_revenue']])

        if not all_runs_dfs:
            continue

        # Concatenate all runs for the policy
        concat_df = pd.concat(all_runs_dfs)
        
        # Calculate mean and std dev across runs
        learning_curve = concat_df.groupby('auction_id')['cum_avg_revenue'].agg(['mean', 'std']).reset_index()
        
        # Apply a rolling average to smooth the mean curve
        learning_curve['mean_smoothed'] = learning_curve['mean'].rolling(window=50, min_periods=1).mean()
        
        learning_curve_data[policy_name] = learning_curve
        
    return learning_curve_data



all_run_summaries = []
surface_samples = []
for policy_name in policy_explanations:
    summaries, samples = load_policy_runs(policy_name)
    all_run_summaries.extend(summaries)
    surface_samples.extend(samples)

if not all_run_summaries:
    st.warning("No simulation outputs were detected. Run the experiments to populate the dashboards.")
else:
    summary_df = pd.DataFrame(all_run_summaries)
    policy_summary = (
        summary_df.groupby("policy")
        .agg(
            expected_revenue=("expected_revenue", "mean"),
            revenue_volatility=("revenue_volatility", "mean"),
            allocative_efficiency=("allocative_efficiency", "mean"),
            market_thickness_gap=("market_thickness_gap", "mean"),
            inclusion_rate=("inclusion_rate", "mean"),
            regret=("regret", "mean"),
        )
        .reset_index()
    )
    # Add full policy names
    policy_summary['full_policy_name'] = policy_summary['policy'].map(policy_display_names)

    default_rf = float(max(0.0, summary_df["expected_revenue"].min() * 0.25))
    risk_free_rate = st.number_input(
        "Risk-free benchmark revenue per auction (quintillion USD)",
        value=round(default_rf, 3),
        min_value=0.0,
        step=0.05,
    )
    policy_summary["sharpe_like"] = (policy_summary["expected_revenue"] - risk_free_rate) / (
        policy_summary["revenue_volatility"] + 1e-9
    )
    tangency_row = policy_summary.loc[policy_summary["sharpe_like"].idxmax()]

    plots_to_save = {}

    st.subheader("Revenue-Efficiency Surface (Efficient Frontier)")
    st.markdown(
        """
        This 3D scatter plot visualizes the **Convex Opportunity Set** across different simulation policies.
        Each point on the plot represents a segment of auction outcomes for a specific policy,
        showing the interplay between:
        - **Risk (x-axis):** The standard deviation of revenue, measured in quintillion USD.
        - **Expected Revenue (y-axis):** The average revenue generated, also in quintillion USD.
        - **Entrant Share of Wins (z-axis):** The proportion of wins captured by new entrants, indicating market inclusion.

        The **translucent surface** interpolates these points, revealing the **convex volatility surface**.
        This surface represents the set of all achievable combinations of risk, revenue, and inclusion.
        Its convexity is a key insight from Modern Portfolio Theory, suggesting that diversification
        (or mixing different policies) can lead to better risk-adjusted outcomes.

        **What is saved:** The underlying data for this plot comes from `results_<policy_name>_<run_id>.csv` files,
        which contain detailed auction-level outcomes (payments, winners, etc.), and `metrics_<policy_name>_<run_id>.json`
        files, which provide aggregated metrics. These raw simulation outputs are processed to calculate
        segment-wise risk, expected return, and inclusion rates.

        **What is displayed:** The plot displays these calculated segment-wise metrics. The `color` and `symbol`
        of each point differentiate between policies. The interactive nature allows for exploration of this
        multi-dimensional trade-off space.
        """
    )

    # Convert surface_samples to DataFrame for plotting
    surface_df = pd.DataFrame(surface_samples)

    if not surface_df.empty:
        fig_volatility_surface = go.Figure()

        # Add scatter plot
        for policy in surface_df['policy'].unique():
            policy_df = surface_df[surface_df['policy'] == policy]
            fig_volatility_surface.add_trace(go.Scatter3d(
                x=policy_df["risk"],
                y=policy_df["expected_return"],
                z=policy_df["inclusion_rate"],
                mode='markers',
                marker=dict(
                    size=5,
                    symbol='circle'
                ),
                name=policy
            ))

        # Add convex hull
        hull_points = surface_df[['risk', 'expected_return', 'inclusion_rate']].values
        if len(hull_points) >= 4:
            hull = ConvexHull(hull_points)
            fig_volatility_surface.add_trace(go.Mesh3d(
                x=hull_points[:, 0],
                y=hull_points[:, 1],
                z=hull_points[:, 2],
                i=hull.simplices[:, 0],
                j=hull.simplices[:, 1],
                k=hull.simplices[:, 2],
                opacity=0.1,
                color='gray',
                name='Convex Hull'
            ))

        fig_volatility_surface.update_layout(
            title="Convex Volatility Surface (Risk, Revenue, Inclusion)",
            scene=dict(
                xaxis_title=f"Risk (std. dev.), {REVENUE_UNIT_LABEL}",
                yaxis_title=f"Expected Revenue, {REVENUE_UNIT_LABEL}",
                zaxis_title="Entrant Share of Wins",
            ),
            height=700
        )

        st.plotly_chart(fig_volatility_surface, use_container_width=True)
        plots_to_save["volatility_surface"] = fig_volatility_surface
    else:
        st.warning("No data available to plot the Convex Volatility Surface. Please run simulations.")

    st.subheader("Time-Series Learning Curves")
    st.markdown(
        """
        This plot shows the cumulative average revenue over simulation rounds for each policy.
        The solid line represents the smoothed mean performance across all runs (seeds), and the shaded area
        represents the variability (one standard deviation). This helps visualize how quickly each
        policy converges to its long-run performance and how stable that performance is.
        """
    )

    learning_curve_data = load_learning_curve_data(policy_explanations)

    if learning_curve_data:
        fig_learning_curve = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, (policy, data) in enumerate(learning_curve_data.items()):
            color = colors[i % len(colors)]
            
            fig_learning_curve.add_trace(go.Scatter(
                x=data['auction_id'],
                y=data['mean_smoothed'] + data['std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name=f"{policy} upper",
            ))
            fig_learning_curve.add_trace(go.Scatter(
                x=data['auction_id'],
                y=data['mean_smoothed'] - data['std'],
                mode='lines',
                line=dict(width=0),
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                fill='tonexty',
                showlegend=False,
                name=f"{policy} lower",
            ))
            fig_learning_curve.add_trace(go.Scatter(
                x=data['auction_id'],
                y=data['mean_smoothed'],
                mode='lines',
                line=dict(color=color),
                name=policy_display_names[policy]
            ))

        fig_learning_curve.update_layout(
            title="Cumulative Average Revenue Learning Curves",
            xaxis_title="Simulation Round (Auction ID)",
            yaxis_title=f"Cumulative Average Revenue, {REVENUE_UNIT_LABEL}",
            hovermode="x unified"
        )
        st.plotly_chart(fig_learning_curve, use_container_width=True)
        plots_to_save["learning_curves"] = fig_learning_curve
    else:
        st.warning("No data available to plot learning curves. Please run simulations.")

    st.subheader("Thickness vs. Revenue (Market Depth)")
    st.write(
        "Competitive tension is critical for keeping launch prices disciplined. We map average bid-gap thickness against expected revenue and colour the points by entrant share to connect pricing power with marketplace diversity."
    )

    fig_thickness = px.scatter(
        policy_summary,
        x="market_thickness_gap",
        y="expected_revenue",
        color="inclusion_rate",
        size_max=20,
        hover_name="full_policy_name",
        title="Revenue vs. Market Thickness",
        labels={
            "market_thickness_gap": "Mean bid gap between top providers (market thinness)",
            "expected_revenue": f"Expected revenue, {REVENUE_UNIT_LABEL}",
            "inclusion_rate": "Entrant Share of Wins",
        },
        color_continuous_scale=px.colors.sequential.Plasma,
    )
    fig_thickness.update_traces(marker=dict(size=15, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig_thickness)
    # plots_to_save["thickness_revenue"] = fig_thickness

    st.markdown(
        "\n".join(
            [
                "- Points toward the left indicate thicker markets with tighter bid gaps, often correlating with healthier entrant participation.",
                "- Policies clustering high on the y-axis but with wide bid gaps risk over-reliance on incumbents, inviting future fragility.",
                "- The colour gradient shows that raising entrant inclusion does not always trade off with revenue—important for mission resilience.",
            ]
        )
    )

    st.subheader("Inclusion of Entrants vs. Incumbents")
    st.write(
        "We split win rates between incumbents and entrants to check whether each policy keeps the rideshare market open to newcomers while still rewarding reliable providers."
    )

    # Reshape data for stacked bar chart
    inclusion_df = policy_summary[["policy", "inclusion_rate", "full_policy_name"]].copy() # Include full_policy_name
    inclusion_df["entrant_pct"] = inclusion_df["inclusion_rate"] * 100
    inclusion_df["incumbent_pct"] = 100 - inclusion_df["entrant_pct"]

    # Melt the DataFrame to have 'category' and 'percentage' columns for stacking
    melted_inclusion_df = inclusion_df.melt(
        id_vars=["policy", "full_policy_name"], # Include full_policy_name in id_vars
        value_vars=["incumbent_pct", "entrant_pct"],
        var_name="Provider Type",
        value_name="Share of Wins (%)",
    )
    melted_inclusion_df["Provider Type"] = melted_inclusion_df["Provider Type"].map(
        {"incumbent_pct": "Incumbents", "entrant_pct": "Entrants"}
    )

    # --- New: Transform for Diverging Bar Chart ---
    # Make Incumbent percentages negative for diverging bar chart
    melted_inclusion_df.loc[melted_inclusion_df['Provider Type'] == 'Incumbents', 'Share of Wins (%)'] *= -1

    fig_inclusion = px.bar(
        melted_inclusion_df,
        x="Share of Wins (%)", # X-axis is now the percentage
        y="full_policy_name", # Y-axis is now the policy
        color="Provider Type",
        orientation="h", # Horizontal bars
        title="Provider Inclusion by Policy (Entrants vs. Incumbents)",
        labels={
            "Share of Wins (%)": "Share of wins (%)",
            "full_policy_name": "Policy",
            "Provider Type": "Provider Type"
        },
        color_discrete_map={"Incumbents": "slategray", "Entrants": "mediumseagreen"},
        barmode='relative' # Important for diverging bars
    )

    # Customize layout for diverging bars
    fig_inclusion.update_layout(
        xaxis_title="Share of Wins (%)",
        yaxis_title="Policy",
        bargap=0.1,
        xaxis_tickprefix="", # Remove default negative sign for incumbent side
        xaxis_tickformat=".0f",
        hovermode="y unified"
    )
    # Add a vertical line at 0 for clarity
    fig_inclusion.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")

    st.plotly_chart(fig_inclusion)
    # plots_to_save["inclusion"] = fig_inclusion

    st.markdown(
        "\n".join(
            [
                "- Entrant win rates highlight how exploration settings influence access for emerging providers.",
                "- Monitoring the balance keeps the simulation aligned with the mission objective: a robust, open rideshare marketplace, not just short-term revenue maximisation.",
                "- Combine this with the surface plot to prioritise mechanisms that live on the frontier while still elevating new suppliers.",
            ]
        )
    )

    st.subheader(f"Learning Trajectory ({policy_display_names['TS-SP']})")
    st.write(
        "The Thompson Sampling mechanism adapts auction by auction. We smooth the realised quality of winners to see whether the policy learns to pick high-quality launch providers faster than incumbency bias alone would allow."
    )


    current_seed = st.session_state.get("seed", config["simulation"]['seed'])
    ts_results_file = pathlib.Path("outputs") / f"results_TS-SP_{current_seed}.csv"
    if ts_results_file.exists():
        ts_results = pd.read_csv(ts_results_file)
        ts_results["winner_true_quality_rolling"] = (
            ts_results["true_winner_quality"].rolling(window=40, min_periods=20).mean()
        )

        fig_learning = px.line(
            ts_results,
            x="auction_id",
            y="winner_true_quality_rolling",
            title="Learning Curve Proxy – Thompson Sampling",
            labels={
                "auction_id": "Auction number",
                "winner_true_quality_rolling": "Winner's true quality (rolling mean)",
            },
            line_dash_sequence=["solid"],
            color_discrete_sequence=["indigo"],
        )
        fig_learning.update_traces(line=dict(width=2))
        st.plotly_chart(fig_learning)
        # plots_to_save["learning_curve"] = fig_learning

        st.markdown(
            f"""
            - The upward trend confirms the policy is reallocating mass toward higher-quality launchers as evidence accumulates.
            - Dips reveal exploration bursts; aligning them with market thickness events can uncover when the platform intentionally probes new providers.
            - A plateau would signal the need for retuning (e.g., tighter priors or dynamic confidence scaling).
            """
        )
    else:
        st.info(
            "Raw results for TS-SP not found for the current seed. Re-run the simulation to analyse the learning trajectory."
        )

    st.header("Optimal Arm Probabilities Analysis")
    st.write("Visualize the probability of each arm being optimal over time, allowing for detailed analysis of policy learning and performance.")
    st.markdown(
        r"""
        ### What are Optimal Arm Probabilities?
        In the context of Multi-Armed Bandit (MAB) problems, an "optimal arm" is the arm (e.g., a launch provider or auction mechanism) that yields the highest true expected reward. Since these true rewards are unknown, policies must learn them through exploration.

        The **Optimal Arm Probability ($p_{optimal}$)** for a given arm represents the probability that this arm is, in fact, the true optimal arm among all available arms, given the data observed so far. This metric is crucial for understanding how confident a policy is in its assessment of each arm's quality. A higher $p_{optimal}$ indicates greater confidence that the arm is indeed the best.

        ### The Math Behind $p_{optimal}$
        The calculation of $p_{optimal}$ typically involves Bayesian inference, especially when using algorithms like Thompson Sampling. For each arm $i$, we maintain a belief about its true success rate (or reward distribution). If we assume a Bernoulli reward distribution (success/failure) and use a Beta distribution as a conjugate prior for the success probability $p_i$, then after observing $\alpha_i-1$ successes and $\beta_i-1$ failures, the posterior distribution for $p_i$ is $\text{Beta}(\alpha_i, \beta_i)$.

        To find the probability that arm $i$ is optimal, we need to calculate:
        $P(\text{arm } i \text{ is optimal}) = P(p_i > p_j \text{ for all } j \neq i)$

        This is often computed using Monte Carlo simulation:
        1.  **Sample from Posteriors:** For each arm $j$, draw a random sample $s_j$ from its current posterior distribution, $\text{Beta}(\alpha_j, \beta_j)$. This sample represents a plausible true success rate for arm $j$.
        2.  **Identify Best Arm:** Find the arm $k$ that has the highest sampled value $s_k$.
        3.  **Repeat:** Repeat steps 1 and 2 many times (e.g., 1,000 or 10,000 times).
        4.  **Estimate $p_{optimal}$:** The $p_{optimal}$ for arm $i$ is the proportion of simulations in which arm $i$ was identified as the best arm.

        This process provides a dynamic measure of how likely each arm is to be the best, reflecting the policy's learning and uncertainty over time.
        """
    )

    # Sidebar for this section
    st.sidebar.header("Optimal Arm Probabilities Controls")
    selected_run_id_opt_arm = st.sidebar.selectbox("Select Run ID for Optimal Arms", streamlit_helpers.get_run_ids(), key="opt_arm_run_id")

    if selected_run_id_opt_arm:
        optimal_probs_df = streamlit_helpers.load_optimal_arm_probs(selected_run_id_opt_arm)
        arm_truths_df = streamlit_helpers.load_arm_truths(selected_run_id_opt_arm)
        config_data = streamlit_helpers.load_config(selected_run_id_opt_arm)

        if optimal_probs_df.empty:
            st.warning("No optimal arm probability data available for the selected run ID.")
        else:
            st.write(f"Displaying data for Run ID: **{selected_run_id_opt_arm}**")
            st.write("Configuration:", config_data)

            # --- Sidebar Filters for Optimal Arm Probabilities ---
            st.sidebar.subheader("Optimal Arm Data Filters")

            all_mechanisms = optimal_probs_df['mechanism'].unique()
            selected_mechanisms = st.sidebar.multiselect(
                "Select Mechanism(s)",
                options=all_mechanisms,
                default=list(all_mechanisms),
                key="opt_arm_mechanisms"
            )

            all_policies = optimal_probs_df['policy'].unique()
            selected_policies = st.sidebar.multiselect(
                "Select Policy(ies)",
                options=all_policies,
                default=list(all_policies),
                key="opt_arm_policies"
            )

            all_context_years = optimal_probs_df['context_year'].unique()
            selected_context_years = st.sidebar.multiselect(
                "Select Context Year(s)",
                options=all_context_years,
                default=list(all_context_years),
                key="opt_arm_context_years"
            )

            # Time window slider
            min_t = int(optimal_probs_df['t'].min())
            max_t = int(optimal_probs_df['t'].max())
            selected_t_range = st.sidebar.slider(
                "Time Window (t)",
                min_value=min_t,
                max_value=max_t,
                value=(min_t, max_t),
                key="opt_arm_t_range"
            )

            # Smoothing option
            st.sidebar.subheader("Optimal Arm Smoothing Options")
            enable_smoothing = st.sidebar.checkbox("Enable Smoothing (Rolling Mean)", value=False, key="opt_arm_smoothing")
            smoothing_window = 0
            if enable_smoothing:
                smoothing_window = st.sidebar.number_input(
                    "Smoothing Window Size (periods)",
                    min_value=1,
                    max_value=max_t - min_t + 1,
                    value=3,
                    step=1,
                    key="opt_arm_smoothing_window"
                )

            # Animation option
            enable_animation = st.sidebar.checkbox("Enable Animation (by time 't')", value=False, key="opt_arm_animation")

            # Arms filter and Top K Arms
            st.sidebar.subheader("Optimal Arm Selection")
            show_top_k_arms = st.sidebar.checkbox("Show Top K Arms", value=False, key="opt_arm_show_top_k")
            top_k_value = 0
            if show_top_k_arms:
                top_k_value = st.sidebar.number_input(
                    "Number of Top Arms to Show (K)",
                    min_value=1,
                    value=min(5, len(optimal_probs_df['arm'].unique())),
                    step=1,
                    key="opt_arm_top_k_value"
                )
                selected_arms = []
            else:
                all_arms = optimal_probs_df['arm'].unique()
                selected_arms = st.sidebar.multiselect(
                    "Select Arm(s)",
                    options=all_arms,
                    default=list(all_arms),
                    key="opt_arm_selected_arms"
                )

            # --- Data Filtering ---
            filtered_df = optimal_probs_df[
                (optimal_probs_df['mechanism'].isin(selected_mechanisms)) &
                (optimal_probs_df['policy'].isin(selected_policies)) &
                (optimal_probs_df['context_year'].isin(selected_context_years)) &
                (optimal_probs_df['t'] >= selected_t_range[0]) &
                (optimal_probs_df['t'] <= selected_t_range[1])
            ].copy()

            # Handle Top K Arms
            if show_top_k_arms and not filtered_df.empty:
                arm_avg_p_optimal = filtered_df.groupby('arm')['p_optimal'].mean().nlargest(top_k_value).index.tolist()
                filtered_df = filtered_df[filtered_df['arm'].isin(arm_avg_p_optimal)]
                selected_arms = arm_avg_p_optimal
            elif not show_top_k_arms and selected_arms:
                filtered_df = filtered_df[filtered_df['arm'].isin(selected_arms)]
            elif not selected_arms and not show_top_k_arms:
                st.warning("No arms selected. Please select arms or enable 'Show Top K Arms'.")
                filtered_df = pd.DataFrame()

            if filtered_df.empty:
                st.warning("No data to display after applying filters.")
            else:
                # --- Data Transformations ---
                if enable_smoothing:
                    filtered_df['p_optimal_smoothed'] = filtered_df.groupby(['mechanism', 'policy', 'context_year', 'arm'])['p_optimal'].transform(
                        lambda x: x.rolling(window=smoothing_window, min_periods=1).mean()
                    )
                    p_optimal_col = 'p_optimal_smoothed'
                else:
                    p_optimal_col = 'p_optimal'

                if enable_animation:
                    filtered_df['t_str'] = filtered_df['t'].astype(str)
                    animation_frame_col = 't_str'
                else:
                    animation_frame_col = None

                st.subheader("Optimal Arm Probability Over Time")
                optimal_prob_fig = streamlit_helpers.build_optimal_prob_plot(filtered_df, p_optimal_col, animation_frame_col)
                st.plotly_chart(optimal_prob_fig, use_container_width=True)

                st.markdown(
                    """
                    ### Interpreting the Optimal Arm Probability Plot: Concrete Insights

                    This plot is crucial for understanding the learning dynamics of the simulation policies. Here's what to look for and how to derive actionable insights:

                    *   **Convergence to 1.0:** When an arm's $p_{optimal}$ converges rapidly to 1.0, it indicates that the policy has quickly identified this arm as the most likely true optimal launch provider.
                        *   **Insight:** Policies that achieve rapid convergence for truly optimal arms are efficient learners. This is desirable for quickly identifying the best launch providers in a dynamic rideshare marketplace.
                        *   **Applicable Insight:** If a new launch provider consistently shows a high $p_{optimal}$ early on, it suggests the system is effectively identifying high-potential entrants. This could inform decisions on early-stage contracts, preferential slot allocation, or marketing efforts to foster their growth and market participation.

                    *   **Divergence and Fluctuations:** If multiple arms (launch providers) maintain significant $p_{optimal}$ values for an extended period, or if probabilities fluctuate wildly, it suggests high uncertainty or that the policy is still actively exploring the capabilities of these providers.
                        *   **Insight:** High uncertainty might mean the launch providers are genuinely very close in performance, or the policy is struggling to differentiate them due to noise in launch outcomes or insufficient data.
                        *   **Applicable Insight:** Persistent fluctuations for a critical launch route or provider might signal a need to gather more data (e.g., run more simulated auctions, or conduct more detailed due diligence in real-world scenarios) or to refine the policy's exploration strategy. It could also highlight areas where true performance differences are subtle, requiring more sophisticated differentiation or risk assessment before committing to long-term contracts.

                    *   **Comparison Across Mechanisms/Policies:** Observe how $p_{optimal}$ curves differ between selected mechanisms (e.g., TS-SP vs. UCB-SP) in identifying optimal launch providers.
                        *   **Insight:** Some policies might be more aggressive explorers (e.g., UCB-SP might maintain higher $p_{optimal}$ for more launch providers initially), while others might converge more smoothly (e.g., TS-SP).
                        *   **Applicable Insight:** If a policy like UCB-SP shows a broader distribution of $p_{optimal}$ across more launch providers, it suggests it's better at discovering new, potentially optimal, providers. This is valuable in nascent or rapidly evolving markets where innovation and new entrants are critical for long-term market health. Conversely, if TS-SP converges faster to a single optimal launch provider, it might be preferred in stable markets where exploiting known good options and minimizing risk are paramount.

                    *   **Impact of Context (e.g., `context_year`):** If `context_year` is a variable, analyze how $p_{optimal}$ changes across different years for various launch providers.
                        *   **Insight:** This can reveal how the optimality of launch providers shifts with changing market conditions, technological advancements, or the competitive landscape of the space industry.
                        *   **Applicable Insight:** If a launch provider's $p_{optimal}$ drops significantly in a later `context_year`, it might indicate a decline in their relative performance, reliability issues, or the emergence of stronger competitors. This provides a data-driven signal for re-evaluating provider contracts, adjusting slot allocation strategies, or even considering new partnerships.

                    *   **Identifying Sub-optimal Arms:** Launch providers whose $p_{optimal}$ consistently stays low or quickly drops to zero are likely sub-optimal for the given context.
                        *   **Insight:** The policy has learned that these launch providers are not the best performers in terms of the defined reward (e.g., reliability, cost-effectiveness).
                        *   **Applicable Insight:** This can inform decisions to de-prioritize certain launch providers for specific missions, adjust their pricing in future auctions, or even remove them from consideration for critical payloads, thereby optimizing resource utilization and mission success rates.
                    """
                )

                st.subheader("True Success Rates")
                if not arm_truths_df.empty:
                    arms_in_filtered_df = filtered_df['arm'].unique()
                    filtered_arm_truths_df = arm_truths_df[arm_truths_df['arm'].isin(arms_in_filtered_df)]

                    if filtered_arm_truths_df.empty:
                        st.warning("No true success rate data available for the selected arms.")
                    else:
                        truth_bars_fig = streamlit_helpers.build_truth_bars(filtered_arm_truths_df)
                        st.plotly_chart(truth_bars_fig, use_container_width=True)

                        st.markdown(
                            """
                            ### Interpreting True Success Rates in Conjunction with Optimal Arm Probabilities

                            This plot displays the *actual* underlying success rates (or true qualities) of each arm. Comparing these true values with the `Optimal Arm Probability Over Time` plot provides critical insights into the effectiveness of the learning policy:

                            *   **Policy Accuracy:** Ideally, the launch providers with the highest true success rates should eventually have their $p_{optimal}$ converge to 1.0 in the `Optimal Arm Probability Over Time` plot.
                                *   **Insight:** If a policy consistently assigns high $p_{optimal}$ to a launch provider with a genuinely high true success rate, it demonstrates the policy's accuracy and ability to identify superior options for smallsat launches.
                                *   **Applicable Insight:** This confirms that the chosen mechanism is effectively learning and exploiting the best available launch providers. It provides confidence in using the policy for real-world slot allocation, especially when the true qualities (e.g., reliability, on-time performance) are initially unknown. This can justify long-term contracts or increased allocation to such providers.

                            *   **Misjudgment or Slow Learning:** If a launch provider with a high true success rate never achieves a high $p_{optimal}$, or if a provider with a low true success rate maintains a high $p_{optimal}$ for too long, it indicates a potential misjudgment or slow learning by the policy.
                                *   **Insight:** This could be due to insufficient exploration of that provider, a noisy environment (e.g., inconsistent launch data), or a policy that is not well-suited to differentiate between providers with similar performance profiles.
                                *   **Applicable Insight:** Such discrepancies highlight areas for policy refinement. For instance, if a truly high-quality launch provider is consistently overlooked, it might suggest increasing the exploration parameter (`ucb_c` for UCB, or adjusting priors for TS) to give such providers more chances to prove their worth. Conversely, if a low-quality provider is over-explored, it points to inefficient learning that wastes valuable launch slots and potentially risks mission success.

                            *   **Impact of Exploration-Exploitation Trade-off:** Observe how policies balance exploring launch providers with unknown but potentially high true success rates versus exploiting providers already known to have good performance.
                                *   **Insight:** An effective policy should quickly identify and exploit launch providers with high true success rates while still exploring others to ensure no truly optimal provider is missed.
                                *   **Applicable Insight:** If the `Optimal Arm Probability Over Time` plot shows rapid convergence to a sub-optimal launch provider (as revealed by the `True Success Rates`), it suggests the policy is exploiting too early. This might necessitate adjusting simulation parameters to favor more exploration, especially in dynamic environments where new, better launch options might emerge or existing providers improve their capabilities.

                            *   **Robustness to Noise:** In scenarios with high variance in launch outcomes (e.g., occasional delays or partial failures), a good policy should still be able to discern the true optimal launch providers.
                                *   **Insight:** Compare the $p_{optimal}$ curves against true success rates in simulations with varying levels of noise. A robust policy will still align $p_{optimal}$ with true success rates despite the noise.
                                *   **Applicable Insight:** If a policy struggles in noisy environments, it might be too sensitive to short-term fluctuations in launch performance. This could lead to suboptimal decisions in real-world applications where launch data is rarely perfectly clean. Consider policies or parameters that incorporate more robust statistical methods or longer-term averaging to smooth out noise and identify underlying true performance.
                            """
                        )
                else:
                    st.info("Arm truths data not available for this run.")

                st.subheader("Metrics Snapshots")
                if not filtered_df.empty:
                    avg_p_optimal = filtered_df[p_optimal_col].mean()
                    max_p_optimal = filtered_df[p_optimal_col].max()
                    min_p_optimal = filtered_df[p_optimal_col].min()

                    st.metric(label="Average Optimal Arm Probability", value=f"{avg_p_optimal:.3f}")
                    st.metric(label="Max Optimal Arm Probability", value=f"{max_p_optimal:.3f}")
                    st.metric(label="Min Optimal Arm Probability", value=f"{min_p_optimal:.3f}")
                else:
                    st.info("No data to calculate metrics.")
    else:
        st.info("Please select a run ID from the sidebar for Optimal Arm Probabilities Analysis.")

    if st.button("Save Plots"):
        output_dir = pathlib.Path("outputs")
        output_dir.mkdir(exist_ok=True)
        for name, figure in plots_to_save.items():
            figure.savefig(output_dir / f"{name}.png", bbox_inches="tight", dpi=200)
        st.success("Plots saved to outputs directory.")

st.header("Conclusion & Actionable Insights for Smallsat Rideshare Marketplaces")
st.markdown(
    """
    Our simulation study provides a robust framework for understanding the complex dynamics of smallsat rideshare launch slot allocation. By integrating macroeconomic data, micro-level mission data, and advanced mechanism design with multi-armed bandit algorithms, we've generated several actionable insights for designing efficient and robust marketplaces for space launch services.

    ### Key Takeaways and Recommendations:

    1.  **Balancing Revenue and Risk (from Revenue-Efficiency Surface):**
        *   **Insight:** The "Revenue-Efficiency Surface" (Convex Volatility Surface) clearly illustrates the trade-offs between expected revenue, revenue volatility (risk), and entrant inclusion. No single mechanism perfectly optimizes all three simultaneously.
        *   **Actionable Insight:** Launch platforms should use this surface to identify their optimal operating point based on strategic priorities. If maximizing revenue with acceptable risk is paramount, mechanisms closer to the "efficient frontier" are preferred. If fostering market diversity (inclusion) is a key objective, a slight reduction in peak revenue might be acceptable for a significant gain in entrant share. This suggests a need for dynamic mechanism selection or hybrid approaches that can adapt to evolving market goals.

    2.  **Fostering Market Thickness and Competitiveness (from Thickness vs. Revenue):**
        *   **Insight:** The "Thickness vs. Revenue" plot demonstrates the critical link between market competitiveness (bid gap) and revenue generation. Thicker markets, characterized by smaller bid gaps, often correlate with healthier entrant participation and can lead to more sustainable revenue streams.
        *   **Actionable Insight:** Policies that encourage a smaller bid gap (i.e., higher market thickness) should be prioritized. This might involve mechanisms that reduce information asymmetry, lower barriers to entry for new providers, or actively promote competition among existing ones. A platform should monitor this metric closely; a widening bid gap could signal declining competition and potential market fragility.

    3.  **Promoting Entrant Inclusion for Market Health (from Inclusion of Entrants):**
        *   **Insight:** The "Inclusion of Entrants" analysis highlights how different policies impact the participation of new launch providers. A healthy balance between incumbent and entrant wins is crucial for long-term market innovation and resilience.
        *   **Actionable Insight:** If a platform aims to stimulate innovation and prevent market monopolization, it should favor mechanisms that demonstrate higher entrant inclusion rates, even if they don't yield the absolute highest short-term revenue. This could involve setting aside a percentage of slots for new providers or using mechanisms (like UCB-SP) that inherently favor exploration of less-known entities.

    4.  **Optimizing Learning and Provider Selection (from Optimal Arm Probabilities Analysis):**
        *   **Insight:** The "Optimal Arm Probability Over Time" and "True Success Rates" plots provide deep insights into how effectively a mechanism learns the true quality of launch providers. Rapid convergence to the true optimal provider indicates efficient learning.
        *   **Actionable Insight:**
            *   **For Platform Operators:** Continuously monitor the $p_{optimal}$ curves. If a truly high-quality new entrant is consistently overlooked (low $p_{optimal}$ despite high true success rate), consider adjusting exploration parameters or even temporarily boosting their visibility to ensure fair evaluation. Conversely, if a low-quality provider maintains a high $p_{optimal}$ due to insufficient data, increase exploration for that arm.
            *   **For New Entrants:** Understanding these learning dynamics can help new launch providers strategically position themselves. Demonstrating consistent reliability and transparent performance data can accelerate their $p_{optimal}$ convergence, leading to increased slot allocations.
            *   **For Policy Makers:** The analysis can inform regulations that encourage data sharing or standardized performance metrics, which would enhance the learning efficiency of all market mechanisms and foster a more transparent and competitive environment.

    ### Future Directions:

    This simulation serves as a foundational tool. Future work could involve:
    *   Integrating more complex bidding behaviors and strategic interactions among providers.
    *   Modeling dynamic payload demand and varying launch window constraints.
    *   Exploring multi-objective optimization techniques to explicitly balance revenue, efficiency, and inclusion.
    *   Validating simulation results against real-world launch data as it becomes more accessible.

    By continuously refining these models and leveraging the insights gained, stakeholders can design more effective and resilient marketplaces for the burgeoning smallsat rideshare industry.
    """
)
