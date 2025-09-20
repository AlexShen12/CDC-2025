
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import pathlib
import subprocess
from scipy.interpolate import griddata

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Smallsat Rideshare Simulation")

# --- Load Config ---
@st.cache_data
def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# --- App Header ---
st.title("Smallsat Rideshare Launch Slot Allocation Simulation")
st.write("""
This application simulates a marketplace for small satellite (smallsat) rideshare launch slots. 
It uses auction mechanisms augmented with multi-armed bandits to explore the trade-offs between revenue, efficiency, and market inclusion.
""")

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

policy = st.selectbox("Select Policy", list(policy_explanations.keys()))
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
    """
    Multi-Armed Bandits (MABs) are a class of online learning problems where an agent repeatedly chooses between several actions (arms) with unknown reward distributions. The goal is to maximize the cumulative reward over a sequence of trials. In our simulation, each 'arm' represents a potential launch provider or a specific auction mechanism, and the 'reward' is related to metrics like revenue or efficiency.

    The core challenge in MABs is the **exploration-exploitation trade-off**:
    - **Exploration:** Trying new arms to learn more about their reward distributions.
    - **Exploitation:** Choosing the arm currently believed to have the highest expected reward.

    Different bandit algorithms offer various strategies to balance this trade-off.
    """
)

st.subheader("Thompson Sampling (TS)")
st.markdown(
    """
    Thompson Sampling is a probabilistic MAB algorithm that chooses arms based on their probability of being optimal. It maintains a belief (a probability distribution) over the reward parameters for each arm. In each round:
    1. A random sample is drawn from the belief distribution for each arm.
    2. The arm with the highest sampled value is chosen.
    3. The belief distribution for the chosen arm is updated based on the observed reward.

    Mathematically, for a Bernoulli bandit (where rewards are binary, e.g., success/failure), if we use a Beta distribution as the prior for the success probability $p$ of each arm:
    - Prior: $p_i \sim \text{Beta}(\alpha_i, \beta_i)$
    - If arm $i$ is chosen and yields a reward (success): $\alpha_i \leftarrow \alpha_i + 1$
    - If arm $i$ is chosen and yields no reward (failure): $\beta_i \leftarrow \beta_i + 1$

    In our simulation, Thompson Sampling is applied to learn the quality of launch providers. The 'TS-SP' policy uses Thompson Sampling to decide which provider to select, aiming to identify high-quality providers efficiently. The learning curve (shown in the Results Dashboard) illustrates how the policy's choices improve over time as it gathers more evidence.
    """
)

st.subheader("Upper Confidence Bound (UCB)")
st.markdown(
    """
    The Upper Confidence Bound (UCB) algorithm is another popular MAB strategy. It selects the arm that maximizes an upper confidence bound on its expected reward. This bound is typically calculated as:

    $\text{UCB}_i = \bar{x}_i + c \sqrt{\frac{\ln N}{n_i}}$

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
    """
    The Plackett-Luce model is a probabilistic model for rankings. In the context of the `PL-DSIC` (Probabilistic DSIC) policy, it\'s likely used to model the probability of different providers being chosen or ranked based on their estimated qualities or scores. While the exact implementation details are within the simulation module, the core idea is to assign probabilities to outcomes (e.g., a provider winning an auction) based on a set of underlying parameters (e.g., provider scores or utilities).

    If $s_1, s_2, \dots, s_K$ are the scores assigned to $K$ items (e.g., providers), the probability of item $i$ being ranked first (or chosen) is given by:

    $P(\text{item } i \text{ is first}) = \frac{e^{s_i}}{\sum_{j=1}^K e^{s_j}}$

    This model allows for a probabilistic allocation mechanism, where the `pl_tau` parameter might influence the sensitivity of these probabilities to the underlying scores, effectively controlling the exploration-exploitation balance within this policy.
    """
)

st.subheader("Modern Portfolio Theory (MPT) Concepts")
st.markdown(
    """
    The simulation results are analyzed using concepts from Modern Portfolio Theory (MPT), a framework for constructing investment portfolios to maximize expected return for a given level of market risk.

    - **Expected Revenue (Return):** The average revenue generated by a policy over multiple simulation runs.
    - **Revenue Volatility (Risk):** The standard deviation of the revenue generated by a policy, representing the variability or risk associated with that policy.
    - **Efficient Frontier:** A set of optimal portfolios that offer the highest expected return for a defined level of risk or the lowest risk for a given level of expected return. In our "Revenue-Efficiency Surface" plot, this represents the optimal trade-off between risk, revenue, and inclusion rate. The surface itself is convex, illustrating that diversification (or mixing policies) can lead to better risk-adjusted outcomes.
    - **Capital Allocation Line (CAL):** A line created on a graph of all possible portfolios, representing the risk-free asset and a risky portfolio. The slope of the CAL is the Sharpe ratio.
    - **Sharpe Ratio:** A measure of risk-adjusted return. It describes how much excess return you receive for the volatility you endure for holding a riskier asset.

    $\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}$

    Where:
    - $E(R_p)$ is the expected return of the portfolio (policy).
    - $R_f$ is the risk-free rate (benchmark revenue).
    - $\sigma_p$ is the standard deviation of the portfolio's (policy's) excess return.

    A higher Sharpe ratio indicates a better risk-adjusted performance.
    """
)


st.header("Results Dashboard")
st.write(
    "We translate the simulation outputs into the language of Modern Portfolio Theory (MPT) so that each policy can be assessed in terms of risk, expected revenue, and strategic depth for the smallsat marketplace."
)

REVENUE_SCALE = 1e18
REVENUE_UNIT_LABEL = "quintillion USD"


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
    st.write(
        "Each point represents a block of auctions for one policy and shows how risk (x-axis), expected revenue (y-axis), and the share of wins captured by new entrants (z-axis) co-move. The translucent surface interpolates these points to visualise the convex opportunity set implied by the simulation."
    )

    surface_df = pd.DataFrame(surface_samples)
    fig_surface = plt.figure(figsize=(10, 7))
    ax_surface = fig_surface.add_subplot(111, projection="3d")

    if not surface_df.empty:
        for policy_name, group in surface_df.groupby("policy"):
            ax_surface.scatter(
                group["risk"],
                group["expected_return"],
                group["inclusion_rate"],
                label=policy_name,
                s=35,
            )

        if len(surface_df) >= 5:
            grid_x, grid_y = np.mgrid[
                surface_df["risk"].min() : surface_df["risk"].max() : 60j,
                surface_df["expected_return"].min() : surface_df["expected_return"].max() : 60j,
            ]
            grid_z = griddata(
                (surface_df["risk"], surface_df["expected_return"]),
                surface_df["inclusion_rate"],
                (grid_x, grid_y),
                method="linear",
            )
            if np.isfinite(grid_z).any():
                ax_surface.plot_surface(
                    grid_x,
                    grid_y,
                    np.nan_to_num(grid_z, nan=np.nanmean(surface_df["inclusion_rate"])),
                    cmap="viridis",
                    alpha=0.35,
                    linewidth=0,
                )

        ax_surface.scatter(
            tangency_row["revenue_volatility"],
            tangency_row["expected_revenue"],
            tangency_row["inclusion_rate"],
            color="gold",
            marker="*",
            s=200,
            label=f"Tangency (max Sharpe): {tangency_row['policy']}",
        )

    ax_surface.set_xlabel(f"Risk (std. dev.), {REVENUE_UNIT_LABEL}")
    ax_surface.set_ylabel(f"Expected revenue, {REVENUE_UNIT_LABEL}")
    ax_surface.set_zlabel("Entrant share of wins")
    ax_surface.set_title("Convex Opportunity Set Across Policies")
    ax_surface.view_init(elev=25, azim=-45)
    ax_surface.legend()
    st.pyplot(fig_surface)
    plots_to_save["revenue_efficiency_surface"] = fig_surface

    st.markdown(
        f"""
- **Efficient Frontier (Convex Surface):** The surface highlights the convex combinations of risk, expected revenue, and inclusion rate that the platform can achieve by mixing different policies. Its convexity implies that diversification across policies can lead to a more optimal balance of risk and return than any single policy alone. The ridge of this surface represents the efficient frontier, where no policy or combination of policies can offer a higher expected revenue for the same level of risk (and inclusion), or lower risk for the same expected revenue (and inclusion).
- **Tangency portfolio:** The gold star marks the policy with the best risk-adjusted revenue (Sharpe analogue) given a risk-free benchmark of {risk_free_rate:.3f} {REVENUE_UNIT_LABEL}. This point represents the optimal blend of policies for maximizing the Sharpe ratio.
- **Inclusion as depth:** The vertical axis tracks entrant win share, emphasising how efficient revenue profiles also relate to market access for new launch providers, adding a crucial dimension to the traditional MPT framework.
"""
    )

    st.subheader("Capital Allocation Projection")
    st.write(
        "This 2D projection mirrors the classic Capital Allocation Line (CAL). It compares policy-level risk and return, draws the CAL through the best Sharpe point, and shows how far each mechanism sits from the efficient frontier."
    )

    fig_frontier, ax_frontier = plt.subplots(figsize=(10, 6))
    ordered = policy_summary.sort_values("revenue_volatility")
    ax_frontier.plot(
        ordered["revenue_volatility"],
        ordered["expected_revenue"],
        color="steelblue",
        linestyle="--",
        label="Efficient frontier (envelope)",
    )
    ax_frontier.scatter(
        policy_summary["revenue_volatility"],
        policy_summary["expected_revenue"],
        s=120,
        c="navy",
        label="Policy outcome",
    )
    ax_frontier.axhline(risk_free_rate, color="dimgray", linestyle=":", label="Risk-free benchmark")
    ax_frontier.plot(
        [0, tangency_row["revenue_volatility"]],
        [risk_free_rate, tangency_row["expected_revenue"]],
        color="darkorange",
        linewidth=2,
        label="Capital Allocation Line",
    )
    ax_frontier.annotate(
        tangency_row["policy"],
        (tangency_row["revenue_volatility"], tangency_row["expected_revenue"]),
        textcoords="offset points",
        xytext=(10, 10),
        color="darkorange",
    )
    ax_frontier.set_xlabel(f"Risk (std. dev.), {REVENUE_UNIT_LABEL}")
    ax_frontier.set_ylabel(f"Expected revenue, {REVENUE_UNIT_LABEL}")
    ax_frontier.set_title("Risk-Return Projection with Capital Allocation Line")
    ax_frontier.grid(True, linestyle=":")
    ax_frontier.legend()
    st.pyplot(fig_frontier)
    plots_to_save["capital_allocation_projection"] = fig_frontier

    st.markdown(
        "\n".join(
            [
                "- Policies above the CAL would dominate via better risk-adjusted revenue; points below suggest room to retune exploration intensity or pricing rules.",
                "- The slope of the CAL is the implied Sharpe ratio, capturing how much additional revenue the tangency mechanism delivers per unit of volatility.",
                "- Comparing points along the frontier helps prioritise which auction design to scale when aiming for a thicker yet efficient rideshare marketplace.",
            ]
        )
    )

    st.subheader("Thickness vs. Revenue (Market Depth)")
    st.write(
        "Competitive tension is critical for keeping launch prices disciplined. We map average bid-gap thickness against expected revenue and colour the points by entrant share to connect pricing power with marketplace diversity."
    )

    fig_thickness, ax_thickness = plt.subplots(figsize=(10, 6))
    scatter = ax_thickness.scatter(
        policy_summary["market_thickness_gap"],
        policy_summary["expected_revenue"],
        c=policy_summary["inclusion_rate"],
        cmap="plasma",
        s=220,
        edgecolor="black",
    )
    for _, row in policy_summary.iterrows():
        ax_thickness.annotate(
            row["policy"],
            (row["market_thickness_gap"], row["expected_revenue"]),
            textcoords="offset points",
            xytext=(6, 6),
        )
    cbar = fig_thickness.colorbar(scatter, ax=ax_thickness)
    cbar.set_label("Entrant share of wins")
    ax_thickness.set_xlabel("Mean bid gap between top providers (market thinness)")
    ax_thickness.set_ylabel(f"Expected revenue, {REVENUE_UNIT_LABEL}")
    ax_thickness.set_title("Revenue vs. Market Thickness")
    ax_thickness.grid(True, linestyle=":")
    st.pyplot(fig_thickness)
    plots_to_save["thickness_revenue"] = fig_thickness

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

    fig_inclusion, ax_inclusion = plt.subplots(figsize=(10, 6))
    entrant_pct = policy_summary["inclusion_rate"] * 100
    incumbent_pct = 100 - entrant_pct
    ax_inclusion.bar(policy_summary["policy"], incumbent_pct, label="Incumbents", color="slategray")
    ax_inclusion.bar(
        policy_summary["policy"],
        entrant_pct,
        bottom=incumbent_pct,
        label="Entrants",
        color="mediumseagreen",
    )
    ax_inclusion.set_ylabel("Share of wins (%)")
    ax_inclusion.set_title("Provider Inclusion by Policy")
    ax_inclusion.legend()
    st.pyplot(fig_inclusion)
    plots_to_save["inclusion"] = fig_inclusion

    st.markdown(
        "\n".join(
            [
                "- Entrant win rates highlight how exploration settings influence access for emerging providers.",
                "- Monitoring the balance keeps the simulation aligned with the mission objective: a robust, open rideshare marketplace, not just short-term revenue maximisation.",
                "- Combine this with the surface plot to prioritise mechanisms that live on the frontier while still elevating new suppliers.",
            ]
        )
    )

    st.subheader("Learning Trajectory (TS-SP)")
    st.write(
        "The Thompson Sampling mechanism adapts auction by auction. We smooth the realised quality of winners to see whether the policy learns to pick high-quality launch providers faster than incumbency bias alone would allow."
    )


    current_seed = st.session_state.get("seed", config["simulation"]["seed"])
    ts_results_file = pathlib.Path("outputs") / f"results_TS-SP_{current_seed}.csv"
    if ts_results_file.exists():
        ts_results = pd.read_csv(ts_results_file)
        ts_results["winner_true_quality_rolling"] = (
            ts_results["true_winner_quality"].rolling(window=40, min_periods=20).mean()
        )

        fig_learning, ax_learning = plt.subplots(figsize=(10, 6))
        ax_learning.plot(
            ts_results["auction_id"],
            ts_results["winner_true_quality_rolling"],
            color="indigo",
            linewidth=2,
        )
        ax_learning.set_xlabel("Auction number")
        ax_learning.set_ylabel("Winner's true quality (rolling mean)")
        ax_learning.set_title("Learning Curve Proxy – Thompson Sampling")
        ax_learning.grid(True, linestyle=":")
        st.pyplot(fig_learning)
        plots_to_save["learning_curve"] = fig_learning

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

    if st.button("Save Plots"):
        output_dir = pathlib.Path("outputs")
        output_dir.mkdir(exist_ok=True)
        for name, figure in plots_to_save.items():
            figure.savefig(output_dir / f"{name}.png", bbox_inches="tight", dpi=200)
        st.success("Plots saved to outputs directory.")
