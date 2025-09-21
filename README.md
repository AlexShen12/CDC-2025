# Mechanism Design Simulation Platform for Smallsat Rideshare Allocation

## 1. Overview

This project is an interactive simulation platform for exploring and analyzing auction mechanisms for allocating small satellite (smallsat) rideshare launch slots. The platform addresses a critical challenge in **mechanism design**: how to create rules for allocating scarce resources when participants (agents) have private information and act strategically.

In the context of commercial space launches, a central authority (the launch platform) must allocate limited mass and volume capacity on rockets among many competing small satellite payloads. The problem is to design a mechanism that encourages truthful reporting of values, allocates slots efficiently, and balances revenue generation with fairness and innovation (e.g., giving new launch providers a chance to compete).

This simulation platform tackles this by:

1.  **Defining the Problem:** We model scenarios where "payloads" (small satellites with a given mass and value) must be allocated to "providers" (launch companies such as SpaceX, Rocket Lab, Firefly, or new entrants). Each provider has an uncertain reliability (probability of success), and each payload has a value proportional to its mass and orbital target. Providers "bid" implicitly through their performance and cost profiles.
2.  **Exploring Solutions (Mechanisms):** We implement and compare various **mechanisms** – sets of rules that determine how payloads are allocated and what payments are made based on the submitted bids.
3.  **Evaluating Performance:** The platform runs simulations using these mechanisms over multiple "seeds" (different random initial conditions) to assess their performance across key metrics such as revenue, efficiency, market thickness, and inclusion.

The ultimate goal is to provide a "digital twin" of the smallsat launch market, allowing for in-silico experimentation with different market designs to find the one that best achieves the desired outcomes.

## 2. System Architecture

The project is composed of several interconnected modules that work together to provide the full simulation and analysis experience.

-   **`data/`**: This directory contains the raw, source data files used as inputs for the simulation.
-   **`etl/`**: This directory contains the ETL (Extract, Transform, Load) pipeline.
    -   `build_tables.py`: The main script that orchestrates the ETL process, calling the other scripts in this directory.
    -   `bea.py`: Processes economic data from the BEA to establish a macroeconomic context.
    -   `global_ds.py`: Processes a global dataset of space missions to create a catalog of launch providers and their characteristics.
    -   `spacex.py`: Processes detailed data on SpaceX launches to model payload and launch characteristics.
-   **`sim/`**: This is the core of the simulation engine.
    -   `run_experiments.py`: The main script for running simulations from the command line.
    -   `models.py`: Defines the core data structures (Pydantic models) for `Provider`, `Payload`, and `Context`.
    -   `mechanisms.py`: Contains the implementation of the different auction mechanisms (TS-SP, UCB-SP, PL-DSIC).
    -   `metrics.py`: Contains the logic for calculating performance metrics from the simulation results.
-   **`outputs/`**: This directory is where all the generated data from the ETL and simulation processes are stored.
-   **`app.py`**: The main Streamlit application file. It provides the user interface, orchestrates the simulation runs, and visualizes the results.
-   **`config.yaml`**: A configuration file for setting simulation parameters.
-   **`requirements.txt`**: A list of all the Python packages required to run the project.

## 3. Data Sources

The simulation is grounded in three real-world datasets:

-   **`data/Business.xlsx`**: This file, from the U.S. Bureau of Economic Analysis (BEA), provides data on the value added by the U.S. space economy.
    -   **How it's used:** We extract the annual "Space economy" value-added figures from Table 2. This is used to create a `value_scale_usd` for each year in the simulation, providing a macroeconomic context that influences payload values.
-   **`data/Global_Space_Exploration_Dataset.csv`**: A comprehensive dataset of historical space missions from around the world.
    -   **How it's used:** We process this data to create a list of unique launch providers (`providers.csv`). For each provider, we determine their `first_year` of operation and their historical `success_rate`, which is used to set their prior quality belief in the simulation. The `incumbent_flag` is set based on the `first_year` relative to a defined threshold.
-   **`data/spacex_launch_data.csv`**: Detailed data on SpaceX launches.
    -   **How it's used:** This data is used to model the characteristics of a typical rideshare launch. We derive statistics like the average number of payloads per launch (`avg_payloads_per_launch`), the distribution of orbits (`orbit_mix`), and the typical mass of payloads. This is used to create the `payloads_catalog.csv` and parts of the `contexts.csv`.

## 4. Getting Started

### Prerequisites

-   Python 3.9+
-   pip

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Platform

1.  **Run the ETL Process:**
    Before running the simulation for the first time, you must process the raw data. This is a critical step that generates the necessary input files for the simulation. You can do this by clicking the **"Re-run ETL"** button in the "Data Explorer" section of the Streamlit application. This will execute the `etl/build_tables.py` script.

2.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    This will open the interactive dashboard in your web browser at `http://localhost:8501`.

3.  **Run Simulations:**
    From the web interface, you can configure simulation parameters in the "Simulation Configuration" section. Then, in the "Run Simulation" section, you can select a policy and click the **"Run All Simulations"** button to generate results for all policies. This will execute the `sim/run_experiments.py` script for each policy.

## 5. Simulation Details

### Mathematical Foundations

The simulation is built on concepts from online learning and auction theory. The core problem is framed as a **Multi-Armed Bandit (MAB)** problem, where each launch provider is an "arm" with an unknown reward distribution (their true quality or success rate). The goal of the platform (the agent) is to maximize the cumulative reward (e.g., revenue) over a series of auctions by balancing the **exploration-exploitation trade-off**:

-   **Exploration:** Choosing less-known providers to learn more about their quality.
-   **Exploitation:** Choosing the provider that is currently believed to be the best.

### Auction Mechanisms

The platform implements three different mechanisms to solve this problem:

#### TS-SP (Thompson Sampling + Second Price)

-   **How it works:** Thompson Sampling is a probabilistic algorithm that chooses an arm based on its probability of being the optimal one. It maintains a belief (a probability distribution) over the quality of each provider, modeled as a Beta distribution (`Beta(alpha, beta)`). In each auction:
    1.  A random quality sample is drawn from each provider's Beta distribution.
    2.  The provider with the highest sampled quality is chosen as the winner.
    3.  The payment is determined by a **second-price rule**: the winner pays a price related to the second-highest bid (or score). This encourages truthful bidding.
    4.  After the auction, the outcome (success or failure) is observed, and the winner's Beta distribution is updated (`alpha` is incremented for a success, `beta` for a failure).
-   **Actionable Insight:** TS is generally very effective at quickly converging to the optimal provider while still exploring other options. It provides a good balance between revenue and learning.

#### UCB-SP (Upper Confidence Bound + Second Price)

-   **How it works:** UCB is an optimistic algorithm that selects the arm with the highest *upper confidence bound* on its expected reward. The UCB score is calculated as:
    $$ \text{UCB}_i = \bar{x}_i + c \sqrt{\frac{\ln N}{n_i}} $$
    Where $\bar{x}_i$ is the current average quality of provider $i$, $N$ is the total number of auctions, $n_i$ is the number of times provider $i$ has been chosen, and $c$ is an exploration parameter. 
    The second term is the "exploration bonus," which is larger for providers that have been tried less often.
-   **Actionable Insight:** UCB is more explicitly focused on exploration than TS. It can be very effective at discovering new, high-quality providers, making it a good choice for fostering market inclusion. The `ucb_c` parameter in `config.yaml` allows you to tune the level of exploration.

#### PL-DSIC (Probabilistic DSIC - Plackett-Luce)

-   **How it works:** This mechanism uses the Plackett-Luce model, a probabilistic model for rankings, to introduce randomness into the selection process.
    1.  Each provider is assigned a score based on their mean quality.
    2.  These scores are used to generate a probability distribution over the providers:
        $$ P(\text{provider } i \text{ wins}) = \frac{e^{s_i / \tau}}{\sum_{j=1}^K e^{s_j / \tau}} $$
        Where $s_i$ is the score of provider $i$, and $\tau$ (tau) is a "temperature" parameter that controls the level of exploration. A higher `tau` leads to more random choices (more exploration).
    3.  The winner is chosen by sampling from this distribution.
-   **Actionable Insight:** This mechanism provides a tunable way to balance exploration and exploitation. It's designed to be "approximately" strategy-proof (DSIC - Dominant-Strategy Incentive Compatible), meaning providers have an incentive to bid truthfully.

### Key Metrics

-   **Revenue:** The total payment collected. The payment in each auction is based on a second-price rule, where the winner pays an amount related to the second-highest score. This is a proxy for the platform's profit.
-   **Allocative Efficiency:** Measures how often the "best" provider (the one with the highest true quality, known as the "oracle" choice) was chosen. An efficiency of 100% means the mechanism always selected the optimal provider.
-   **Market Thickness:** Measured as the average gap between the top two provider scores in each auction. A smaller gap indicates a "thicker," more competitive market.
-   **Inclusion (Entrant Share of Wins):** The percentage of auctions won by "entrant" (non-incumbent) providers. This is a critical metric for assessing market health and innovation.
-   **Regret:** The cumulative difference between the quality of the oracle choice and the quality of the chosen winner. It represents the "opportunity cost" of imperfect information and exploration. A lower cumulative regret indicates a more efficient learning process.

## 6. Interpreting the Dashboard

The dashboard provides a rich set of visualizations to analyze the simulation results.

#### Revenue-Efficiency Surface (Efficient Frontier)

-   **What it shows:** A 3D scatter plot of `Risk` (revenue volatility) vs. `Expected Revenue` vs. `Entrant Share of Wins`. Each point represents a segment of a simulation run for a specific policy.
-   **How it works:** The plot is generated using Plotly's `go.Scatter3d`. A `go.Mesh3d` trace is added to show the convex hull of all the points. The convex hull represents the "efficient frontier" – the set of optimal, non-dominated outcomes.
-   **Actionable Insights:**
    -   **Identify Dominant Policies:** Policies that lie on the surface of the convex hull are "efficient." They offer the best possible trade-off for a given level of risk, revenue, or inclusion.
    -   **Identify Dominated Policies:** Policies that lie inside the hull are "dominated," meaning there is another policy on the surface that is better in at least one dimension without being worse in others.
    -   **Understand Trade-offs:** The shape of the hull reveals the fundamental trade-offs. For example, you might see that increasing entrant inclusion comes at the cost of lower revenue or higher risk.

#### Time-Series Learning Curves

-   **What it shows:** The cumulative average revenue for each policy over the course of the simulation.
-   **How it works:** For each policy, the plot shows the mean cumulative average revenue across all simulation runs (seeds). The line is smoothed using a rolling average to make the trend clearer. The shaded error band represents one standard deviation of the cumulative average revenue across the runs, indicating the stability of the policy's performance. The plot is created using Plotly's `go.Scatter` with `fill='tonexty'` for the error bands.
-   **Actionable Insights:**
    -   **Convergence Speed:** See how quickly each policy learns and converges to its long-run performance level. A steeper curve indicates faster learning.
    -   **Long-Run Performance:** The level at which the curve flattens out shows the long-run expected revenue for that policy. This helps identify which policy is best in the long run.
    -   **Performance Variability:** A wider error band indicates that the policy's performance is more variable and less predictable across different scenarios (seeds).

#### Thickness vs. Revenue (Market Depth)

-   **What it shows:** A 2D scatter plot of `Market Thickness` (mean bid gap) vs. `Expected Revenue`. The color of each point represents the `Inclusion Rate`.
-   **How it works:** This plot is a `px.scatter` plot.
-   **Actionable Insights:**
    -   **Market Competitiveness:** Points on the left (smaller bid gap) represent thicker, more competitive markets.
    -   **Revenue and Competition:** Analyze the relationship between market competition and revenue. Does higher competition lead to higher or lower revenue?
    -   **Inclusion and Competition:** The color of the points shows how entrant inclusion relates to market thickness and revenue.

#### Inclusion of Entrants vs. Incumbents

-   **What it shows:** A diverging bar chart showing the percentage of wins for incumbent vs. entrant providers for each policy.
-   **How it works:** This is a `px.bar` plot with a horizontal orientation.
-   **Actionable Insights:**
    -   **Market Openness:** Directly compare how "friendly" each policy is to new entrants.
    -   **Balancing Act:** A good policy should reward reliable incumbents while still providing opportunities for new entrants. This plot makes it easy to see if a policy is overly biased towards one group.

## 7. Configuration

The `config.yaml` file allows you to configure the simulation without changing the code.

-   **`value_scaling`**:
    -   `price_per_kg`: The base price per kilogram for payloads. Used in the `get_payload_value` function.
    -   `sigma_v`: The standard deviation of the value multiplier. A higher value introduces more randomness into payload valuations.
-   **`bandits`**:
    -   `ucb_c`: The exploration parameter for the UCB-SP policy. Higher values encourage more exploration.
    -   `pl_tau`: The temperature parameter for the PL-DSIC policy. Higher values lead to more random exploration.
-   **`simulation`**:
    -   `num_auctions`: The number of auctions (time steps) to run in a single simulation.
    -   `num_runs`: The number of times to run the simulation for each policy, each with a different random seed. This is crucial for getting statistically significant results and for generating the error bands in the learning curve plot.
    -   `seed`: The starting random seed for reproducibility.
