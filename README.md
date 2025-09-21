# Mechanism Design Simulation Platform for Smallsat Rideshare Allocation

## Overview

This project is an interactive simulation platform designed to explore and analyze different auction mechanisms for allocating small satellite (smallsat) rideshare launch slots. The platform models a complex market where a central authority must allocate scarce launch capacity among competing payloads, each with private valuations and uncertain launch provider quality.

The primary goal is to provide a tool for researchers, policymakers, and industry stakeholders to understand the trade-offs between different allocation strategies and their impact on revenue, market efficiency, and innovation.

## Features

-   **Interactive Simulation:** Run simulations with different auction mechanisms (policies) and parameters.
-   **Data-Driven Scenarios:** The simulation is grounded in real-world data from the space economy, global launch records, and SpaceX missions.
-   **Mechanism Comparison:** Compare the performance of different auction mechanisms:
    -   Thompson Sampling + Second Price (TS-SP)
    -   Upper Confidence Bound + Second Price (UCB-SP)
    -   Probabilistic DSIC (Plackett-Luce) (PL-DSIC)
-   **Rich Visualization Dashboard:** Analyze simulation results through a variety of interactive plots based on Modern Portfolio Theory (MPT) concepts.
-   **ETL Pipeline:** A full ETL (Extract, Transform, Load) pipeline to process raw data into a format suitable for the simulation.

## System Architecture

The project is divided into three main components:

1.  **`etl/`**: The ETL pipeline that processes raw data from the `data/` directory and generates cleaned, structured data in the `outputs/` directory.
2.  **`sim/`**: The core simulation engine, containing the models for providers, payloads, and contexts, as well as the implementation of the different auction mechanisms.
3.  **`app.py`**: The main Streamlit application that provides the user interface for running simulations and visualizing the results.

## Data Sources

The simulation is powered by three main data sources located in the `data/` directory:

-   **`Business.xlsx`**: Contains economic data from the U.S. Bureau of Economic Analysis (BEA) on the value of the space economy. This is used to provide a macroeconomic context for the simulation.
-   **`Global_Space_Exploration_Dataset.csv`**: A comprehensive dataset of global space missions, used to derive information about launch providers, their first year of operation, and their historical success rates.
-   **`spacex_launch_data.csv`**: Detailed data on SpaceX launches, used to model payload characteristics, launch capacity, and orbit types.

## Getting Started

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
    Before running the simulation for the first time, you need to process the raw data. You can do this by clicking the "Re-run ETL" button in the "Data Explorer" section of the Streamlit application, or by running the script from the command line:
    ```bash
    python -m etl.build_tables
    ```

2.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    This will open the interactive dashboard in your web browser.

3.  **Run Simulations:**
    From the web interface, you can configure simulation parameters and run simulations for different policies. Click the "Run All Simulations" button to generate results.

## Simulation Details

### Auction Mechanisms

The platform implements the following auction mechanisms (policies):

-   **TS-SP (Thompson Sampling + Second Price):** A sophisticated exploration strategy where the platform samples from its belief about each provider's quality. This allows for efficient learning and high revenue.
-   **UCB-SP (Upper Confidence Bound + Second Price):** A more optimistic approach where the platform chooses providers based on their potential quality. This can lead to faster discovery of high-quality new entrants.
-   **PL-DSIC (Probabilistic DSIC - Plackett-Luce):** A probabilistic mechanism that allocates slots based on scores, providing a balance between exploration and exploitation.

### Key Metrics

The performance of each mechanism is evaluated based on the following metrics:

-   **Revenue:** The total payment collected by the central authority.
-   **Allocative Efficiency:** How well launch slots are allocated to maximize overall value.
-   **Market Thickness:** The competitiveness of the market, measured by the gap between the top two bids.
-   **Inclusion:** The share of launch slots won by new or entrant providers.
-   **Regret:** The cumulative loss in quality compared to an oracle that always knows the best provider.

## Interpreting the Dashboard

The "Results Dashboard" section of the application provides several visualizations to help you analyze the simulation outcomes:

-   **Revenue-Efficiency Surface (Efficient Frontier):** A 3D scatter plot showing the trade-offs between risk (revenue volatility), expected revenue, and entrant inclusion. The convex hull overlay highlights the efficient frontier of achievable outcomes.
-   **Time-Series Learning Curves:** This plot shows the cumulative average revenue for each policy over time, with smoothed lines and shaded error bands. It helps visualize how quickly each policy converges to its long-run performance.
-   **Thickness vs. Revenue (Market Depth):** A scatter plot that maps market thickness (bid gap) against expected revenue, with points colored by entrant share. This helps connect pricing power with marketplace diversity.
-   **Inclusion of Entrants vs. Incumbents:** A bar chart that shows the share of wins for incumbent vs. entrant providers for each policy.

## Configuration

You can customize the simulation parameters in the `config.yaml` file or through the "Simulation Configuration" section in the Streamlit app. Key parameters include:

-   `price_per_kg`: The base price per kilogram for payloads.
-   `sigma_v`: The standard deviation of the value multiplier, introducing stochasticity.
-   `ucb_c`: The exploration parameter for the UCB-SP policy.
-   `pl_tau`: The temperature parameter for the PL-DSIC policy.
-   `num_auctions`: The number of auctions to run in a simulation.
-   `num_runs`: The number of simulation runs (with different seeds) for each policy.