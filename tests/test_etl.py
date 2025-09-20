
import pytest
import pandas as pd
import pathlib
from etl.build_tables import build_derived_tables

@pytest.fixture(scope="module")
def etl_output():
    """Runs the ETL process once for the test module."""
    data_dir = "data"
    output_dir = "outputs"
    build_derived_tables(data_dir, output_dir)
    return output_dir

def test_output_files_exist(etl_output):
    output_dir = pathlib.Path(etl_output)
    expected_files = [
        "bea_contexts.csv", "contexts.csv", "global_missions.csv",
        "payloads_catalog.csv", "providers.csv", "spacex_launches.csv"
    ]
    for f in expected_files:
        assert (output_dir / f).exists(), f"{f} was not created"

def test_contexts_schema(etl_output):
    contexts_path = pathlib.Path(etl_output) / "contexts.csv"
    df = pd.read_csv(contexts_path)
    expected_cols = ["year", "demand_rate", "value_scale_usd", "avg_payloads_per_launch", "capacity_mass_kg", "orbit_mix"]
    assert all(col in df.columns for col in expected_cols)
    assert not df.isnull().values.any(), "Found null values in contexts.csv"

def test_providers_schema(etl_output):
    providers_path = pathlib.Path(etl_output) / "providers.csv"
    df = pd.read_csv(providers_path)
    expected_cols = ["provider", "first_year", "incumbent_flag", "prior_mean_quality", "prior_alpha", "prior_beta"]
    assert all(col in df.columns for col in expected_cols)

def test_payloads_catalog_schema(etl_output):
    payloads_path = pathlib.Path(etl_output) / "payloads_catalog.csv"
    df = pd.read_csv(payloads_path)
    expected_cols = ["year", "orbit", "payload_mass_kg"]
    assert all(col in df.columns for col in expected_cols)
    assert (df["payload_mass_kg"] >= 0).all(), "Found negative payload mass"
