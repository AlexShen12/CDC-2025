
import pytest
from sim.models import Provider, Payload, Context
from sim.mechanisms import ThompsonSamplingSecondPrice, UCBSecondPrice, ProbabilisticDSIC, score_payload, greedy_density_allocation

@pytest.fixture
def mock_providers():
    return [
        Provider(name="Provider A", prior_alpha=8, prior_beta=2, incumbent_flag=True),
        Provider(name="Provider B", prior_alpha=3, prior_beta=2, incumbent_flag=False)
    ]

@pytest.fixture
def mock_payloads():
    return [
        Payload(payload_id="p1", mass_kg=50, orbit="LEO", base_value_usd=325000),
        Payload(payload_id="p2", mass_kg=100, orbit="GTO", base_value_usd=650000)
    ]

@pytest.fixture
def mock_context():
    return Context(year=2023, demand_rate=1, value_scale_usd=1, avg_payloads_per_launch=2, capacity_mass_kg=1000, orbit_mix={})

def test_score_payload(mock_providers):
    provider = mock_providers[0]
    quality_estimate = 0.8
    score = score_payload(provider, quality_estimate)
    assert isinstance(score, float)
    assert score == 0.8

def test_ts_sp_auction(mock_providers, mock_payloads, mock_context):
    mechanism = ThompsonSamplingSecondPrice(mock_providers, seed=42)
    result = mechanism.run_auction(mock_payloads, mock_context, mock_context.capacity_mass_kg)
    assert "winner" in result
    assert "allocation" in result
    assert "payment" in result
    assert result["payment"] >= 0

def test_ucb_sp_auction(mock_providers, mock_payloads, mock_context):
    mechanism = UCBSecondPrice(mock_providers, seed=42)
    result = mechanism.run_auction(mock_payloads, mock_context, mock_context.capacity_mass_kg)
    assert "winner" in result
    assert "allocation" in result
    assert "payment" in result
    assert result["payment"] >= 0

def test_pl_dsic_auction(mock_providers, mock_payloads, mock_context):
    mechanism = ProbabilisticDSIC(mock_providers, seed=42)
    result = mechanism.run_auction(mock_payloads, mock_context, mock_context.capacity_mass_kg)
    assert "winner" in result
    assert "allocation" in result
    assert "payment" in result
    assert result["payment"] >= 0

def test_observe_outcome(mock_providers):
    provider = mock_providers[0]
    initial_alpha = provider.alpha
    initial_beta = provider.beta
    mechanism = ThompsonSamplingSecondPrice(mock_providers, seed=42)
    mechanism.observe_outcome(provider.name, success=True)
    assert provider.alpha == initial_alpha + 1
    mechanism.observe_outcome(provider.name, success=False)
    assert provider.beta == initial_beta + 1
