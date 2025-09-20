
import numpy as np
from typing import List, Dict, Tuple, Protocol
from abc import ABC, abstractmethod

from sim.models import Provider, Payload, Context

# --- Value and Scoring Models ---

def get_payload_value(payload: Payload, context: Context, price_per_kg: float, sigma_v: float) -> float:
    """Calculates the value for a payload with some stochasticity."""
    base_price = payload.mass_kg * price_per_kg * context.value_scale_usd
    # Introduce stochasticity based on a normal distribution
    epsilon = np.random.normal(0, sigma_v)
    value = base_price * (1 + epsilon)
    return max(0, value)

def score_payload(provider: Provider, quality_estimate: float) -> float:
    """Simple score: provider quality."""
    # The value is incorporated later in the allocation logic (value/kg)
    return quality_estimate

# --- Allocation Strategy ---

def greedy_density_allocation(payloads: List[Payload], providers: List[Provider], scores: Dict[str, float], capacity_mass_kg: float, context: Context) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Allocates payloads to providers based on score/kg density."""
    allocation = {}
    # We have one slot per provider to fill
    provider_slots = {provider.name: capacity_mass_kg for provider in providers}
    
    # Sort payloads by value/mass ratio
    payloads_sorted = sorted(payloads, key=lambda p: get_payload_value(p, context, 6500, 0.2) / p.mass_kg, reverse=True)

    # Assign payloads to best provider available
    for payload in payloads_sorted:
        best_provider = None
        best_score = -1
        for provider in providers:
            # Check if provider has capacity and is better than current best
            if provider_slots[provider.name] >= payload.mass_kg and scores[provider.name] > best_score:
                best_provider = provider
                best_score = scores[provider.name]
        
        if best_provider:
            allocation[payload.payload_id] = best_provider.name
            provider_slots[best_provider.name] -= payload.mass_kg

    # This is a simplified allocation for multiple payloads to one provider, 
    # a full knapsack would be more complex. This is a greedy fill.
    # For this simulation, we will simplify to one winner provider per auction.
    winner_provider_name = max(scores, key=scores.get)
    allocated_payloads = {}
    current_mass = 0
    for p in payloads_sorted:
        if current_mass + p.mass_kg <= capacity_mass_kg:
            allocated_payloads[p.payload_id] = winner_provider_name
            current_mass += p.mass_kg

    return allocated_payloads, scores

# --- Auction Mechanism Protocol & ABC ---

class AuctionMechanism(ABC):
    """Abstract base class for an auction mechanism."""
    def __init__(self, providers: List[Provider], seed: int = 42):
        self.providers = {p.name: p for p in providers}
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def run_auction(self, payloads: List[Payload], context: Context, capacity_mass_kg: float) -> Dict:
        """Run a single auction round."""
        pass

    def observe_outcome(self, winner_name: str, success: bool):
        """Update provider beliefs based on the outcome."""
        provider = self.providers[winner_name]
        if success:
            provider.alpha += 1
        else:
            provider.beta += 1

# --- Bandit Implementations ---

class ThompsonSamplingSecondPrice(AuctionMechanism):
    """Thompson Sampling with a Second-Price rule."""
    def run_auction(self, payloads: List[Payload], context: Context, capacity_mass_kg: float) -> Dict:
        # 1. Sample quality estimates from the Beta distribution
        quality_samples = {name: self.rng.beta(p.alpha, p.beta) for name, p in self.providers.items()}
        
        # 2. Score providers
        scores = {name: score_payload(p, quality_samples[name]) for name, p in self.providers.items()}
        
        # 3. Determine winner
        winner_name = max(scores, key=scores.get)
        winner_score = scores[winner_name]
        
        # 4. Determine second price
        second_highest_score = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        
        # 5. Allocate payloads (greedy knapsack)
        # For simplicity, we assume the winner takes a bundle of payloads up to capacity
        allocated_payloads, _ = greedy_density_allocation(payloads, list(self.providers.values()), scores, capacity_mass_kg, context)
        
        # Price is based on the second-highest score.
        # Total value of allocated payloads
        total_value = sum(get_payload_value(p, context, 6500, 0.2) for p in payloads if p.payload_id in allocated_payloads)
        payment = second_highest_score * total_value

        return {
            "winner": winner_name,
            "allocation": allocated_payloads,
            "payment": payment,
            "scores": scores
        }

class UCBSecondPrice(AuctionMechanism):
    """Upper Confidence Bound with a Second-Price rule."""
    def __init__(self, providers: List[Provider], c: float = 2, seed: int = 42):
        super().__init__(providers, seed)
        self.c = c
        self.t = 1 # Total number of rounds
        self.n_i = {p.name: 1 for p in providers} # Number of times each provider was chosen

    def run_auction(self, payloads: List[Payload], context: Context, capacity_mass_kg: float) -> Dict:
        self.t += 1
        
        # 1. Calculate UCB scores
        mean_qualities = {name: p.alpha / (p.alpha + p.beta) for name, p in self.providers.items()}
        ucb_bonuses = {name: self.c * np.sqrt(np.log(self.t) / self.n_i[name]) for name in self.providers}
        ucb_scores = {name: mean_qualities[name] + ucb_bonuses[name] for name in self.providers}
        
        # 2. Score providers
        scores = {name: score_payload(p, ucb_scores[name]) for name, p in self.providers.items()}
        
        # 3. Determine winner
        winner_name = max(scores, key=scores.get)
        self.n_i[winner_name] += 1
        
        # 4. Determine second price
        second_highest_score = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        
        # 5. Allocate
        allocated_payloads, _ = greedy_density_allocation(payloads, list(self.providers.values()), scores, capacity_mass_kg, context)
        total_value = sum(get_payload_value(p, context, 6500, 0.2) for p in payloads if p.payload_id in allocated_payloads)
        payment = second_highest_score * total_value

        return {
            "winner": winner_name,
            "allocation": allocated_payloads,
            "payment": payment,
            "scores": scores
        }

class ProbabilisticDSIC(AuctionMechanism):
    """Probabilistic auction based on Plackett-Luce / Gumbel-Softmax."""
    def __init__(self, providers: List[Provider], tau: float = 0.1, seed: int = 42):
        super().__init__(providers, seed)
        self.tau = tau

    def run_auction(self, payloads: List[Payload], context: Context, capacity_mass_kg: float) -> Dict:
        # 1. Use mean quality as the base score
        mean_qualities = {name: p.alpha / (p.alpha + p.beta) for name, p in self.providers.items()}
        scores = {name: score_payload(p, mean_qualities[name]) for name, p in self.providers.items()}

        # 2. Add Gumbel noise for probabilistic selection
        gumbel_noise = self.rng.gumbel(loc=0, scale=1, size=len(self.providers))
        noisy_scores = {name: scores[name] + gumbel_noise[i] for i, name in enumerate(self.providers.keys())}
        
        # 3. Use softmax to get probabilities
        exp_scores = {name: np.exp(s / self.tau) for name, s in noisy_scores.items()}
        sum_exp_scores = sum(exp_scores.values())
        probabilities = {name: exp_s / sum_exp_scores for name, exp_s in exp_scores.items()}
        
        # 4. Select winner based on probabilities
        winner_name = self.rng.choice(list(self.providers.keys()), p=list(probabilities.values()))
        
        # 5. Pricing (simplified Vickrey-like payment)
        # A true DSIC mechanism has complex pricing. We approximate.
        # Price is the score needed to have the same probability as the second-highest original score.
        # This is a simplification.
        sorted_scores = sorted(scores.values(), reverse=True)
        second_highest_score = sorted_scores[1] if len(sorted_scores) > 1 else 0

        # 6. Allocate
        allocated_payloads, _ = greedy_density_allocation(payloads, list(self.providers.values()), scores, capacity_mass_kg, context)
        total_value = sum(get_payload_value(p, context, 6500, 0.2) for p in payloads if p.payload_id in allocated_payloads)
        payment = second_highest_score * total_value

        return {
            "winner": winner_name,
            "allocation": allocated_payloads,
            "payment": payment,
            "scores": scores,
            "probabilities": probabilities
        }
