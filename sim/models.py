
from pydantic import BaseModel, Field
from typing import Dict

class Context(BaseModel):
    year: int
    demand_rate: float = Field(..., ge=0)
    value_scale_usd: float = Field(..., ge=0)
    avg_payloads_per_launch: float = Field(..., ge=0)
    capacity_mass_kg: float = Field(..., ge=0)
    orbit_mix: Dict[str, float] = Field(default_factory=dict)

class Provider(BaseModel):
    name: str
    prior_alpha: float = Field(..., gt=0)
    prior_beta: float = Field(..., gt=0)
    incumbent_flag: bool
    # Live simulation state
    alpha: float = 0.0
    beta: float = 0.0

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize live alpha/beta with prior values
        if 'prior_alpha' in data:
            self.alpha = data['prior_alpha']
        if 'prior_beta' in data:
            self.beta = data['prior_beta']

    class Config:
        arbitrary_types_allowed = True

class Payload(BaseModel):
    payload_id: str
    mass_kg: float = Field(..., gt=0)
    orbit: str
    base_value_usd: float = Field(..., ge=0)
