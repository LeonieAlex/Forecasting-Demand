"""
demand.py
---------
OD-level demand simulation.

Demand is generated as an absolute passenger count for the OD corridor —
independent of how many flights or total seats are available.  Capacity is
supply-side and handled in the engine when distributing OD demand to flights.

simulate_od_demand() takes a base_daily_demand (passengers/day at baseline)
and applies seasonal, DOW, trend, shock, and noise multipliers to produce
the day's latent passenger count for that corridor.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from simulator.config import Country
from simulator.seasonality import get_seasonal_index, get_dow_index
from simulator.shocks import ShockEngine


@dataclass
class DemandParams:
    noise_sigma: float = 0.08             # day-to-day Gaussian noise σ
    price_elasticity: float = 0.80        # yield sensitivity to LF
    base_yield_economy_usd: float = 750   # economy yield at baseline LF (USD)
    base_load_factor: float = 0.83        # reference LF for yield computation
    direct_yield_premium: float = 0.22    # direct pax pay more than connecting
    connecting_share_base: float = 0.28   # base connecting pax fraction
    connecting_share_hub_bonus: float = 0.12  # extra share for 6th-freedom routes
    global_demand_trend: float = 0.035   # baseline annual growth applied to every market (+3.5%/yr)

    cabin_yield_mult: dict[str, float] = None

    def __post_init__(self):
        if self.cabin_yield_mult is None:
            self.cabin_yield_mult = {
                "first":       8.0,
                "business":    4.5,
                "premium_eco": 1.8,
                "economy":     1.0,
            }


def simulate_od_demand(
    country: Country,
    market_size: float,          # μ_OD — fixed daily passengers at base year, supply-independent
    month_idx: int,              # 0-indexed month in simulation horizon
    calendar_month: int,         # 0=Jan … 11=Dec
    dow: int,                    # 0=Mon … 6=Sun
    shock_engine: ShockEngine,
    params: DemandParams,
    rng: np.random.Generator,
) -> dict:
    """
    Layer 1 — market-level latent demand for one (od_pair, date).

    λ_OD(t) = μ_OD · seasonal(t) · dow(t) · trend(t) · shock(t) · noise(t)

    market_size (μ_OD) is a fixed property of the OD corridor — calibrated
    once, independent of fleet size.  Adding or removing flights does not
    change this value; it only changes how much of the resulting demand can
    be served (Layers 2 and 3).

    Returns expected_demand (the Poisson mean) and the factor breakdown.
    The engine converts expected_demand into per-flight Poisson arrivals.
    """
    seasonal   = get_seasonal_index(country.season_profile, calendar_month)
    dow_factor = get_dow_index(dow)

    years_elapsed = month_idx / 12
    accel  = 1.0 + 0.5 * np.exp(-years_elapsed / 3.0)
    effective_trend = params.global_demand_trend + country.demand_trend
    trend  = 1.0 + effective_trend * accel * years_elapsed

    shock  = shock_engine.get_multiplier(month_idx, country.id)
    noise  = 1.0 + params.noise_sigma * rng.standard_normal()

    # λ_OD(t) — the Poisson mean passed to the booking simulation
    latent_raw   = market_size * seasonal * dow_factor * trend * shock * noise
    latent_seats = max(0, round(float(latent_raw)))

    # Economy yield — driven by how demand compares to the reference baseline
    # (used by the engine to compute per-flight revenue)
    lf_proxy      = latent_raw / market_size   # ratio vs baseline (can exceed 1)
    lf_deviation  = float(np.clip(lf_proxy, 0.15, 1.5)) - params.base_load_factor
    economy_yield = params.base_yield_economy_usd * (
        1.0 + params.price_elasticity * lf_deviation
    )
    economy_yield = max(150.0, economy_yield)

    return {
        "expected_demand":   float(latent_raw),      # Poisson mean — passed to booking sim
        "latent_seats":      latent_seats,            # rounded (updated from flight sums in engine)
        "yield_economy_usd": round(economy_yield, 2),
        "shock_factor":      round(float(shock), 4),
        "seasonal_index":    round(float(seasonal), 4),
        "dow_factor":        round(float(dow_factor), 4),
        "trend_factor":      round(float(trend), 4),
    }
