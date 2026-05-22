"""
demand.py
---------
Demand and yield simulation for each route-month.

Models:
  1. Load factor (LF) — fraction of seats filled, per route per month
  2. Cabin class split — how LF distributes across F / J / W / Y cabins
  3. Connecting vs direct passenger split
  4. Yield (revenue per seat) — correlated with LF via price elasticity
  5. Blended yield — weighted average across cabin and passenger type

Key design choices:
  - LF is computed multiplicatively so each effect is inspectable independently
  - Price elasticity is applied at the market level (higher demand = higher yield)
  - Connecting pax always yield less than direct (typically 20–30% discount)
  - Cabin yields are anchored to economy; premium multipliers are applied on top
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from simulator.config import Route, Aircraft, Country
from simulator.seasonality import get_seasonal_index, get_dow_index
from simulator.shocks import ShockEngine


@dataclass
class DemandParams:
    base_load_factor: float = 0.72       # global baseline LF
    noise_sigma: float = 0.08            # Gaussian noise σ (fraction)
    seasonal_noise_sigma: float = 0.06   # stochastic variation around monthly seasonal index
    dow_noise_sigma: float = 0.05        # stochastic variation around DOW multiplier
    price_elasticity: float = 0.80       # how much yield rises with LF
    base_yield_economy_usd: float = 750  # base economy yield (USD, one-way)
    direct_yield_premium: float = 0.22   # direct pax pay this much more than connecting
    connecting_share_base: float = 0.28  # fraction of seats from connecting pax
    connecting_share_hub_bonus: float = 0.12  # extra connecting share for hub routes

    # Cabin yield multipliers relative to economy
    cabin_yield_mult: dict[str, float] = None

    # Cabin load factor skew — premium cabins often sell at higher LF than economy
    cabin_lf_skew: dict[str, float] = None

    def __post_init__(self):
        if self.cabin_yield_mult is None:
            self.cabin_yield_mult = {
                "first":       8.0,
                "business":    4.5,
                "premium_eco": 1.8,
                "economy":     1.0,
            }
        if self.cabin_lf_skew is None:
            # Business/first often sell out faster on transpacific
            self.cabin_lf_skew = {
                "first":       1.10,
                "business":    1.08,
                "premium_eco": 1.02,
                "economy":     0.98,
            }


def simulate_route_day(
    route: Route,
    aircraft: Aircraft,
    country: Country,
    month_idx: int,         # 0-indexed month within simulation horizon
    calendar_month: int,    # 0=Jan … 11=Dec
    dow: int,               # 0=Monday … 6=Sunday
    shock_engine: ShockEngine,
    params: DemandParams,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate one route × month record.

    Returns a flat dict ready for DataFrame construction.
    """

    # ── 1. Load factor components ─────────────────────────────────────────────

    route_base = route.base_load_factor           # route-specific baseline
    seasonal   = get_seasonal_index(country.season_profile, calendar_month)
    seasonal   = seasonal * (1.0 + params.seasonal_noise_sigma * rng.standard_normal())
    dow_factor = get_dow_index(dow)
    dow_factor = dow_factor * (1.0 + params.dow_noise_sigma * rng.standard_normal())
    # Front-loaded growth: high acceleration in early years, tapering to base rate.
    # accel decays from 1.5x to 1.0x over the first ~3 years.
    years_elapsed = month_idx / 12
    accel  = 1.0 + 0.5 * np.exp(-years_elapsed / 3.0)
    trend  = 1.0 + country.demand_trend * accel * years_elapsed
    shock      = shock_engine.get_multiplier(month_idx, country.id)
    noise      = 1.0 + params.noise_sigma * rng.standard_normal()

    lf_raw = route_base * seasonal * dow_factor * trend * shock * noise
    lf = float(np.clip(lf_raw, 0.15, 0.99))

    # ── 2. Seats by cabin ─────────────────────────────────────────────────────

    cabin_seats = {}
    cabin_lf    = {}
    for cabin, skew in params.cabin_lf_skew.items():
        raw_cabin_lf = lf * skew
        raw_cabin_lf = float(np.clip(raw_cabin_lf, 0.10, 0.99))
        cabin_lf[cabin] = raw_cabin_lf

        cap = getattr(aircraft, f"{cabin}_seats", 0)
        cabin_seats[cabin] = round(cap * raw_cabin_lf)

    total_seats_sold = sum(cabin_seats.values())
    total_capacity   = aircraft.total_seats
    latent_seats_sold = round(float(lf_raw) * total_capacity)

    # ── 3. Connecting vs direct split ─────────────────────────────────────────

    conn_share = params.connecting_share_base
    if route.is_connecting:
        conn_share = min(0.75, conn_share + params.connecting_share_hub_bonus + 0.10)

    # Connecting pax are predominantly in economy / premium-eco
    conn_seats   = round(total_seats_sold * conn_share)
    direct_seats = total_seats_sold - conn_seats

    # ── 4. Yield per cabin ────────────────────────────────────────────────────

    # Base economy yield adjusted for LF elasticity
    lf_deviation = lf - params.base_load_factor
    economy_yield = params.base_yield_economy_usd * (
        1.0 + params.price_elasticity * lf_deviation
    )
    economy_yield = max(150.0, economy_yield)

    cabin_yield = {
        cabin: round(economy_yield * mult, 2)
        for cabin, mult in params.cabin_yield_mult.items()
    }

    # Connecting pax discount (applies on blended yield, weighted to economy)
    direct_yield_blended  = _blended_yield(cabin_seats, cabin_yield, aircraft, conn_fraction=0.0)
    connect_yield_blended = direct_yield_blended * (1.0 - params.direct_yield_premium)

    # Overall blended yield
    if total_seats_sold > 0:
        overall_yield = (
            (direct_seats * direct_yield_blended + conn_seats * connect_yield_blended)
            / total_seats_sold
        )
    else:
        overall_yield = direct_yield_blended

    # ── 5. Revenue ────────────────────────────────────────────────────────────

    revenue = round(
        direct_seats * direct_yield_blended + conn_seats * connect_yield_blended, 2
    )

    return {
        # Identifiers
        "flight_od":         route.flight_od,
        "origin_airport":    route.origin_airport,
        "dest_airport":      route.dest_airport,
        "market_country":    country.id,
        "market_country_name": country.name,
        "is_connecting":     int(route.is_connecting),
        "connecting_hub":    route.connecting_hub or "",
        "aircraft_type":     aircraft.type,
        "distance_km":       route.distance_km,

        # Demand — constrained (clipped to capacity) and latent (true desire)
        "load_factor":       round(lf, 4),
        "lf_raw":            round(float(lf_raw), 4),
        "seats_sold":        total_seats_sold,
        "seats_capacity":    total_capacity,
        "latent_seats_sold": latent_seats_sold,
        "seats_direct":      direct_seats,
        "seats_connecting":  conn_seats,

        # Cabin seats sold
        "seats_first":       cabin_seats["first"],
        "seats_business":    cabin_seats["business"],
        "seats_premium_eco": cabin_seats["premium_eco"],
        "seats_economy":     cabin_seats["economy"],

        # Cabin load factors
        "lf_first":          round(cabin_lf["first"], 4),
        "lf_business":       round(cabin_lf["business"], 4),
        "lf_premium_eco":    round(cabin_lf["premium_eco"], 4),
        "lf_economy":        round(cabin_lf["economy"], 4),

        # Yield
        "yield_economy_usd":      round(economy_yield, 2),
        "yield_business_usd":     cabin_yield["business"],
        "yield_first_usd":        cabin_yield["first"],
        "yield_premium_eco_usd":  cabin_yield["premium_eco"],
        "yield_direct_blended":   round(direct_yield_blended, 2),
        "yield_connect_blended":  round(connect_yield_blended, 2),
        "yield_overall_blended":  round(overall_yield, 2),

        # Revenue
        "revenue_usd":       revenue,

        # Simulation meta
        "seasonal_index":    round(seasonal, 4),
        "dow_factor":        round(dow_factor, 4),
        "shock_factor":      round(shock, 4),
        "trend_factor":      round(trend, 4),
    }


def _blended_yield(
    cabin_seats: dict[str, int],
    cabin_yield: dict[str, float],
    aircraft: Aircraft,
    conn_fraction: float = 0.0,
) -> float:
    """
    Weighted average yield across cabins, based on seats sold.
    conn_fraction reserved for future connecting-cabin weighting.
    """
    total = sum(cabin_seats.values())
    if total == 0:
        return 0.0
    weighted = sum(cabin_seats[c] * cabin_yield[c] for c in cabin_seats)
    return weighted / total
