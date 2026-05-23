"""
engine.py
---------
Simulation orchestrator — three-layer latent demand model.

Layer 1 — Market demand (demand.py)
  λ_OD(t) = μ_OD · seasonal(t) · dow(t) · trend(t) · shock(t) · noise(t)
  μ_OD is a fixed calibrated constant from OD_MARKET_SIZE (config.py).
  It does NOT depend on seat counts.  Adding a flight increases capacity
  but never increases demand.

Layer 2 — Supply allocation (engine.py)
  Flights compete for the latent demand proportionally by seat count.
  flight_share = seats_this_flight / total_OD_seats
  This is allocation only — demand already exists before this step.

Layer 3 — Booking / censoring (simulation.py)
  Discrete Poisson arrivals over the booking window, FCFS per cabin.
  Produces oracle (latent) vs realized (observed) and BOH trajectory.

Two outputs:

  run()            → OD-pair level DataFrame (one row per od_pair × date)
                     latent_seats = sum of per-flight oracle pax (Poisson draws)
                     expected_demand = raw float Poisson mean from demand model

  get_flights_df() → Per-flight DataFrame (one row per flight_od × date)
                     Includes cabin breakdown, BOH trajectory, revenue.
                     is_censored = 1 only when a cabin sold out completely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from simulator.config import (
    COUNTRIES, ROUTES, AIRCRAFT, AIRPORT_COUNTRY, OD_MARKET_SIZE,
    Country, Route, Aircraft,
)
from simulator.shocks import ShockEngine
from simulator.demand import DemandParams, simulate_od_demand
from simulator.booking_curves import BookingCurveParams, generate_booking_curves
from simulator.simulation import simulate_flight_booking_window, BOH_SNAPSHOTS, CABINS


@dataclass
class SimConfig:
    start_year: int = 2015
    n_years:    int = 10

    active_route_ids:   Optional[list[str]] = None
    active_country_ids: Optional[list[str]] = None

    demand:  DemandParams       = field(default_factory=DemandParams)
    booking: BookingCurveParams = field(default_factory=BookingCurveParams)

    global_shock_rate:  float = 0.8
    country_shock_rate: float = 1.5
    geopolitical_rate:  float = 0.4

    random_seed: Optional[int] = 42


class SimulationEngine:

    def __init__(self, config: SimConfig | None = None):
        self.config    = config or SimConfig()
        self.rng       = np.random.default_rng(self.config.random_seed)
        self.routes    = self._select_routes()
        self.countries = self._select_countries()
        self.n_months  = self.config.n_years * 12
        start          = date(self.config.start_year, 1, 1)
        end            = date(self.config.start_year + self.config.n_years, 1, 1)
        self.n_days    = (end - start).days
        self._flights_df: pd.DataFrame | None = None

        # Booking curves generated once — shared across all flight simulations.
        # Cabin columns: first, business, premium_eco, economy.
        self._booking_curves = generate_booking_curves(self.config.booking)

        # Build OD route map and base daily demand — computed once at startup,
        # independent of simulation time.  Adding flights increases supply
        # but does not change demand.
        self._od_route_map: dict[str, list[tuple[Route, Aircraft]]] = {}
        self._od_base_demand: dict[str, float] = {}
        self._od_dest_country: dict[str, str] = {}
        self._od_market_country: dict[str, str] = {}

        for route in self.routes:
            dest_country = AIRPORT_COUNTRY.get(route.dest_airport, "??")
            od_pair      = f"{route.market_country}→{dest_country}"
            aircraft     = AIRCRAFT.get(route.aircraft_type, AIRCRAFT["B789"])

            self._od_route_map.setdefault(od_pair, []).append((route, aircraft))
            self._od_dest_country[od_pair]   = dest_country
            self._od_market_country[od_pair] = route.market_country

        for od_pair in self._od_route_map:
            # Layer 1: market size is a supply-independent calibrated constant.
            # It does NOT change when routes are added or removed — only
            # capacity (Layer 2) changes.  If an OD pair has no entry, warn
            # and fall back to a conservative estimate.
            if od_pair in OD_MARKET_SIZE:
                self._od_base_demand[od_pair] = OD_MARKET_SIZE[od_pair]
            else:
                routes_aircraft = self._od_route_map[od_pair]
                fallback = sum(a.total_seats for _, a in routes_aircraft) * 0.75
                print(f"  WARNING: no OD_MARKET_SIZE entry for '{od_pair}' "
                      f"— using fallback {fallback:.0f} pax/day")
                self._od_base_demand[od_pair] = fallback

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Run the simulation.  Returns OD-pair level DataFrame.
        Also stores per-flight DataFrame; call get_flights_df() to retrieve it.
        """
        n_od = len(self._od_route_map)
        print(f"Simulating {self.n_days} days | "
              f"{len(self.routes)} flights across {n_od} OD pairs | "
              f"{len(self.countries)} markets | "
              f"booking window = {self.config.booking.booking_window_days}d")

        shock_engine = ShockEngine(
            n_months=self.n_months,
            country_ids=list(self.countries.keys()),
            global_shock_rate=self.config.global_shock_rate,
            country_shock_rate=self.config.country_shock_rate,
            geopolitical_rate=self.config.geopolitical_rate,
            rng=self.rng,
        )
        print(f"  Shocks generated: {len(shock_engine.shocks)} events")

        od_records     = []
        flight_records = []
        start_date = date(self.config.start_year, 1, 1)

        for day_idx in range(self.n_days):
            current_date   = start_date + timedelta(days=day_idx)
            year           = current_date.year
            month          = current_date.month
            day            = current_date.day
            dow            = current_date.weekday()
            calendar_month = month - 1
            month_idx      = (year - self.config.start_year) * 12 + (month - 1)

            for od_pair, routes_aircraft in self._od_route_map.items():
                market_country = self._od_market_country[od_pair]
                dest_country   = self._od_dest_country[od_pair]
                country        = self.countries.get(market_country)
                if country is None:
                    continue

                # ── Step 1: OD-level expected demand ──────────────────────────
                # Returns expected_demand (Poisson mean) and factor breakdown.
                # Shocks and seasonality are applied here — upstream of flights.
                # Layer 1 — generate market-level latent demand.
                # market_size is supply-independent; shocks/season/trend
                # are the only things that move it.
                od = simulate_od_demand(
                    country=country,
                    market_size=self._od_base_demand[od_pair],
                    month_idx=month_idx,
                    calendar_month=calendar_month,
                    dow=dow,
                    shock_engine=shock_engine,
                    params=self.config.demand,
                    rng=self.rng,
                )

                od_expected    = od["expected_demand"]   # float Poisson mean
                total_od_cap   = sum(a.total_seats for _, a in routes_aircraft)
                n_flights      = len(routes_aircraft)
                econ_yield     = od["yield_economy_usd"]

                # ── Step 2: per-flight discrete booking simulation ─────────────
                # Each flight independently simulates the full booking window
                # with Poisson arrivals and FCFS cabin capacity management.
                flight_sims: list[tuple[Route, Aircraft, dict]] = []

                for route, aircraft in routes_aircraft:
                    share            = aircraft.total_seats / total_od_cap
                    flight_expected  = od_expected * share

                    capacity_by_cabin = {
                        "first":       aircraft.first_seats,
                        "business":    aircraft.business_seats,
                        "premium_eco": aircraft.premium_eco_seats,
                        "economy":     aircraft.economy_seats,
                    }

                    sim = simulate_flight_booking_window(
                        capacity_by_cabin=capacity_by_cabin,
                        expected_demand=flight_expected,
                        booking_curves=self._booking_curves,
                        rng=self.rng,
                    )
                    flight_sims.append((route, aircraft, sim))

                # OD latent = sum of all flight oracle draws
                od_latent = sum(s["latent_seats"] for _, _, s in flight_sims)

                od_records.append({
                    "date":             current_date.isoformat(),
                    "year":             year,
                    "month":            month,
                    "day":              day,
                    "dow":              dow,
                    "od_pair":          od_pair,
                    "market_country":   market_country,
                    "dest_country":     dest_country,
                    "n_flights":        n_flights,
                    "total_capacity":   total_od_cap,
                    "expected_demand":  round(od_expected, 1),
                    "latent_seats":     od_latent,
                    "yield_economy_usd": econ_yield,
                    "shock_factor":     od["shock_factor"],
                    "seasonal_index":   od["seasonal_index"],
                    "dow_factor":       od["dow_factor"],
                    "trend_factor":     od["trend_factor"],
                })

                # ── Step 3: build per-flight records ──────────────────────────
                for route, aircraft, sim in flight_sims:
                    flight_cap  = aircraft.total_seats
                    observed    = sim["observed_seats"]

                    # Revenue: cabin-weighted yield
                    cabin_yield_mult = self.config.demand.cabin_yield_mult
                    revenue = 0.0
                    for cab in CABINS:
                        obs_cab  = sim[f"observed_{cab}_pax"]
                        mult     = cabin_yield_mult.get(cab, 1.0)
                        revenue += obs_cab * econ_yield * mult

                    # Connecting pax discount
                    conn_share = self.config.demand.connecting_share_base
                    if route.is_connecting:
                        conn_share = min(
                            0.75,
                            conn_share
                            + self.config.demand.connecting_share_hub_bonus
                            + 0.10,
                        )
                    direct_premium = self.config.demand.direct_yield_premium
                    revenue = round(revenue * (1.0 - conn_share * direct_premium), 2)

                    rec = {
                        "date":           current_date.isoformat(),
                        "year":           year,
                        "month":          month,
                        "day":            day,
                        "dow":            dow,
                        "od_pair":        od_pair,
                        "market_country": market_country,
                        "dest_country":   dest_country,
                        "flight_od":      route.flight_od,
                        "is_connecting":  int(route.is_connecting),
                        "aircraft_type":      aircraft.type,
                        "seats_capacity":     flight_cap,
                        "first_capacity":     aircraft.first_seats,
                        "business_capacity":  aircraft.business_seats,
                        "premium_eco_capacity": aircraft.premium_eco_seats,
                        "economy_capacity":   aircraft.economy_seats,
                        "distance_km":        route.distance_km,
                        # Demand
                        "latent_seats":   sim["latent_seats"],
                        "observed_seats": observed,
                        "is_censored":    sim["is_censored"],
                        "revenue_usd":    revenue,
                        # Shared OD-level factors
                        "shock_factor":   od["shock_factor"],
                        "seasonal_index": od["seasonal_index"],
                        "dow_factor":     od["dow_factor"],
                        "trend_factor":   od["trend_factor"],
                    }

                    # Per-cabin oracle and observed
                    for cab in CABINS:
                        rec[f"oracle_{cab}_pax"]   = sim[f"oracle_{cab}_pax"]
                        rec[f"observed_{cab}_pax"] = sim[f"observed_{cab}_pax"]

                    # BOH snapshots
                    for snap in sorted(BOH_SNAPSHOTS, reverse=True):
                        rec[f"boh_dtp_{snap}"] = sim[f"boh_dtp_{snap}"]

                    flight_records.append(rec)

        df_od = pd.DataFrame(od_records)
        df_od["date"] = pd.to_datetime(df_od["date"])
        df_od = df_od.sort_values(["od_pair", "date"]).reset_index(drop=True)

        self._flights_df = pd.DataFrame(flight_records)
        self._flights_df["date"] = pd.to_datetime(self._flights_df["date"])
        self._flights_df = (
            self._flights_df
            .sort_values(["flight_od", "date"])
            .reset_index(drop=True)
        )

        print(f"  Done. {len(df_od):,} OD rows | "
              f"{len(self._flights_df):,} flight rows | "
              f"{self._flights_df['flight_od'].nunique()} individual flights")
        return df_od

    def get_flights_df(self) -> pd.DataFrame:
        """Per-flight DataFrame from the last run().  Call run() first."""
        if self._flights_df is None:
            raise RuntimeError("Call run() before get_flights_df().")
        return self._flights_df.copy()

    def shock_log(self) -> pd.DataFrame:
        engine = ShockEngine(
            n_months=self.n_months,
            country_ids=list(self.countries.keys()),
            global_shock_rate=self.config.global_shock_rate,
            country_shock_rate=self.config.country_shock_rate,
            geopolitical_rate=self.config.geopolitical_rate,
            rng=np.random.default_rng(self.config.random_seed),
        )
        return pd.DataFrame(engine.shock_summary())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _select_routes(self) -> list[Route]:
        ids = self.config.active_route_ids
        if ids is None:
            return ROUTES
        return [r for r in ROUTES if r.id in ids]

    def _select_countries(self) -> dict[str, Country]:
        ids = self.config.active_country_ids
        if ids is None:
            return COUNTRIES
        return {k: v for k, v in COUNTRIES.items() if k in ids}
