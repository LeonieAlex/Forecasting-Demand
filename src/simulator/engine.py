"""
engine.py
---------
Main simulation orchestrator.

Wires together:
  - Config (routes, countries, aircraft)
  - Shock engine
  - Fuel price simulation
  - Demand / yield per route-month
  - Payload calculation
  - Output assembly into a pandas DataFrame

Usage:
    from simulator.engine import SimulationEngine, SimConfig

    cfg = SimConfig(start_year=2015, n_years=10)
    engine = SimulationEngine(cfg)
    df = engine.run()
    df.to_csv("output/simulation.csv", index=False)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from simulator.config import (
    COUNTRIES, ROUTES, AIRCRAFT,
    Country, Route, Aircraft,
)
from simulator.shocks import ShockEngine
from simulator.fuel import (
    FuelConsumptionParams, compute_fuel_consumption, fuel_efficiency_metrics,
)
from simulator.demand import DemandParams, simulate_route_day
from simulator.booking_curves import (
    BookingCurveParams, generate_booking_curves,
    simulate_reading_date,
)


@dataclass
class SimConfig:
    # Time horizon
    start_year: int = 2015
    n_years: int = 10

    # Route / country selection (None = use all defined)
    active_route_ids: Optional[list[str]] = None
    active_country_ids: Optional[list[str]] = None

    # Sub-module params
    demand: DemandParams = field(default_factory=DemandParams)
    fuel: FuelConsumptionParams = field(default_factory=FuelConsumptionParams)
    booking: BookingCurveParams = field(default_factory=BookingCurveParams)

    # Reading dates: DTP snapshots to include in output (None = skip)
    # e.g. [90, 60, 30, 14, 7, 1] — each adds columns to output
    reading_dates: Optional[list[int]] = None

    # Shock rates (shocks per year)
    global_shock_rate: float = 0.8
    country_shock_rate: float = 1.5
    geopolitical_rate: float = 0.4

    # Reproducibility
    random_seed: Optional[int] = 42


class SimulationEngine:

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

        # Select active routes and countries
        self.routes  = self._select_routes()
        self.countries = self._select_countries()
        self.n_months = self.config.n_years * 12   # shock engine granularity
        start = date(self.config.start_year, 1, 1)
        end   = date(self.config.start_year + self.config.n_years, 1, 1)
        self.n_days = (end - start).days            # accounts for leap years

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Run the full simulation. Returns a tidy DataFrame where each row is
        one flight-leg × month × market-origin-country record.
        """
        print(f"Simulating {self.n_days} days across {len(self.routes)} routes "
              f"and {len(self.countries)} markets...")

        # 1. Shock engine
        shock_engine = ShockEngine(
            n_months=self.n_months,
            country_ids=list(self.countries.keys()),
            global_shock_rate=self.config.global_shock_rate,
            country_shock_rate=self.config.country_shock_rate,
            geopolitical_rate=self.config.geopolitical_rate,
            rng=self.rng,
        )
        print(f"  Shocks generated: {len(shock_engine.shocks)} total events")

        # 2. Booking curves (generated once, shared across all routes/months)
        df_curves = generate_booking_curves(self.config.booking)
        reading_dates = self.config.reading_dates
        if reading_dates:
            print(f"  Booking curves: {self.config.booking.booking_window_days}-day window, "
                  f"reading dates DTP={reading_dates}")

        # 3. Simulate each route × day
        records = []
        start_date = date(self.config.start_year, 1, 1)

        for day_idx in range(self.n_days):
            current_date   = start_date + timedelta(days=day_idx)
            year           = current_date.year
            month          = current_date.month
            day            = current_date.day
            dow            = current_date.weekday()          # 0=Mon … 6=Sun
            calendar_month = month - 1                       # 0-indexed
            month_idx      = (year - self.config.start_year) * 12 + (month - 1)

            for route in self.routes:
                country  = self.countries.get(route.market_country)
                if country is None:
                    continue
                aircraft = AIRCRAFT.get(route.aircraft_type, AIRCRAFT["B789"])

                # ── Passenger demand & yield ──────────────────────────────────
                rec = simulate_route_day(
                    route=route,
                    aircraft=aircraft,
                    country=country,
                    month_idx=month_idx,
                    calendar_month=calendar_month,
                    dow=dow,
                    shock_engine=shock_engine,
                    params=self.config.demand,
                    rng=self.rng,
                )

                # ── Passenger payload (pax body + baggage, 100 kg/seat) ──────
                # Source: Lufthansa operational data
                pax_payload_kg = rec["seats_sold"] * 100

                # ── Fuel consumption (OpenAP + ICAO Annex 6) ─────────────────
                fuel_data = compute_fuel_consumption(
                    aircraft_type=route.aircraft_type,
                    distance_km=route.distance_km,
                    pax_payload_kg=pax_payload_kg,
                    params=self.config.fuel,
                    market_country=route.market_country,
                    dest_region="us",
                )

                # ── Fuel efficiency KPIs ──────────────────────────────────────
                fuel_eff = fuel_efficiency_metrics(
                    fuel_data=fuel_data,
                    seats_sold=rec["seats_sold"],
                    seats_capacity=aircraft.total_seats,
                    distance_km=route.distance_km,
                )

                # ── Booking curve reading date snapshots ──────────────────────
                booking_snap = {}
                if reading_dates:
                    seats_by_cabin = {
                        "first":       rec["seats_first"],
                        "business":    rec["seats_business"],
                        "premium_eco": rec["seats_premium_eco"],
                        "economy":     rec["seats_economy"],
                    }
                    for dtp in reading_dates:
                        snap = simulate_reading_date(
                            df_curves=df_curves,
                            days_prior=dtp,
                            total_seats_by_segment=seats_by_cabin,
                            rng=self.rng,
                        )
                        for cabin, data in snap.items():
                            col_prefix = f"dtp{dtp:02d}_{cabin}"
                            booking_snap[f"{col_prefix}_booked"] = data["booked_seats"]
                            booking_snap[f"{col_prefix}_pct"]    = data["pct_booked"]

                # ── Assemble full record ──────────────────────────────────────
                records.append({
                    "year":  year,
                    "month": month,
                    "day":   day,
                    "dow":   dow,
                    "date":  current_date.isoformat(),
                    **rec,
                    # Passenger payload
                    "pax_payload_kg": pax_payload_kg,
                    # Fuel consumption (physical quantities only)
                    **fuel_data,
                    # Fuel efficiency KPIs
                    **fuel_eff,
                    # Booking snapshots
                    **booking_snap,
                })

        df = pd.DataFrame(records)
        df = self._add_derived_columns(df)

        print(f"  Done. {len(df):,} records, {len(df.columns)} columns.")
        return df

    def shock_log(self) -> pd.DataFrame:
        """Convenience: re-run shock gen and return shock summary as DataFrame."""
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

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived KPI columns after the main loop."""

        # RASK (revenue per available seat-km) — passenger revenue only
        df["rask_usd"] = (
            df["revenue_usd"] / df["ask_km"].replace(0, np.nan)
        ).round(6)

        # Revenue per available seat
        df["rpas_usd"] = (df["revenue_usd"] / df["seats_capacity"]).round(2)

        # Total payload = pax payload (no cargo in this model)
        df["total_payload_kg"] = df["pax_payload_kg"]

        return df

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
