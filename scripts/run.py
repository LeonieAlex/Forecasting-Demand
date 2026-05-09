"""
run.py
------
Entry point. Run this to generate your 10-year transpacific dataset.

Usage:
    python run.py                    # full simulation, all routes
    python run.py --routes nrt-lax hnd-jfk pvg-lax   # specific routes
    python run.py --seed 99          # different random seed
    python run.py --years 5          # shorter horizon
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.engine import SimulationEngine, SimConfig
from simulator.demand import DemandParams
from simulator.fuel import FuelConsumptionParams
from simulator.booking_curves import BookingCurveParams


def parse_args():
    p = argparse.ArgumentParser(description="Transpacific payload & fuel simulator")
    p.add_argument("--years",   type=int,   default=10,  help="Number of years to simulate")
    p.add_argument("--start",   type=int,   default=2015, help="Start year")
    p.add_argument("--seed",    type=int,   default=42,   help="Random seed")
    p.add_argument("--routes",  nargs="*",  default=None, help="Route IDs to include (default: all)")
    p.add_argument("--countries", nargs="*", default=None, help="Country IDs (e.g. JP CN KR)")
    p.add_argument("--out",     type=str,   default="output/simulation.csv")
    return p.parse_args()


def run_simulation(args) -> pd.DataFrame:
    cfg = SimConfig(
        start_year=args.start,
        n_years=args.years,
        active_route_ids=args.routes,
        active_country_ids=args.countries,
        random_seed=args.seed,

        demand=DemandParams(
            base_load_factor=0.83,
            noise_sigma=0.08,
            price_elasticity=0.80,
            base_yield_economy_usd=750,
            direct_yield_premium=0.22,
            connecting_share_base=0.28,
        ),

        fuel=FuelConsumptionParams(
            cruise_step_min=30,
            taxi_time_min=18,
            apply_wind_correction=True,
        ),

        booking=BookingCurveParams(
            booking_window_days=90,
        ),

        # Standard RM reading date snapshots — comment out to skip (faster run)
        reading_dates=[90, 60, 30, 14, 7, 1],

        global_shock_rate=0.8,
        country_shock_rate=1.5,
        geopolitical_rate=0.4,
    )

    engine = SimulationEngine(cfg)
    return engine.run()


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("SIMULATION SUMMARY")
    print("=" * 65)

    print(f"\nRecords:        {len(df):,}")
    print(f"Columns:        {len(df.columns)}")
    print(f"Date range:     {df['date'].min()} → {df['date'].max()}")
    print(f"Routes:         {df['flight_od'].nunique()}")
    print(f"Markets:        {df['market_country'].nunique()}")

    print("\n── Load factor ──────────────────────────────────────────")
    print(df.groupby("market_country_name")["load_factor"].describe().round(3).to_string())

    print("\n── Yield (economy, USD) ─────────────────────────────────")
    print(df.groupby("market_country_name")["yield_economy_usd"].describe().round(0).to_string())

    print("\n── Passenger payload ────────────────────────────────────")
    print(f"Avg pax payload:  {df['pax_payload_kg'].mean():,.0f} kg/flight  (seats × 100 kg, source: Lufthansa)")

    print("\n── Fuel consumption (ICAO Annex 6) ─────────────────────")
    print(f"  m_trip:       {df['trip_fuel_kg'].mean():,.0f} kg  (climb + cruise + descent)")
    print(f"    climb:      {df['climb_fuel_kg'].mean():,.0f} kg")
    print(f"    cruise:     {df['cruise_fuel_kg'].mean():,.0f} kg")
    print(f"    descent:    {df['descent_fuel_kg'].mean():,.0f} kg")
    print(f"  m_cont:       {df['contingency_fuel_kg'].mean():,.0f} kg  (5% of trip)")
    print(f"  m_altn:       {df['alternate_fuel_kg'].mean():,.0f} kg  (200nm diversion)")
    print(f"  m_final_res:  {df['final_reserve_kg'].mean():,.0f} kg  (30 min holding)")
    print(f"  m_taxi:       {df['taxi_fuel_kg'].mean():,.0f} kg  (ICAO EDB idle flow)")
    print(f"  m_total:      {df['total_fuel_uplifted_kg'].mean():,.0f} kg uplifted")
    print(f"  fuel (tonnes):{df['fuel_tonnes'].mean():.1f} t/flight")
    print(f"  fuel/seat:    {df['fuel_per_seat_kg'].mean():.1f} kg/seat sold")
    print(f"  fuel/ASK:     {df['fuel_per_ask_kg'].mean():.5f} kg/ASK")
    print(f"  fuel/RPK:     {df['fuel_per_rpk_kg'].mean():.5f} kg/RPK")

    print("\n── Revenue ──────────────────────────────────────────────")
    print(f"Total revenue:    ${df['revenue_usd'].sum()/1e9:.2f}B (10yr simulated)")
    print(f"Avg RASK:         ${df['rask_usd'].mean():.4f}/ASK")

    dtp_cols = [c for c in df.columns if c.startswith("dtp") and c.endswith("_pct")]
    if dtp_cols:
        print("\n── Booking curve snapshots (avg % booked at reading date) ──")
        dtps = sorted(set(int(c.split("_")[0][3:]) for c in dtp_cols), reverse=True)
        cabins = ["first", "business", "premium_eco", "economy"]
        print(f"{'DTP':<6}" + "".join(f"{c:>14}" for c in cabins))
        for dtp in dtps:
            row = f"{dtp:<6}"
            for cabin in cabins:
                col = f"dtp{dtp:02d}_{cabin}_pct"
                row += f"{df[col].mean():>13.1%} " if col in df.columns else f"{'—':>14}"
            print(row)

    print("=" * 65)


def seasonal_check(df: pd.DataFrame):
    """Print monthly average load factor to verify seasonal patterns."""
    print("\n── Seasonal check: avg LF by month ──────────")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    by_country = df.groupby(["market_country","month"])["load_factor"].mean().unstack()
    by_country.columns = [month_names[m-1] for m in by_country.columns]
    print(by_country.round(3).to_string())


def dow_check(df: pd.DataFrame):
    """Print average load factor by day of week to verify DOW patterns."""
    print("\n── DOW check: avg LF by day of week ─────────")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = df.groupby("dow")["load_factor"].mean()
    by_dow.index = [dow_names[i] for i in by_dow.index]
    print(by_dow.round(3).to_string())


def save_outputs(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows → {out_path}")

    # Also save a slim version with just the key forecasting columns
    slim_cols = [
        "date", "year", "month", "day", "dow", "flight_od", "market_country", "is_connecting",
        "aircraft_type", "openap_aircraft_code", "engine_name", "distance_km",
        "load_factor", "seats_sold", "seats_capacity",
        "seats_direct", "seats_connecting",
        "yield_overall_blended", "revenue_usd",
        "pax_payload_kg",
        # ICAO Annex 6 fuel components (Eq. 3.7)
        "trip_fuel_kg", "climb_fuel_kg", "cruise_fuel_kg", "descent_fuel_kg",
        "contingency_fuel_kg", "alternate_fuel_kg", "final_reserve_kg",
        "taxi_fuel_kg", "total_fuel_uplifted_kg",
        # Units and efficiency KPIs
        "fuel_tonnes", "fuel_litres",
        "fuel_per_seat_kg", "fuel_per_ask_kg", "fuel_per_rpk_kg",
        "wind_correction",
        "seasonal_index", "dow_factor", "shock_factor", "rask_usd",
    ]
    slim_path = out_path.replace(".csv", "_slim.csv")
    df[slim_cols].to_csv(slim_path, index=False)
    print(f"Saved slim version → {slim_path}")


if __name__ == "__main__":
    args = parse_args()
    df = run_simulation(args)
    print_summary(df)
    seasonal_check(df)
    dow_check(df)
    save_outputs(df, args.out)
