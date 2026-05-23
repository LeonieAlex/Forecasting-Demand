"""
run.py
------
Entry point. Generates two complementary datasets:

  data/OD_LatentDemand.csv
      One row per (od_pair, date).
      latent_seats = total true demand across all flights on the corridor,
      before any aircraft capacity constraint.
      Use this to answer: "how much total demand was there from JP to US?"

  data/flights_latent.csv
      One row per (flight_od, date).
      Each flight keeps its own fixed seats_capacity (same aircraft type =
      same seat count every day).  Connecting and direct services are both
      included.  is_censored = 1 only when that specific flight sold out.
      Use this to answer: "how many passengers wanted this exact service?"
      Also used by batch_unconstrain.py for per-flight EM/PD unconstraining.

Usage:
    python scripts/run.py                          # full 10-year run
    python scripts/run.py --years 5                # shorter horizon
    python scripts/run.py --countries JP CN KR     # subset of markets
    python scripts/run.py --seed 99                # different random seed
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.engine import SimulationEngine, SimConfig
from simulator.demand import DemandParams
from simulator.booking_curves import BookingCurveParams
from simulator.config import OD_MARKET_SIZE, ROUTES, AIRCRAFT, AIRPORT_COUNTRY


def parse_args():
    p = argparse.ArgumentParser(description="OD-pair + per-flight latent demand simulator")
    p.add_argument("--years",     type=int,  default=10,   help="Years to simulate")
    p.add_argument("--start",     type=int,  default=2015, help="Start year")
    p.add_argument("--seed",      type=int,  default=42,   help="Random seed")
    p.add_argument("--routes",    nargs="*", default=None, help="Route IDs (default: all)")
    p.add_argument("--countries", nargs="*", default=None, help="Country ISO-2 codes")
    p.add_argument("--out-dir",        type=str,   default="data", help="Output directory")
    p.add_argument("--global-trend",   type=float, default=0.035,
                   help="Global annual demand growth applied to all markets (default: 0.035 = +3.5%%/yr)")
    return p.parse_args()


def build_config(args) -> SimConfig:
    return SimConfig(
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
            global_demand_trend=args.global_trend,
        ),
        booking=BookingCurveParams(booking_window_days=90),
        global_shock_rate=0.8,
        country_shock_rate=1.5,
        geopolitical_rate=0.4,
    )


def print_summary(df_od: pd.DataFrame, df_flights: pd.DataFrame):
    print("\n" + "=" * 65)
    print("SIMULATION SUMMARY")
    print("=" * 65)

    print(f"\n── OD-pair dataset ──────────────────────────────────────────")
    print(f"  Rows:        {len(df_od):,}")
    print(f"  OD pairs:    {df_od['od_pair'].nunique()}")
    print(f"  Date range:  {df_od['date'].min().date()} → {df_od['date'].max().date()}")

    print(f"\n── Per-flight dataset ───────────────────────────────────────")
    print(f"  Rows:        {len(df_flights):,}")
    print(f"  Flights:     {df_flights['flight_od'].nunique()}")
    direct     = (~df_flights["is_connecting"].astype(bool)).sum()
    connecting = df_flights["is_connecting"].sum()
    print(f"  Direct obs:  {direct:,}    Connecting obs: {connecting:,}")
    cens_pct = df_flights["is_censored"].mean()
    print(f"  Censored:    {df_flights['is_censored'].sum():,} flight-days ({cens_pct:.1%})")

    print(f"\n── Latent vs observed seats by OD pair (avg/day) ────────────")
    od_summary = (
        df_od.groupby("od_pair")[["latent_seats", "expected_demand"]]
        .mean()
        .round(1)
        .sort_values("latent_seats", ascending=False)
    )
    print(od_summary.to_string())

    print(f"\n── Cabin breakdown (avg pax/flight-day) ─────────────────────")
    cabin_cols_oracle   = [c for c in df_flights.columns if c.startswith("oracle_")]
    cabin_cols_observed = [c for c in df_flights.columns if c.startswith("observed_") and "_pax" in c]
    if cabin_cols_oracle:
        cab_summary = pd.DataFrame({
            "oracle":   df_flights[cabin_cols_oracle].mean().rename(lambda x: x.replace("oracle_","").replace("_pax","")),
            "observed": df_flights[cabin_cols_observed].mean().rename(lambda x: x.replace("observed_","").replace("_pax","")),
        }).round(1)
        print(cab_summary.to_string())

    print(f"\n── BOH trajectory (avg across all flights) ──────────────────")
    boh_cols = sorted([c for c in df_flights.columns if c.startswith("boh_dtp_")],
                      key=lambda x: -int(x.split("_")[-1]))
    if boh_cols:
        boh_avg = df_flights[boh_cols].mean().round(1)
        boh_avg.index = [c.replace("boh_dtp_", "DTP-") for c in boh_avg.index]
        print(boh_avg.to_string())

    print(f"\n── Per-flight capacity (fixed by aircraft type) ─────────────")
    cap_summary = (
        df_flights.groupby(["flight_od", "aircraft_type"])["seats_capacity"]
        .first()
        .reset_index()
        .sort_values("seats_capacity", ascending=False)
    )
    print(cap_summary.to_string(index=False))

    print(f"\n── Seasonal pattern (avg latent seats by month) ─────────────")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    seasonal = df_od.groupby(["market_country","month"])["latent_seats"].mean().unstack()
    seasonal.columns = [month_names[m - 1] for m in seasonal.columns]
    print(seasonal.round(0).to_string())

    print("=" * 65)


def plot_latent_vs_realised(df_flights: pd.DataFrame, out_dir: str):
    """Latent vs realised time-series for every route, one subplot per flight."""
    import matplotlib.dates as mdates

    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0e1117", "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d", "grid.color": "#21262d",
        "grid.linewidth": 0.6, "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e", "ytick.color": "#8b949e",
        "text.color": "#c9d1d9", "font.size": 9,
    })

    routes    = sorted(df_flights["flight_od"].unique())
    year_show = df_flights["date"].dt.year.max() - 1   # last complete year

    ncols = 4
    nrows = -(-len(routes) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, nrows * 3.2))
    axes_flat = axes.flatten()

    for idx, route in enumerate(routes):
        ax   = axes_flat[idx]
        data = (df_flights[(df_flights["flight_od"] == route) &
                           (df_flights["date"].dt.year == year_show)]
                .sort_values("date"))

        if data.empty:
            ax.set_visible(False)
            continue

        cap = int(data["seats_capacity"].iloc[0])

        ax.fill_between(data["date"], data["observed_seats"], data["latent_seats"],
                        color="#f85149", alpha=0.25, label="Spillage")
        ax.plot(data["date"], data["latent_seats"],   color="#58a6ff", lw=1.4, label="Latent")
        ax.plot(data["date"], data["observed_seats"], color="#3fb950", lw=1.4, label="Realised")
        ax.axhline(cap, color="#d29922", lw=1.1, ls="--", alpha=0.7, label=f"Cap {cap}")

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_title(f"{route} — latent vs realised ({year_show})", fontsize=8.5)
        ax.set_ylabel("Passengers", fontsize=8)
        ax.grid(True, axis="y")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7.5)

    for idx in range(len(routes), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Latent vs Realised passengers — all routes — {year_show}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    out_path = os.path.join(results_dir, "latent_vs_realised.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved latent vs realised   → {out_path}")


def save(df_od: pd.DataFrame, df_flights: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    od_path      = os.path.join(out_dir, "OD_LatentDemand.csv")
    flights_path = os.path.join(out_dir, "flights_latent.csv")

    df_od.to_csv(od_path, index=False)
    print(f"\nSaved OD-pair dataset     → {od_path}  ({len(df_od):,} rows)")

    df_flights.to_csv(flights_path, index=False)
    print(f"Saved per-flight dataset  → {flights_path}  ({len(df_flights):,} rows)")


if __name__ == "__main__":
    args   = parse_args()
    cfg    = build_config(args)
    engine = SimulationEngine(cfg)

    df_od      = engine.run()
    df_flights = engine.get_flights_df()

    print_summary(df_od, df_flights)
    save(df_od, df_flights, args.out_dir)
    plot_latent_vs_realised(df_flights, args.out_dir)
