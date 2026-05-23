"""
visualize_truncation.py
-----------------------
Per-flight demand truncation plots.

For each O-D pair, produces a grid of subplots — one per flight_od — showing:
  · Blue fill   : observed (constrained) demand
  · Orange fill : spilled demand (latent − constrained on censored days)
  · Red line    : true latent demand
  · Dashed line : individual flight seat capacity

This directly motivates the unconstraining step: the orange "spill" is
the demand the EM/PD algorithms try to recover.

Input:  output/simulation_slim.csv
Output: unconstraining/results/truncation_<OD>.png   (one per O-D)
        unconstraining/results/truncation_all.png    (4-panel summary)

Usage (from Simulation root):
    python unconstraining/visualize_truncation.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

ROOT    = os.path.join(os.path.dirname(__file__), "..")
SLIM    = os.path.join(ROOT, "output", "simulation_slim.csv")
OUT_DIR = os.path.join(ROOT, "unconstraining", "results")

ROLL = 14   # days rolling mean for smoothing

C_OBS  = "#4472C4"   # observed (constrained)
C_SPIL = "#FF8C00"   # spilled demand
C_LAT  = "#C00000"   # latent demand line
C_CAP  = "#222222"   # capacity cap


def _smooth(series: pd.Series) -> pd.Series:
    return series.rolling(ROLL, center=True, min_periods=1).mean()


def _plot_flight(ax, g: pd.DataFrame, flight_id: str):
    cap  = int(g["seats_capacity"].iloc[0])
    dates = g["date"]

    obs    = _smooth(g["seats_sold"])
    lat    = _smooth(g["latent_seats_sold"])
    spill  = (lat - obs).clip(lower=0)

    ax.fill_between(dates, 0, obs,          alpha=0.55, color=C_OBS,  label="Observed")
    ax.fill_between(dates, obs, obs + spill, alpha=0.65, color=C_SPIL, label="Spilled")
    ax.plot(dates, lat, color=C_LAT, lw=1.1, label="Latent")
    ax.axhline(cap, color=C_CAP, lw=0.9, ls="--", alpha=0.6, label=f"Cap {cap}")

    cens_rate = g["is_censored"].mean()
    ax.set_title(f"{flight_id}   cap={cap}   cens={cens_rate:.0%}", fontsize=8.5)
    ax.set_ylim(0, cap * 1.25)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.18)


def plot_od(df: pd.DataFrame, od: str, out_dir: str):
    od_df   = df[df["od_pair"] == od].sort_values("date")
    flights = sorted(od_df["flight_od"].unique())
    n       = len(flights)

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 3.2 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, fid in enumerate(flights):
        g = od_df[od_df["flight_od"] == fid]
        _plot_flight(axes_flat[i], g, fid)
        if i == 0:
            axes_flat[i].legend(fontsize=7, loc="upper left", framealpha=0.7)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{od}  —  Flight-level Demand Truncation  ({ROLL}-day rolling mean)\n"
        "Blue = observed seats   ·   Orange = spilled demand   ·   Red = true latent demand",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    slug = od.replace("→", "_to_").replace(" ", "")
    out  = os.path.join(out_dir, f"truncation_{slug}.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  {od:<12}  ({n} flights)  →  {out}")


def plot_summary_grid(df: pd.DataFrame, out_dir: str):
    """4-panel summary: one representative flight per market, same scale."""
    # Pick the highest-capacity flight in each origin market
    markets = sorted(df["market_country"].unique())
    reps = []
    for mkt in markets:
        sub = df[df["market_country"] == mkt]
        fid = (sub.groupby("flight_od")["seats_capacity"]
                  .first().idxmax())
        reps.append((mkt, fid))

    n     = len(reps)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, (mkt, fid) in enumerate(reps):
        g = df[df["flight_od"] == fid].sort_values("date")
        _plot_flight(axes_flat[i], g, fid)
        axes_flat[i].set_title(f"[{mkt}]  {fid}\ncens={g['is_censored'].mean():.0%}  "
                                f"cap={int(g['seats_capacity'].iloc[0])}",
                                fontsize=8)
        if i == 0:
            axes_flat[i].legend(fontsize=7, loc="upper left", framealpha=0.7)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Demand Truncation Summary — Highest-Capacity Flight per Market\n"
        f"({ROLL}-day rolling mean;  Blue=Observed  ·  Orange=Spilled  ·  Red=Latent)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "truncation_all.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n  Summary grid → {out}")


def main():
    if not os.path.exists(SLIM):
        print(f"ERROR: {SLIM} not found. Run scripts/run.py first.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {SLIM}...")
    df = pd.read_csv(SLIM, parse_dates=["date"])
    print(f"  {len(df):,} flight-day records, {df['flight_od'].nunique()} flights, "
          f"{df['od_pair'].nunique()} O-D pairs\n")

    od_pairs = sorted(df["od_pair"].unique())
    print(f"Per-O-D truncation plots ({len(od_pairs)} markets):")
    for od in od_pairs:
        plot_od(df, od, OUT_DIR)

    plot_summary_grid(df, OUT_DIR)


if __name__ == "__main__":
    main()
