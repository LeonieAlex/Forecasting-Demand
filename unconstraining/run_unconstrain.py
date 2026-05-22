"""
run_unconstrain.py
------------------
Applies unconstraining models to passenger demand per O-D pair.

Because this is *simulated* data we have ground-truth latent demand
(latent_seats_sold / lf_raw) alongside the capacity-constrained
observation (seats_sold / is_censored).  That lets us validate how
well each model recovers the true demand.

Usage (from Simulation root):
    python unconstraining/run_unconstrain.py            # all O-D pairs
    python unconstraining/run_unconstrain.py --od JP→US # one pair
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from models import (
    NaiveUnconstrainer,
    EMUnconstrainer,
    EMPriceUnconstrainer,
    PDUnconstrainer,
    PDPriceUnconstrainer,
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join(ROOT, "output", "simulation_slim.csv")
RESULTS_DIR = os.path.join(ROOT, "forecasting", "results")
MODELS = {
    "Naive":    NaiveUnconstrainer(),
    "EM":       EMUnconstrainer(),
    "EM+Price": EMPriceUnconstrainer(),
    "PD":       PDUnconstrainer(),
    "PD+Price": PDPriceUnconstrainer(),
}
COLORS = {
    "Naive":    "#999999",
    "EM":       "#3B6FA0",
    "EM+Price": "#5BA3D9",
    "PD":       "#2E8B57",
    "PD+Price": "#52C28A",
}


# ── Per-O-D unconstraining ────────────────────────────────────────────────────

def run_od(od_df: pd.DataFrame, od_label: str) -> pd.DataFrame:
    """
    Fit all unconstraining models to one O-D pair's daily data.
    Returns a validation summary DataFrame (one row per model).
    """
    od_df = od_df.sort_values("date").reset_index(drop=True)

    seats_obs   = od_df["seats_sold"].values.astype(float)
    latent      = od_df["latent_seats_sold"].values.astype(float)   # ground truth
    capacity    = int(od_df["seats_capacity"].iloc[0])
    price       = od_df["yield_overall_blended"].values.astype(float)
    is_cens     = od_df["is_censored"].values.astype(bool)

    n_cens = is_cens.sum()
    print(f"\n{'─'*60}")
    print(f"  O-D: {od_label}   |  {len(od_df):,} days  |  capacity={capacity}")
    print(f"  Censored: {n_cens} days ({n_cens/len(od_df):.1%})  "
          f"|  avg latent={latent.mean():.1f}  avg observed={seats_obs.mean():.1f}")

    rows = []
    ests  = {}
    for name, model in MODELS.items():
        kwargs = {"price_per_kg": price} if "Price" in name else {}
        est = model.fit(seats_obs, is_cens, capacity, **kwargs)
        ests[name] = est

        # Validation vs ground truth (only on censored days where it matters)
        mae_all   = np.abs(est - latent).mean()
        mae_cens  = np.abs(est[is_cens] - latent[is_cens]).mean() if n_cens else np.nan
        bias_cens = (est[is_cens] - latent[is_cens]).mean() if n_cens else np.nan
        lift      = est[is_cens].mean() - seats_obs[is_cens].mean() if n_cens else np.nan

        print(f"  {name:<12}  MAE_all={mae_all:5.1f}  "
              f"MAE_censored={mae_cens:5.1f}  bias={bias_cens:+.1f}  lift=+{lift:.1f}")

        rows.append({
            "od_pair":            od_label,
            "model":              name,
            "capacity":           capacity,
            "n_days":             len(od_df),
            "censored_days":      int(n_cens),
            "censored_rate":      round(n_cens / len(od_df), 4),
            "mean_observed":      round(seats_obs.mean(), 1),
            "mean_latent":        round(latent.mean(), 1),
            "mean_unconstrained": round(est.mean(), 1),
            "mae_all_days":       round(mae_all, 2),
            "mae_censored_days":  round(mae_cens, 2) if not np.isnan(mae_cens) else None,
            "bias_censored_days": round(bias_cens, 2) if not np.isnan(bias_cens) else None,
            "lift_censored":      round(lift, 2) if not np.isnan(lift) else None,
        })

    return pd.DataFrame(rows), ests, seats_obs, latent, is_cens, capacity


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_od(od_df, od_label, ests, seats_obs, latent, is_cens, capacity, out_dir):
    dates = od_df["date"]
    mask  = dates.dt.year == dates.dt.year.max()   # last simulated year

    fig, axes = plt.subplots(2, 1, figsize=(16, 11))

    # — Time series (last year zoom) ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(dates[mask], seats_obs[mask], color="black", lw=1.2, alpha=0.5,
            label="Constrained (boarded)", zorder=5)
    ax.plot(dates[mask], latent[mask], color="orange", lw=1.0, alpha=0.7,
            ls="--", label="Latent (true demand)", zorder=6)
    ax.axhline(capacity, color="red", lw=1.0, ls=":", alpha=0.6,
               label=f"Capacity ({capacity})")
    for name, est in ests.items():
        if name == "Naive":
            continue
        ax.plot(dates[mask], est[mask], color=COLORS[name], lw=0.9, alpha=0.85,
                label=name)
    cens_dates = dates[mask & is_cens]
    ax.scatter(cens_dates, [capacity] * len(cens_dates),
               color="red", s=8, alpha=0.4, zorder=7, label="Censored flights")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title(f"{od_label}  — Constrained vs Latent vs Unconstrained  "
                 f"({dates.dt.year.max()})", fontsize=11)
    ax.set_ylabel("Passengers")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    ax.grid(True, alpha=0.25)

    # — Distribution ──────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(seats_obs, bins=40, alpha=0.45, color="black",
             label="Constrained", density=True)
    ax2.hist(latent, bins=40, alpha=0.35, color="orange",
             label="Latent (truth)", density=True, histtype="step", lw=2.0)
    for name, est in ests.items():
        if name == "Naive":
            continue
        ax2.hist(est, bins=40, alpha=0.35, color=COLORS[name],
                 label=name, density=True, histtype="step", lw=1.5)
    ax2.axvline(capacity, color="red", lw=1.2, ls="--", alpha=0.7,
                label=f"Capacity ({capacity})")
    ax2.set_title("Demand Distribution: Constrained vs Latent vs Unconstrained", fontsize=11)
    ax2.set_xlabel("Passengers per Flight")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.25)

    plt.suptitle(f"Unconstraining Analysis — {od_label}", fontsize=13, y=1.01)
    plt.tight_layout()

    slug = od_label.replace("→", "_to_").replace(" ", "")
    out  = os.path.join(out_dir, f"unconstrained_{slug}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--od", default=None,
                        help="Single O-D pair to process (e.g. JP→US). Default: all.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Check required columns exist
    required = {"latent_seats_sold", "is_censored", "lf_raw"}
    missing  = required - set(df.columns)
    if missing:
        print(f"ERROR: simulation_slim.csv is missing columns: {missing}")
        print("Re-run  python scripts/run.py  to regenerate the simulation data.")
        sys.exit(1)

    # od_pair column is market_country → dest_country
    if "od_pair" not in df.columns:
        print("ERROR: od_pair column not found. Re-run the simulation.")
        sys.exit(1)

    od_pairs = [args.od] if args.od else sorted(df["od_pair"].unique())
    print(f"Processing {len(od_pairs)} O-D pair(s)...")

    all_summaries = []
    for od in od_pairs:
        sub = df[df["od_pair"] == od]
        if len(sub) < 30:
            print(f"  Skipping {od} (only {len(sub)} rows)")
            continue
        summary, ests, seats_obs, latent, is_cens, capacity = run_od(sub.copy(), od)
        all_summaries.append(summary)
        plot_od(sub, od, ests, seats_obs, latent, is_cens, capacity, RESULTS_DIR)

    # Combined summary
    combined = pd.concat(all_summaries, ignore_index=True)
    out_csv  = os.path.join(RESULTS_DIR, "unconstrained_summary.csv")
    combined.to_csv(out_csv, index=False)
    print(f"\n{'='*60}")
    print("OVERALL BEST MODEL PER O-D (by MAE on censored days)")
    print("="*60)
    best = (
        combined[combined["censored_days"] > 0]
        .sort_values("mae_censored_days")
        .groupby("od_pair")
        .first()[["model", "mae_censored_days", "bias_censored_days", "censored_rate"]]
    )
    print(best.to_string())
    print(f"\nFull results → {out_csv}")


if __name__ == "__main__":
    main()
