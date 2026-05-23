"""
batch_unconstrain.py
--------------------
Rolling-window unconstraining at the individual flight level.

EM and PD are applied to each flight_od separately — where censoring
actually occurs (a flight hitting its own seat capacity) — then the
per-flight unconstrained estimates are summed to the O-D level.

Running unconstraining at the aggregate O-D level is incorrect because
multiple routes share the same market: aggregate demand almost never
reaches aggregate capacity even when individual flights are full.

Inputs:  output/simulation_slim.csv   (one row per flight-day)
Outputs: unconstraining/results/<od>.parquet  (one per O-D pair)
         unconstraining/results/batch_unconstrained.csv  (combined)

Usage (from Simulation root):
    python unconstraining/batch_unconstrain.py
    python unconstraining/batch_unconstrain.py --window 180 --workers 4
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))

from models import EMUnconstrainer, PDUnconstrainer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join(ROOT, "output", "simulation_slim.csv")
OUT_DIR     = os.path.join(ROOT, "unconstraining", "results")
WINDOW      = 180    # rolling history window (days)
MIN_HISTORY = 60     # minimum days before unconstraining starts


# ── Per-flight rolling unconstraining ─────────────────────────────────────────

def _process_flight(args: tuple) -> tuple[str, pd.DataFrame]:
    """
    Run rolling-window EM and PD on one flight_od series.
    Returns unconstrained estimates at the per-flight level.
    """
    flight_id, group, window, min_history = args

    df = group.sort_values("date").reset_index(drop=True)
    n  = len(df)

    capacity = int(df["seats_capacity"].mode()[0])

    obs     = df["seats_sold"].values.astype(float)
    is_cens = df["is_censored"].values.astype(bool)

    em_out = obs.copy()
    pd_out = obs.copy()

    em_model = EMUnconstrainer()
    pd_model = PDUnconstrainer()

    for t in range(min_history, n):
        start  = max(0, t - window)
        w_obs  = obs[start : t + 1]      # window up to and including day t
        w_cens = is_cens[start : t + 1]

        if w_cens.sum() < 5:             # need enough censored points to fit
            continue

        try:
            em_out[t] = em_model.fit(w_obs, w_cens, capacity)[-1]
        except Exception:
            pass

        try:
            pd_out[t] = pd_model.fit(w_obs, w_cens, capacity)[-1]
        except Exception:
            pass

    result = df[["date", "year", "month", "day", "flight_od", "od_pair",
                 "market_country", "dest_country"]].copy()
    result["seats_sold"]        = obs
    result["latent_seats_sold"] = df["latent_seats_sold"].values
    result["seats_capacity"]    = capacity
    result["is_censored"]       = is_cens.astype(int)
    result["em_seats"]          = em_out
    result["pd_seats"]          = pd_out

    return flight_id, result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",   type=int, default=WINDOW,      help="Rolling window size (days)")
    parser.add_argument("--min-hist", type=int, default=MIN_HISTORY, help="Min history before unconstraining")
    parser.add_argument("--workers",  type=int, default=4,           help="Parallel workers")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    flight_ids = sorted(df["flight_od"].unique())
    print(f"Flights: {len(flight_ids)}  |  window={args.window}d  |  workers={args.workers}\n")

    tasks = [
        (fid, df[df["flight_od"] == fid].copy(), args.window, args.min_hist)
        for fid in flight_ids
    ]

    # ── Run EM/PD per individual flight ───────────────────────────────────────
    all_flight_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_flight, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            fid = futures[fut]
            try:
                flight_id, result = fut.result()
                all_flight_results[flight_id] = result

                n_cens  = result["is_censored"].sum()
                em_lift = (result["em_seats"] - result["seats_sold"]).mean()
                pd_lift = (result["pd_seats"] - result["seats_sold"]).mean()
                cap     = result["seats_capacity"].iloc[0]
                print(f"  {fid:<25}  cap={cap:>3}  "
                      f"censored={n_cens:>4} ({n_cens/len(result):.1%})  "
                      f"EM lift={em_lift:+.1f}  PD lift={pd_lift:+.1f}")
            except Exception as e:
                print(f"  {fid}  ERROR: {e}")

    # ── Aggregate from flight level to O-D level ───────────────────────────────
    print("\nAggregating to O-D level...")
    all_flights_df = pd.concat(all_flight_results.values(), ignore_index=True)

    od_df = (
        all_flights_df
        .groupby(["date", "year", "month", "day", "od_pair",
                  "market_country", "dest_country"])
        .agg(
            constrained_seats = ("seats_sold",         "sum"),
            latent_seats      = ("latent_seats_sold",  "sum"),
            total_capacity    = ("seats_capacity",     "sum"),
            censored_flights  = ("is_censored",        "sum"),
            em_seats          = ("em_seats",           "sum"),
            pd_seats          = ("pd_seats",           "sum"),
        )
        .reset_index()
    )
    od_df["naive_seats"] = od_df["constrained_seats"]   # naive = constrained
    od_df["is_censored"] = (od_df["censored_flights"] > 0).astype(int)

    # ── Save per-O-D parquets ─────────────────────────────────────────────────
    for od in sorted(od_df["od_pair"].unique()):
        g    = od_df[od_df["od_pair"] == od]
        slug = od.replace("→", "_to_").replace(" ", "")
        out  = os.path.join(OUT_DIR, f"{slug}.parquet")
        g.to_parquet(out, index=False)

    # ── Save combined CSV ─────────────────────────────────────────────────────
    combined_path = os.path.join(OUT_DIR, "batch_unconstrained.csv")
    od_df.to_csv(combined_path, index=False)

    print(f"\n  Saved {od_df['od_pair'].nunique()} O-D parquet files → {OUT_DIR}/")
    print(f"  Combined CSV → {combined_path}  ({len(od_df):,} rows)")

    # ── Validation: EM/PD vs oracle on flight-censored days ───────────────────
    print("\n── Validation vs Oracle (flight-censored days only) ─────────────")
    print(f"{'O-D':<12}  {'EM MAE':>8}  {'PD MAE':>8}  {'Naive MAE':>10}  {'cens%':>6}")
    for od in sorted(od_df["od_pair"].unique()):
        g    = od_df[od_df["od_pair"] == od]
        cens = g[g["is_censored"] == 1]
        if len(cens) == 0:
            continue
        truth     = cens["latent_seats"].values
        em_mae    = np.abs(cens["em_seats"].values    - truth).mean()
        pd_mae    = np.abs(cens["pd_seats"].values    - truth).mean()
        naive_mae = np.abs(cens["naive_seats"].values - truth).mean()
        cens_pct  = len(cens) / len(g)
        print(f"  {od:<12}  {em_mae:>8.1f}  {pd_mae:>8.1f}  {naive_mae:>10.1f}  {cens_pct:>6.1%}")


if __name__ == "__main__":
    main()
