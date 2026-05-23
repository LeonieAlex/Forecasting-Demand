"""
batch_unconstrain.py
--------------------
Rolling-window unconstraining on flights_latent.csv (daily per-flight data).

Each flight is processed independently with its own FIXED scalar capacity —
the seats on that aircraft type never change (HND→LAX is always a B777 at
396 seats; MEL→LAX is always an A380 at 491 seats).  This is correct because
censoring happens at the individual flight level, not at the corridor level.

Connecting flights (e.g. ICN→LAX carrying CN pax) are included and processed
with their own capacity, exactly like direct flights.

After per-flight unconstraining, estimates are aggregated to OD pair level
so you can compare EM/PD recovery against the oracle latent_seats in
OD_LatentDemand.csv.

Inputs:
    data/flights_latent.csv

Outputs:
    unconstraining/results/<flight_od>.parquet   (one per flight)
    unconstraining/results/od_unconstrained.csv  (aggregated to OD level)

Usage (from project root):
    python unconstraining/batch_unconstrain.py
    python unconstraining/batch_unconstrain.py --window 365 --workers 4
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

DATA_PATH    = os.path.join(ROOT, "data", "flights_latent.csv")
OUT_DIR      = os.path.join(ROOT, "unconstraining", "results")
WINDOW       = 365    # rolling window — 1 full year of daily data
MIN_HISTORY  = 90     # days before fitting starts
MIN_CENSORED = 5      # censored days needed in window before EM/PD runs


# ── Per-flight rolling unconstraining ─────────────────────────────────────────

def _process_flight(args: tuple) -> tuple[str, pd.DataFrame]:
    """
    Run rolling-window EM and PD for one flight_od time series.

    capacity   — fixed scalar: the aircraft on this route always has the
                 same number of seats.  No averaging, no median.
    is_censored — True only when this specific flight completely sold out
                 (observed_seats == seats_capacity), i.e. latent demand
                 actually hit the hard ceiling.
    """
    flight_id, group, window, min_history, min_censored = args

    df = group.sort_values("date").reset_index(drop=True)
    n  = len(df)

    # Fixed capacity for this flight — same aircraft type every day
    capacity = int(df["seats_capacity"].iloc[0])

    obs     = df["observed_seats"].values.astype(float)
    latent  = df["latent_seats"].values.astype(float)
    is_cens = df["is_censored"].values.astype(bool)

    em_out = obs.copy()
    pd_out = obs.copy()

    em_model = EMUnconstrainer()
    pd_model = PDUnconstrainer()

    for t in range(min_history, n):
        start  = max(0, t - window)
        w_obs  = obs[start : t + 1]
        w_cens = is_cens[start : t + 1]

        if w_cens.sum() < min_censored:
            continue

        try:
            em_out[t] = em_model.fit(w_obs, w_cens, capacity)[-1]
        except Exception:
            pass

        try:
            pd_out[t] = pd_model.fit(w_obs, w_cens, capacity)[-1]
        except Exception:
            pass

    result = df[["date", "year", "month", "day",
                 "flight_od", "od_pair",
                 "market_country", "dest_country",
                 "aircraft_type", "seats_capacity",
                 "is_connecting"]].copy()
    result["latent_seats"]     = latent
    result["observed_seats"]   = obs
    result["is_censored"]      = is_cens.astype(int)
    result["em_unconstrained"] = em_out
    result["pd_unconstrained"] = pd_out

    return flight_id, result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",       type=int, default=WINDOW)
    parser.add_argument("--min-hist",     type=int, default=MIN_HISTORY)
    parser.add_argument("--min-censored", type=int, default=MIN_CENSORED)
    parser.add_argument("--workers",      type=int, default=4)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    flight_ids = sorted(df["flight_od"].unique())
    print(f"Flights: {len(flight_ids)}  |  "
          f"window={args.window}d  |  min_history={args.min_hist}d  |  "
          f"workers={args.workers}\n")

    tasks = [
        (fid, df[df["flight_od"] == fid].copy(),
         args.window, args.min_hist, args.min_censored)
        for fid in flight_ids
    ]

    all_flight_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_flight, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            fid = futures[fut]
            try:
                flight_id, result = fut.result()
                all_flight_results[flight_id] = result

                cap     = result["seats_capacity"].iloc[0]
                n_cens  = result["is_censored"].sum()
                em_lift = (result["em_unconstrained"] - result["observed_seats"]).mean()
                pd_lift = (result["pd_unconstrained"] - result["observed_seats"]).mean()
                print(f"  {flight_id:<28}  cap={cap:>3}  "
                      f"censored={n_cens:>4} ({n_cens/len(result):.1%})  "
                      f"EM lift={em_lift:+.1f}  PD lift={pd_lift:+.1f}")
            except Exception as e:
                print(f"  {fid}  ERROR: {e}")

    # ── Save per-flight parquets ──────────────────────────────────────────────
    for fid, result in all_flight_results.items():
        slug = fid.replace("->", "_to_").replace(" ", "_").replace("(", "").replace(")", "")
        result.to_parquet(os.path.join(OUT_DIR, f"{slug}.parquet"), index=False)

    # ── Aggregate to OD level ─────────────────────────────────────────────────
    print("\nAggregating to OD level...")
    all_flights_df = pd.concat(all_flight_results.values(), ignore_index=True)

    od_df = (
        all_flights_df
        .groupby(["date", "year", "month", "day",
                  "od_pair", "market_country", "dest_country"])
        .agg(
            latent_seats      = ("latent_seats",     "sum"),
            observed_seats    = ("observed_seats",   "sum"),
            em_unconstrained  = ("em_unconstrained", "sum"),
            pd_unconstrained  = ("pd_unconstrained", "sum"),
            total_capacity    = ("seats_capacity",   "sum"),
            n_flights         = ("flight_od",        "count"),
            censored_flights  = ("is_censored",      "sum"),
        )
        .reset_index()
    )
    od_df["any_flight_full"] = (od_df["censored_flights"] > 0).astype(int)

    od_path = os.path.join(OUT_DIR, "od_unconstrained.csv")
    od_df.to_csv(od_path, index=False)

    print(f"\n  Saved {len(all_flight_results)} per-flight parquets → {OUT_DIR}/")
    print(f"  OD-level CSV → {od_path}  ({len(od_df):,} rows)")

    # ── Validation: EM/PD vs oracle on censored flights ───────────────────────
    print("\n── Validation vs Oracle (censored flight-days only) ─────────")
    print(f"{'Flight':<28}  {'cap':>4}  {'EM MAE':>8}  {'PD MAE':>8}  {'cens%':>6}")
    for fid, result in sorted(all_flight_results.items()):
        cens = result[result["is_censored"] == 1]
        if len(cens) == 0:
            continue
        truth  = cens["latent_seats"].values
        em_mae = np.abs(cens["em_unconstrained"].values - truth).mean()
        pd_mae = np.abs(cens["pd_unconstrained"].values - truth).mean()
        cap    = result["seats_capacity"].iloc[0]
        pct    = len(cens) / len(result)
        print(f"  {fid:<28}  {cap:>4}  {em_mae:>8.1f}  {pd_mae:>8.1f}  {pct:>6.1%}")


if __name__ == "__main__":
    main()
