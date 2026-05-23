"""
batch_unconstrain.py
--------------------
Rolling-window unconstraining across all O-D pairs.

For each day t in each O-D pair, fits EM and PD on the preceding
WINDOW days of data only, then unconstrained the observation at t.
This ensures zero lookahead — no future information informs the estimate.

Inputs:  output/simulation_od_demand.csv
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

from models import EMUnconstrainer, PDUnconstrainer, NaiveUnconstrainer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join(ROOT, "output", "simulation_od_demand.csv")
OUT_DIR     = os.path.join(ROOT, "unconstraining", "results")
WINDOW      = 180    # rolling history window (days)
MIN_HISTORY = 60     # minimum days of history before unconstraining starts


# ── Per-O-D rolling unconstraining ────────────────────────────────────────────

def _process_od(args: tuple) -> tuple[str, pd.DataFrame]:
    od, group, window, min_history = args

    df = group.sort_values("date").reset_index(drop=True)
    n  = len(df)

    # Use modal capacity (capacity is fixed per O-D in this simulation)
    capacity = int(df["total_capacity"].mode()[0])

    obs     = df["constrained_seats"].values.astype(float)
    is_cens = (df["censored_flights"] > 0).values

    em_out = obs.copy()
    pd_out = obs.copy()

    em_model = EMUnconstrainer()
    pd_model = PDUnconstrainer()

    for t in range(min_history, n):
        start  = max(0, t - window)
        w_obs  = obs[start : t + 1]      # window ending AT t (inclusive)
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

    result = df[["date", "year", "month", "day", "od_pair",
                 "market_country", "dest_country"]].copy()
    result["constrained_seats"] = obs
    result["latent_seats"]      = df["latent_seats"].values      # oracle (ground truth)
    result["naive_seats"]       = obs                            # naive = constrained
    result["em_seats"]          = em_out
    result["pd_seats"]          = pd_out
    result["total_capacity"]    = capacity
    result["censored_flights"]  = df["censored_flights"].values
    result["is_censored"]       = is_cens.astype(int)

    return od, result


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

    od_pairs = sorted(df["od_pair"].unique())
    print(f"O-D pairs: {len(od_pairs)}  |  window={args.window}d  |  workers={args.workers}\n")

    tasks = [
        (od, df[df["od_pair"] == od].copy(), args.window, args.min_hist)
        for od in od_pairs
    ]

    all_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_od, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            od = futures[fut]
            try:
                od_name, result = fut.result()
                all_results[od_name] = result

                n_cens  = result["is_censored"].sum()
                em_lift = (result["em_seats"] - result["constrained_seats"]).mean()
                pd_lift = (result["pd_seats"] - result["constrained_seats"]).mean()
                print(f"  {od:<10}  {len(result):>5} days  "
                      f"censored={n_cens:>4} ({n_cens/len(result):.1%})  "
                      f"EM lift={em_lift:+.1f}  PD lift={pd_lift:+.1f}")
            except Exception as e:
                print(f"  {od}  ERROR: {e}")

    # Save per-O-D parquets
    for od, result in all_results.items():
        slug = od.replace("→", "_to_").replace(" ", "")
        out  = os.path.join(OUT_DIR, f"{slug}.parquet")
        result.to_parquet(out, index=False)

    # Save combined CSV
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "batch_unconstrained.csv")
    combined.to_csv(combined_path, index=False)

    print(f"\n  Saved {len(all_results)} parquet files → {OUT_DIR}/")
    print(f"  Combined CSV → {combined_path}  ({len(combined):,} rows)")

    # Validation: how close are EM and PD to oracle on censored days?
    print("\n── Validation vs Oracle (censored days only) ────────────────")
    print(f"{'O-D':<10}  {'EM MAE':>8}  {'PD MAE':>8}  {'Naive MAE':>10}")
    for od, result in sorted(all_results.items()):
        cens = result[result["is_censored"] == 1]
        if len(cens) == 0:
            continue
        truth = cens["latent_seats"].values
        em_mae    = np.abs(cens["em_seats"].values    - truth).mean()
        pd_mae    = np.abs(cens["pd_seats"].values    - truth).mean()
        naive_mae = np.abs(cens["naive_seats"].values - truth).mean()
        print(f"  {od:<10}  {em_mae:>8.1f}  {pd_mae:>8.1f}  {naive_mae:>10.1f}")


if __name__ == "__main__":
    main()
