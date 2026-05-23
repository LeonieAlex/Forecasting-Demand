"""
batch_unconstrain.py
--------------------
Rolling-window unconstraining on flights_truncated.csv.

Mirrors the reference architecture (KorawutMing/AirCargoSynthesizedDataset)
but adapted for passenger cabins instead of cargo commodity segments.

Segments = cabin types: first, business, premium_eco, economy.
Each cabin is unconstrained independently with its own hard capacity limit.

Per-cabin censoring: a cabin is censored when it sold out completely
(cab_pax == cabin_capacity).  This is more precise than the flight-level
is_censored flag — economy may sell out while business does not.

Models applied per cabin:
  Naive       — baseline, returns observed (no lift)
  EM          — truncated-normal Expectation-Maximisation (Numba)
  PD_{tau}    — Projection-Detruncation at tau in {0.1, 0.3, 0.5, 0.7, 0.9}

Inputs:
    data/flights_truncated.csv

Outputs:
    unconstraining/results/{flight_od}.parquet   (one per flight)
    unconstraining/results/od_unconstrained.csv  (aggregated to OD level)

Validation (optional):
    If data/flights_latent.csv exists the script compares EM and PD(0.5)
    estimates against oracle_{cab}_pax for each censored cabin-day.

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

from models import NaiveUnconstrainer, EMUnconstrainer, PDUnconstrainer

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

CABINS = ["first", "business", "premium_eco", "economy"]

DATA_PATH    = os.path.join(ROOT, "data", "flights_truncated.csv")
OUT_DIR      = os.path.join(ROOT, "unconstraining", "results")
WINDOW       = 365   # rolling lookback — 1 full year of daily data
MIN_HISTORY  = 90    # days before fitting starts
MIN_CENSORED = 5     # min censored observations in window before EM/PD runs

# Model registry: name → (Class, output_prefix, extra_kwargs)
MODEL_SPECS: dict[str, tuple] = {
    "Naive": (NaiveUnconstrainer, "naive", {}),
    "EM":    (EMUnconstrainer,    "em",    {}),
}
for _tau in [0.1, 0.3, 0.5, 0.7, 0.9]:
    _key = str(_tau).replace(".", "")   # "0.3" → "03"
    MODEL_SPECS[f"PD_{_tau}"] = (PDUnconstrainer, f"pd{_key}", {"tau": _tau})


# ── Per-flight processing ─────────────────────────────────────────────────────

def _process_flight(args: tuple) -> tuple[str, pd.DataFrame]:
    """
    Rolling-window unconstraining for one flight_od, independently per cabin.

    Capacity is the fixed scalar for that cabin on this aircraft type — same
    every day on the same route (B777 economy is always 312 seats).
    Censoring is per-cabin: obs_pax == cabin_capacity (cabin sold out).
    """
    flight_id, group, window, min_history, min_censored = args

    df = group.sort_values("date").reset_index(drop=True)
    n  = len(df)

    # Pre-allocate: est[cabin][prefix] = array of estimates (starts at observed)
    est: dict[str, dict[str, np.ndarray]] = {
        cab: {
            prefix: df[f"{cab}_pax"].values.astype(float).copy()
            for _, prefix, _ in MODEL_SPECS.values()
        }
        for cab in CABINS
    }

    for cab in CABINS:
        cap_col = f"{cab}_capacity"
        pax_col = f"{cab}_pax"

        if pax_col not in df.columns or cap_col not in df.columns:
            continue

        cabin_cap = float(df[cap_col].iloc[0])   # fixed scalar — never changes
        if cabin_cap == 0:
            continue

        obs_arr  = df[pax_col].values.astype(float)
        cens_arr = (obs_arr >= cabin_cap).astype(bool)   # per-cabin censoring

        # Fresh model instances per cabin to avoid cross-contamination
        active_models = {
            name: (ModelClass(), prefix, kwargs)
            for name, (ModelClass, prefix, kwargs) in MODEL_SPECS.items()
        }

        for t in range(min_history, n):
            if not cens_arr[t]:
                # Uncensored: all models return observed — no lift applied
                for _, (_, prefix, _) in active_models.items():
                    est[cab][prefix][t] = obs_arr[t]
                continue

            start  = max(0, t - window)
            w_obs  = obs_arr[start : t + 1]
            w_cens = cens_arr[start : t + 1]

            if w_cens.sum() < min_censored:
                continue

            for name, (model, prefix, model_kwargs) in active_models.items():
                try:
                    imputed = model.fit(
                        observed_bookings=w_obs,
                        is_censored=w_cens,
                        capacity=cabin_cap,
                        max_iter=50,
                        **model_kwargs,
                    )
                    # Take last element (today's estimate); never go below observed
                    est[cab][prefix][t] = max(float(imputed[-1]), obs_arr[t])
                except Exception:
                    fallback = np.mean(w_obs[~w_cens]) if (~w_cens).any() else obs_arr[t]
                    est[cab][prefix][t] = max(float(fallback), obs_arr[t])

    # ── Build output DataFrame ────────────────────────────────────────────────
    keep = [
        "date", "year", "month", "day", "dow",
        "flight_od", "od_pair", "market_country", "dest_country",
        "is_connecting", "aircraft_type", "seats_capacity",
        "first_capacity", "business_capacity", "premium_eco_capacity", "economy_capacity",
        "pax", "is_censored",
        "first_pax", "business_pax", "premium_eco_pax", "economy_pax",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()

    for cab in CABINS:
        for _, prefix, _ in MODEL_SPECS.values():
            out[f"{prefix}_{cab}_est"] = est[cab][prefix]

    # Aggregate cabin estimates to flight total per model
    for _, prefix, _ in MODEL_SPECS.values():
        out[f"{prefix}_total_est"] = sum(
            out[f"{prefix}_{cab}_est"] for cab in CABINS
        )

    return flight_id, out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-cabin rolling unconstraining")
    parser.add_argument("--window",       type=int, default=WINDOW)
    parser.add_argument("--min-hist",     type=int, default=MIN_HISTORY)
    parser.add_argument("--min-censored", type=int, default=MIN_CENSORED)
    parser.add_argument("--workers",      type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found — run scripts/run.py first.")
        sys.exit(1)

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    flight_ids = sorted(df["flight_od"].unique())

    print(f"Flights : {len(flight_ids)}")
    print(f"Cabins  : {CABINS}")
    print(f"Models  : {list(MODEL_SPECS.keys())}")
    print(f"Window  : {args.window}d  |  min_history={args.min_hist}d  |  "
          f"workers={args.workers}\n")

    tasks = [
        (fid, df[df["flight_od"] == fid].copy(),
         args.window, args.min_hist, args.min_censored)
        for fid in flight_ids
    ]

    all_results: dict[str, pd.DataFrame] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_flight, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            fid = futures[fut]
            try:
                flight_id, result = fut.result()
                all_results[flight_id] = result

                cap      = result["seats_capacity"].iloc[0]
                n_cens   = result["is_censored"].sum()
                em_lift  = (result["em_total_est"]  - result["pax"]).mean()
                pd5_lift = (result["pd05_total_est"] - result["pax"]).mean()
                print(f"  {flight_id:<28}  cap={cap:>3}  "
                      f"censored={n_cens:>4} ({n_cens/len(result):.1%})  "
                      f"EM lift={em_lift:+.1f}  PD(0.5) lift={pd5_lift:+.1f}")
            except Exception as e:
                print(f"  {fid}  ERROR: {e}")

    # ── Save per-flight parquets ──────────────────────────────────────────────
    for fid, result in all_results.items():
        slug = (fid.replace("->", "_to_").replace(" ", "_")
                   .replace("(", "").replace(")", ""))
        result.to_parquet(os.path.join(OUT_DIR, f"{slug}.parquet"), index=False)

    # ── Aggregate to OD level ─────────────────────────────────────────────────
    print("\nAggregating to OD level ...")
    all_df = pd.concat(all_results.values(), ignore_index=True)

    agg_dict: dict = {
        "pax":              ("pax",            "sum"),
        "seats_capacity":   ("seats_capacity", "sum"),
        "n_flights":        ("flight_od",      "count"),
        "censored_flights": ("is_censored",    "sum"),
    }
    for _, prefix, _ in MODEL_SPECS.values():
        agg_dict[f"{prefix}_total_est"] = (f"{prefix}_total_est", "sum")

    od_df = (
        all_df
        .groupby(["date", "year", "month", "day",
                  "od_pair", "market_country", "dest_country"])
        .agg(**agg_dict)
        .reset_index()
    )

    od_path = os.path.join(OUT_DIR, "od_unconstrained.csv")
    od_df.to_csv(od_path, index=False)
    print(f"\n  Saved {len(all_results)} per-flight parquets → {OUT_DIR}/")
    print(f"  OD-level CSV → {od_path}  ({len(od_df):,} rows)")

    # ── Validation vs oracle (if flights_latent.csv available) ───────────────
    latent_path = os.path.join(ROOT, "data", "flights_latent.csv")
    if not os.path.exists(latent_path):
        return

    print("\n── Validation vs Oracle (censored cabin-days) ───────────────────")
    oracle = pd.read_csv(latent_path, parse_dates=["date"])

    hdr = f"{'Flight':<28}  {'cab':<12}  {'EM MAE':>8}  {'PD(0.5) MAE':>11}  {'cens%':>6}"
    print(hdr)

    for fid in sorted(all_results.keys()):
        res = all_results[fid]
        orc = oracle[oracle["flight_od"] == fid].sort_values("date").reset_index(drop=True)
        if len(orc) == 0:
            continue
        merged = res.merge(
            orc[["date"] + [f"oracle_{c}_pax" for c in CABINS]],
            on="date", how="inner",
        )
        for cab in CABINS:
            cab_cens = merged[f"{cab}_pax"] >= merged[f"{cab}_capacity"]
            sub = merged[cab_cens]
            if len(sub) == 0:
                continue
            truth   = sub[f"oracle_{cab}_pax"].values
            em_mae  = np.abs(sub[f"em_{cab}_est"].values  - truth).mean()
            pd5_mae = np.abs(sub[f"pd05_{cab}_est"].values - truth).mean()
            pct     = cab_cens.mean()
            print(f"  {fid:<28}  {cab:<12}  {em_mae:>8.1f}  {pd5_mae:>11.1f}  {pct:>6.1%}")


if __name__ == "__main__":
    main()
