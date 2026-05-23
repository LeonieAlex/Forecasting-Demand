"""
simulation.py
-------------
Discrete booking-window simulation for a single flight departure.

Adopted from reference architecture (KorawutMing/AirCargoSynthesizedDataset),
adapted for passenger demand with cabin-level FCFS booking engine.

For each departure date, Poisson arrivals are drawn independently for each
cabin class at each day-prior-to-departure (DTP) step, using Beta-distribution
booking curves to shape the arrival pattern.  FCFS capacity management then
accepts or rejects each cabin's demand — producing natural censoring when a
cabin sells out.

This replaces the simple  floor(od_latent × share)  calculation in the engine
with a richer stochastic process that:
  1. Generates a realistic booking build-up trajectory (BOH)
  2. Censors naturally when a cabin hits its hard seat limit
  3. Produces oracle (latent) vs realized (observed) at the cabin level

Shocks and seasonality are already baked into `expected_demand` (the Poisson
mean) before this function is called — they remain your ShockEngine /
seasonality.py multipliers, applied upstream in demand.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Standard RM snapshot dates (days prior to departure).
# Entries beyond the booking window will be recorded as 0.
BOH_SNAPSHOTS: list[int] = [330, 270, 180, 90, 60, 45, 30, 21, 14, 7, 3, 1]

CABINS: list[str] = ["first", "business", "premium_eco", "economy"]


def simulate_flight_booking_window(
    capacity_by_cabin: dict[str, int],
    expected_demand: float,
    booking_curves: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate the booking process for one flight on one departure date.

    Parameters
    ----------
    capacity_by_cabin : dict
        Fixed seats per cabin, e.g. {"first": 8, "business": 52,
        "premium_eco": 24, "economy": 312}.  Cabins with 0 seats are skipped.
    expected_demand : float
        Total expected passengers for this flight (Poisson mean).
        This already incorporates OD market size, seasonal, DOW, trend,
        and shock multipliers from the upstream demand model.
    booking_curves : pd.DataFrame
        Output of generate_booking_curves().
        Index = days_prior (high → low), columns = cabin names.
        Values = fraction of total cabin demand arriving on that specific day.
    rng : np.random.Generator

    Returns
    -------
    dict with keys:
        latent_seats, observed_seats, is_censored
        oracle_{cabin}_pax, observed_{cabin}_pax  (for each cabin in CABINS)
        boh_dtp_{n}  (for each n in BOH_SNAPSHOTS)
    """
    total_capacity = sum(capacity_by_cabin.values())

    if total_capacity == 0 or expected_demand <= 0:
        return _empty_result()

    # Sort DTP from booking-open (highest) to departure (1)
    dtps = sorted(booking_curves.index, reverse=True)
    dtp_to_idx = {dtp: i for i, dtp in enumerate(dtps)}

    cabin_latent:   dict[str, int] = {}
    cabin_realized: dict[str, int] = {}
    is_censored = False
    cum_boh = np.zeros(len(dtps))   # cumulative realized pax across all cabins

    for cab in CABINS:
        cap = capacity_by_cabin.get(cab, 0)

        if cap == 0 or cab not in booking_curves.columns:
            cabin_latent[cab]   = 0
            cabin_realized[cab] = 0
            continue

        weight = cap / total_capacity

        # Vectorised Poisson draw for all DTPs at once
        lams     = expected_demand * weight * booking_curves.loc[dtps, cab].values
        arrivals = rng.poisson(lams)                     # shape (n_dtps,)

        cumsum       = np.cumsum(arrivals)               # cumulative arrivals
        cum_accepted = np.minimum(cumsum, cap)           # FCFS cap

        cabin_latent[cab]   = int(arrivals.sum())
        cabin_realized[cab] = int(min(int(arrivals.sum()), cap))

        if int(arrivals.sum()) > cap:
            is_censored = True

        cum_boh += cum_accepted

    # BOH snapshots
    boh_snapshots: dict[int, int] = {}
    for snap in BOH_SNAPSHOTS:
        if snap in dtp_to_idx:
            boh_snapshots[snap] = int(cum_boh[dtp_to_idx[snap]])
        else:
            boh_snapshots[snap] = 0   # beyond booking window

    latent_seats   = sum(cabin_latent.values())
    observed_seats = sum(cabin_realized.values())

    result: dict = {
        "latent_seats":   latent_seats,
        "observed_seats": observed_seats,
        "is_censored":    int(is_censored),
    }
    for cab in CABINS:
        result[f"oracle_{cab}_pax"]   = cabin_latent.get(cab, 0)
        result[f"observed_{cab}_pax"] = cabin_realized.get(cab, 0)
    for snap in sorted(BOH_SNAPSHOTS, reverse=True):
        result[f"boh_dtp_{snap}"] = boh_snapshots[snap]

    return result


def _empty_result() -> dict:
    result: dict = {"latent_seats": 0, "observed_seats": 0, "is_censored": 0}
    for cab in CABINS:
        result[f"oracle_{cab}_pax"]   = 0
        result[f"observed_{cab}_pax"] = 0
    for snap in BOH_SNAPSHOTS:
        result[f"boh_dtp_{snap}"] = 0
    return result
