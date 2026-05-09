"""
booking_curves.py
-----------------
Beta-distribution booking curves for passenger demand segments.

CONCEPT — READING DATE
-----------------------
In airline Revenue Management, a "reading date" is the snapshot date at which
you observe how many seats have been sold, expressed as days prior to departure.

    Departure date:   2025-08-15
    Reading date:     2025-08-01
    Days prior (DTP): 14

This module models the *shape* of booking build-up for each cabin/segment
over a booking window (default: 90 days prior to departure).

A Beta distribution is used because it is bounded on [0, 1], flexible enough
to model early-bookers (first class, corporate), last-minute bookers (leisure
economy), and everything in between by choosing (alpha, beta) parameters.

    alpha < beta  → right-skewed → early booking (business/first)
    alpha > beta  → left-skewed  → late booking (leisure economy)
    alpha ≈ beta  → symmetric    → balanced (premium economy)

OUTPUT
------
generate_booking_curves() returns a DataFrame:
  - Index: days_prior (BOOKING_WINDOW_DAYS down to 1)
  - Columns: one per segment (economy, premium_eco, business, first)
  - Values: fraction of total segment demand arriving on that day
            (sums to 1.0 per column over the full window)

simulate_reading_date() returns the cumulative % booked at a specific DTP,
which is what you'd observe in a real PNR/booking file snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import beta as beta_dist


# ── Constants ──────────────────────────────────────────────────────────────────

BOOKING_WINDOW_DAYS = 330   # how far in advance bookings open

# Beta distribution parameters per segment.
# (alpha, beta) chosen to reflect real transpacific booking behavior:
#   - First:       mostly corporate / premium leisure, booked 60–90 days out
#   - Business:    mix of corporate (early) and upgrade (late)
#   - Premium eco: leisure-premium, books 30–60 days out
#   - Economy:     broadest spread; meaningful last-minute tail
#
# Interpretation of (alpha, beta):
#   alpha controls where mass concentrates as a fraction of the window [0,1].
#   Day 90 prior = t=0 in the Beta CDF; day 1 prior = t=1.
#   So a right-skewed Beta (alpha<beta) → more early bookings → business/first.

SEGMENTS: dict[str, dict] = {
    "first": {
        "alpha": 1.5,
        "beta":  5.0,
        # Right-skewed: bulk of first-class bookings 75–90 days prior
        # Reflects corporate/travel-agent early commitment for suite products
        "description": "First — early, corporate, long lead-time",
    },
    "business": {
        "alpha": 2.0,
        "beta":  4.0,
        # Moderately early: most business bookings 45–75 days out
        # Corporate managed travel often has 3-week advance requirement
        "description": "Business — moderate-early, mixed corporate/leisure",
    },
    "premium_eco": {
        "alpha": 3.0,
        "beta":  4.0,
        # Near-symmetric, slight lean toward 30–60 days out
        # Premium leisure — plan but book when deals appear
        "description": "Premium Economy — balanced, leisure-premium",
    },
    "economy": {
        "alpha": 4.5,
        "beta":  3.0,
        # Left-skewed: significant last-minute and mid-window tail
        # Includes VFR, backpackers, LCC-switchers, last-minute deals
        "description": "Economy — late-leaning, broadest spread",
    },
}


# ── Core functions ─────────────────────────────────────────────────────────────

@dataclass
class BookingCurveParams:
    booking_window_days: int = BOOKING_WINDOW_DAYS
    segments: dict = None   # use SEGMENTS default if None


def generate_booking_curves(
    params: BookingCurveParams | None = None,
) -> pd.DataFrame:
    """
    Generate daily arrival fractions for each demand segment over the
    booking window using Beta distributions.

    Returns
    -------
    pd.DataFrame
        Index:   days_prior (int, BOOKING_WINDOW_DAYS → 1)
        Columns: one per segment key in SEGMENTS
        Values:  fraction of total segment demand arriving on that specific day
                 (each column sums to exactly 1.0)
    """
    if params is None:
        params = BookingCurveParams()

    segments = params.segments or SEGMENTS
    window   = params.booking_window_days

    # t in [0,1] maps to the booking window.
    # t=0 = window_days prior to departure (first booking opportunity)
    # t=1 = day of departure
    time_points = np.linspace(0, 1, window + 1)

    curves: dict[str, np.ndarray] = {}

    for segment, seg_params in segments.items():
        a = seg_params["alpha"]
        b = seg_params["beta"]

        # CDF at each time boundary
        cdf_values = beta_dist.cdf(time_points, a, b)

        # Daily fraction = difference in CDF between consecutive day boundaries
        daily_fractions = np.diff(cdf_values)

        # Normalise to correct floating-point drift (sum must be exactly 1.0)
        daily_fractions = daily_fractions / daily_fractions.sum()

        curves[segment] = daily_fractions

    df = pd.DataFrame(curves)

    # days_prior counts DOWN: first row = BOOKING_WINDOW_DAYS days before departure
    df["days_prior"] = list(range(window, 0, -1))
    df = df.set_index("days_prior")

    return df


def cumulative_booking_curve(df_curves: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily arrival fractions into cumulative booking curves.

    Returns a DataFrame with the same shape as df_curves but values represent
    the CUMULATIVE fraction booked from booking_open up to each days_prior point.

    Index is still days_prior (high → low = approaching departure).
    """
    # Reverse so cumsum runs from earliest (highest days_prior) to latest
    return df_curves.iloc[::-1].cumsum().iloc[::-1]


def simulate_reading_date(
    df_curves: pd.DataFrame,
    days_prior: int,
    total_seats_by_segment: dict[str, int],
    noise_sigma: float = 0.03,
    rng: np.random.Generator | None = None,
) -> dict[str, dict]:
    """
    Simulate a booking snapshot at a specific reading date (days_prior to departure).

    This answers: "If I read the PNR file today (X days before departure),
    how many seats in each segment have been sold?"

    Parameters
    ----------
    df_curves : pd.DataFrame
        Output of generate_booking_curves()
    days_prior : int
        The reading date, expressed as days before departure.
        Must be within [1, booking_window_days].
    total_seats_by_segment : dict[str, int]
        Final expected seats sold per segment (from demand simulation).
        This is the "eventual total" that the booking curve builds toward.
    noise_sigma : float
        Day-to-day stochastic noise around the smooth curve (default 3%).
    rng : np.random.Generator, optional

    Returns
    -------
    dict keyed by segment, each containing:
        booked_seats   : int   — seats sold as of reading date
        pct_booked     : float — fraction of segment sold
        remaining_seats: int   — seats still to be sold
        on_curve       : float — smooth curve value (no noise)
    """
    if rng is None:
        rng = np.random.default_rng()

    cumulative = cumulative_booking_curve(df_curves)

    # Clip days_prior to valid range
    max_dtp = df_curves.index.max()
    days_prior = int(np.clip(days_prior, 1, max_dtp))

    results: dict[str, dict] = {}

    for segment, total_seats in total_seats_by_segment.items():
        if segment not in cumulative.columns:
            continue

        # Smooth cumulative fraction from Beta CDF
        on_curve = float(cumulative.loc[days_prior, segment])

        # Add booking noise (bookings are lumpy, not perfectly smooth)
        noise = 1.0 + noise_sigma * rng.standard_normal()
        noisy_fraction = float(np.clip(on_curve * noise, 0.0, 1.0))

        booked = round(total_seats * noisy_fraction)
        booked = int(np.clip(booked, 0, total_seats))

        results[segment] = {
            "booked_seats":    booked,
            "pct_booked":      round(noisy_fraction, 4),
            "remaining_seats": total_seats - booked,
            "on_curve":        round(on_curve, 4),
        }

    return results


def reading_date_series(
    df_curves: pd.DataFrame,
    total_seats_by_segment: dict[str, int],
    reading_dates_dtp: list[int] | None = None,
    noise_sigma: float = 0.03,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Simulate multiple reading date snapshots for one flight.

    Useful for generating a full booking build-up time series
    (e.g. snapshots at DTP 360, 270, 180, 90, 60, 45, 30, 21, 14, 7, 3, 1).

    Returns
    -------
    pd.DataFrame with MultiIndex (segment, days_prior) and columns:
        booked_seats, pct_booked, remaining_seats, on_curve
    """
    if rng is None:
        rng = np.random.default_rng()

    if reading_dates_dtp is None:
        # Standard RM reading dates (common in airline ops)
        reading_dates_dtp = [360, 270, 180, 90, 60, 45, 30, 21, 14, 7, 3, 1]

    rows = []
    for dtp in reading_dates_dtp:
        snapshot = simulate_reading_date(
            df_curves=df_curves,
            days_prior=dtp,
            total_seats_by_segment=total_seats_by_segment,
            noise_sigma=noise_sigma,
            rng=rng,
        )
        for segment, data in snapshot.items():
            rows.append({
                "segment":    segment,
                "days_prior": dtp,
                **data,
            })

    df = pd.DataFrame(rows).set_index(["segment", "days_prior"])
    return df


# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Generating booking curves...")
    df_curves = generate_booking_curves()

    # ── 1. Daily arrival fractions ────────────────────────────────────────────
    print("\n--- Daily Arrival Fractions (first 10 days shown) ---")
    print(df_curves.head(10).round(4))
    print("\nColumn sums (must all be 1.0):")
    print(df_curves.sum().round(6))

    # ── 2. Cumulative booking curves ──────────────────────────────────────────
    df_cumulative = cumulative_booking_curve(df_curves)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Cumulative booking curves
    ax = axes[0]
    colors = {"first": "#185FA5", "business": "#0F6E56",
              "premium_eco": "#BA7517", "economy": "#E24B4A"}
    for col in df_curves.columns:
        ax.plot(
            df_cumulative.index,
            df_cumulative[col] * 100,
            marker="o", markersize=3, linewidth=2,
            label=col.replace("_", " ").title(),
            color=colors.get(col, "gray"),
        )
    ax.set_title("Cumulative Booking Curves by Cabin", fontsize=14)
    ax.set_xlabel("Days Prior to Departure")
    ax.set_ylabel("Cumulative % of Segment Booked")
    ax.set_xlim(BOOKING_WINDOW_DAYS, 1)   # reverse x-axis
    ax.axvline(x=30, color="gray", linestyle="--", alpha=0.5, label="30-day mark")
    ax.axvline(x=14, color="gray", linestyle=":",  alpha=0.5, label="14-day mark")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Daily arrival distribution (Beta PDF shape)
    ax2 = axes[1]
    for col in df_curves.columns:
        ax2.bar(
            df_curves.index,
            df_curves[col] * 100,
            alpha=0.6, label=col.replace("_", " ").title(),
            color=colors.get(col, "gray"), width=1,
        )
    ax2.set_title("Daily Booking Arrivals by Cabin (%)", fontsize=14)
    ax2.set_xlabel("Days Prior to Departure")
    ax2.set_ylabel("% of Total Segment Arriving on Day")
    ax2.set_xlim(BOOKING_WINDOW_DAYS + 1, 0)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("output/booking_curves.png", dpi=150, bbox_inches="tight")
    print("\nSaved: output/booking_curves.png")
    plt.show()

    # ── 3. Reading date simulation ────────────────────────────────────────────
    # Simulate a NRT->LAX flight with 396 seats (B777 config)
    example_seats = {"first": 6, "business": 40, "premium_eco": 20, "economy": 250}
    print("\n--- Reading Date Snapshots (NRT->LAX example, B777) ---")
    series = reading_date_series(
        df_curves=df_curves,
        total_seats_by_segment=example_seats,
        reading_dates_dtp=[90, 60, 45, 30, 21, 14, 7, 3, 1],
    )
    print(series.to_string())
