"""
booking_curve_unconstrain.py
----------------------------
Produces a "Segmented Booking Curves with Capacity Truncation" chart for
a single route, showing:

  - Dashed colored lines: cumulative demand per cabin class (demand sources)
  - Black solid line:     total true (latent) demand
  - Red solid line:       observed demand, truncated at capacity
  - Dotted line:          capacity ceiling
  - Pink shading:         unobserved / spilled demand

Truncation logic: on censored flights the cabin fills before departure,
so all subsequent booking observations are capped at capacity. EM
unconstraining estimates the latent (pre-truncation) true demand level;
the smooth Beta-CDF curves then show how that demand would have built up
over the booking window if capacity had not been binding.

Usage (from Simulation root):
    python unconstraining/booking_curve_unconstrain.py
    python unconstraining/booking_curve_unconstrain.py --route "NRT->LAX"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from models import EMUnconstrainer
from src.simulator.config import AIRCRAFT
from src.simulator.booking_curves import (
    BookingCurveParams, generate_booking_curves, cumulative_booking_curve,
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join(ROOT, "output", "simulation.csv")
RESULTS_DIR = os.path.join(ROOT, "forecasting", "results")
ROUTE       = "HND->LAX"
CABINS      = ["first", "business", "premium_eco", "economy"]
DTPS        = [90, 60, 30, 14, 7, 1]
CENSOR_LF   = 0.97
BOOKING_WINDOW = 90   # days shown on x-axis

CABIN_COLORS = {
    "first":       "#8E7CC3",   # muted purple
    "business":    "#6FA8DC",   # steel blue
    "premium_eco": "#93C47D",   # sage green
    "economy":     "#F6B26B",   # warm orange
}
CABIN_LABELS = {
    "first":       "First Class",
    "business":    "Business",
    "premium_eco": "Premium Economy",
    "economy":     "Economy",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def cabin_capacity(aircraft_type: str) -> dict[str, int]:
    ac = AIRCRAFT.get(aircraft_type, AIRCRAFT["B789"])
    return {
        "first":       ac.first_seats,
        "business":    ac.business_seats,
        "premium_eco": ac.premium_eco_seats,
        "economy":     ac.economy_seats,
    }


def em_unconstrain_cabin(seats_obs: np.ndarray, is_cens: np.ndarray, cap: int) -> float:
    """Return EM-estimated mean latent demand for one cabin."""
    try:
        est = EMUnconstrainer().fit(seats_obs.astype(float), is_cens, cap)
        return float(est[is_cens].mean()) if is_cens.any() else seats_obs.mean()
    except Exception:
        return seats_obs.mean()


# ── Main chart ────────────────────────────────────────────────────────────────

def plot_booking_curve_truncation(route_df: pd.DataFrame, route: str, cap: dict[str, int],
                                   out_dir: str):
    """
    Produce the reference-style chart: cumulative demand by cabin + truncation.
    Uses smooth Beta-CDF curves scaled to EM-unconstrained latent demand levels.
    """
    # ── 1. Compute EM-unconstrained mean seats per cabin ──────────────────────
    latent_seats = {}
    constrained_seats = {}

    for cabin in CABINS:
        cabin_cap = cap[cabin]
        lf_col    = f"lf_{cabin}"
        seat_col  = f"seats_{cabin}"

        if seat_col not in route_df.columns:
            latent_seats[cabin] = cabin_cap * 0.85
            constrained_seats[cabin] = cabin_cap * 0.85
            continue

        obs     = route_df[seat_col].values.astype(float)
        is_cens = (route_df[lf_col].values >= CENSOR_LF) if lf_col in route_df.columns \
                  else np.zeros(len(obs), dtype=bool)

        constrained_seats[cabin] = obs.mean()
        latent_seats[cabin]      = em_unconstrain_cabin(obs, is_cens, cabin_cap)

    total_capacity    = sum(cap.values())
    total_constrained = sum(constrained_seats.values())
    total_latent      = sum(latent_seats.values())

    print(f"  Cabin latent demand (EM-unconstrained avg):")
    for cabin in CABINS:
        print(f"    {CABIN_LABELS[cabin]:<18} cap={cap[cabin]:>4}  "
              f"constrained={constrained_seats[cabin]:5.1f}  "
              f"latent={latent_seats[cabin]:5.1f}")
    print(f"  Total: capacity={total_capacity}  "
          f"constrained={total_constrained:.1f}  latent={total_latent:.1f}")

    # ── 2. Generate smooth cumulative curves via Beta CDF ─────────────────────
    # Use the same 90-day booking window as the simulation (BookingCurveParams default in run.py)
    params    = BookingCurveParams(booking_window_days=BOOKING_WINDOW)
    df_curves = generate_booking_curves(params)
    cum_df    = cumulative_booking_curve(df_curves)

    # cum_df.loc[dtp] = fraction of demand that books in the LAST `dtp` days.
    # 1 - cum_df.loc[dtp] = fraction already booked BEFORE this point in the window.
    # At dtp=BOOKING_WINDOW: 1 - 1.0 = 0.0  (window just opened, nothing booked yet)
    # At dtp=1:              1 - tiny  ≈ 1.0 (almost everything booked by departure)
    # This gives curves that START at 0 and ACCUMULATE toward departure — matching reference.
    dtp_range = sorted(cum_df.index.tolist(), reverse=True)   # 90 → 1
    x = np.array([-d for d in dtp_range] + [0])              # -90 … -1, then 0 (departure)

    cabin_cum = {}
    for cabin in CABINS:
        if cabin in cum_df.columns:
            already_booked = 1.0 - cum_df.loc[dtp_range, cabin].values
        else:
            already_booked = np.linspace(0, 1, len(dtp_range))
        # Append departure point: all demand materialised at day 0
        cabin_cum[cabin] = np.append(already_booked, 1.0) * latent_seats[cabin]

    total_true = sum(cabin_cum.values())
    total_obs  = np.minimum(total_true, total_capacity)

    # ── 3. Draw chart ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Unobserved demand shading (between capacity and true demand, where truncated)
    mask = total_true > total_capacity
    ax.fill_between(
        x, total_capacity, total_true,
        where=mask, color="#E74C3C", alpha=0.10,
        label="Unobserved Demand",
    )

    # Individual cabin curves (thin dashed, colored) — shown as separate demand sources
    for cabin in CABINS:
        ax.plot(x, cabin_cum[cabin], linestyle="--", lw=1.3,
                color=CABIN_COLORS[cabin], alpha=0.75,
                label=CABIN_LABELS[cabin])

    # True demand (black, bold)
    ax.plot(x, total_true, color="black", lw=2.4, label="True Demand")

    # Observed / truncated (red, bold)
    ax.plot(x, total_obs, color="#C0392B", lw=2.0, label="Observed (Truncated)")

    # Capacity line (dotted grey)
    ax.axhline(total_capacity, color="#555555", lw=1.2, ls=":",
               label=f"Capacity ({total_capacity} seats)")

    # Axes formatting
    ax.set_xlim(x[0], 0)
    ax.set_ylim(0)
    ax.set_xlabel("Days Before Departure", fontsize=12)
    ax.set_ylabel("Cumulative Seats Booked", fontsize=12)
    ax.set_title(f"Segmented Booking Curves with Capacity Truncation\n{route}",
                 fontsize=13, fontweight="bold", pad=14)

    # X-tick labels: show every 10 days
    tick_vals = [v for v in range(-BOOKING_WINDOW, 1, 10)]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([str(v) for v in tick_vals], fontsize=9)

    ax.grid(True, alpha=0.20, linestyle="-", color="#AAAAAA")
    ax.spines[["top", "right"]].set_visible(False)

    # Legend — ordered nicely
    handles, labels = ax.get_legend_handles_labels()
    # Put True Demand + Observed first, then capacity, then individual cabins, then shading
    order_labels = (
        ["True Demand", "Observed (Truncated)", f"Capacity ({total_capacity} seats)",
         "Unobserved Demand"]
        + [CABIN_LABELS[c] for c in CABINS]
    )
    ordered = {l: h for h, l in zip(handles, labels)}
    legend_handles = [ordered[l] for l in order_labels if l in ordered]
    legend_labels  = [l for l in order_labels if l in ordered]
    ax.legend(legend_handles, legend_labels, fontsize=9, framealpha=0.9,
              loc="upper left", ncol=1)

    plt.tight_layout()
    slug = route.replace("->", "_to_").replace(" ", "")
    out  = os.path.join(out_dir, f"booking_curve_truncation_{slug}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart → {out}")
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", default=ROUTE)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    route_df = df[df["flight_od"] == args.route].sort_values("date").reset_index(drop=True)
    if route_df.empty:
        print(f"ERROR: route '{args.route}' not found.")
        sys.exit(1)

    aircraft_type = route_df["aircraft_type"].iloc[0]
    cap           = cabin_capacity(aircraft_type)

    print(f"\nRoute: {args.route}  |  Aircraft: {aircraft_type}  |  {len(route_df):,} flights")
    print(f"Cabin capacities: {cap}\n")

    plot_booking_curve_truncation(route_df, args.route, cap, RESULTS_DIR)


if __name__ == "__main__":
    main()
