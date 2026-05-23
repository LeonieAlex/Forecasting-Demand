"""
visualize_demand_dist.py
------------------------
For each O-D pair, fits a Beta distribution to the load-factor distribution
of each demand signal (constrained, EM, PD, oracle/latent) and plots the
fitted curves alongside the empirical histogram.

Load factor = seats / total_capacity  →  bounded in [0, 1], making Beta
the natural distributional assumption (unlike Normal which is unbounded).

Input:  unconstraining/results/batch_unconstrained.csv
Output: unconstraining/results/demand_dist_<OD>.png  (one per O-D pair)
        unconstraining/results/demand_dist_all.png   (grid summary)

Usage (from Simulation root):
    python unconstraining/visualize_demand_dist.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

ROOT    = os.path.join(os.path.dirname(__file__), "..")
IN_CSV  = os.path.join(ROOT, "unconstraining", "results", "batch_unconstrained.csv")
OUT_DIR = os.path.join(ROOT, "unconstraining", "results")

# Demand signals: (column, display label, colour)
SIGNALS = [
    ("constrained_seats", "Constrained", "#555555"),
    ("em_seats",          "EM",          "#3B6FA0"),
    ("pd_seats",          "PD",          "#2E8B57"),
    ("latent_seats",      "Oracle",      "#C05050"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def to_lf(series: pd.Series, capacity: float) -> np.ndarray:
    """Convert seat counts to load factor, clipped to (0.01, 0.99) for Beta fitting."""
    lf = series.values.astype(float) / capacity
    return np.clip(lf, 0.01, 0.99)


def fit_beta(lf: np.ndarray):
    """Fit Beta(alpha, beta) via MLE. Returns (a, b) or None on failure."""
    try:
        a, b, _, _ = stats.beta.fit(lf, floc=0, fscale=1)
        return a, b
    except Exception:
        return None


def plot_od(ax, df_od: pd.DataFrame, od: str, capacity: float):
    """
    One axes: histogram + fitted Beta PDF for each signal.
    """
    x = np.linspace(0.01, 0.99, 300)

    for col, label, color in SIGNALS:
        if col not in df_od.columns:
            continue

        lf     = to_lf(df_od[col], capacity)
        params = fit_beta(lf)

        # Histogram (normalised to density)
        ax.hist(lf, bins=40, density=True, alpha=0.18, color=color)

        # Fitted Beta PDF
        if params:
            a, b = params
            pdf  = stats.beta.pdf(x, a, b)
            ax.plot(x, pdf, color=color, lw=2.0,
                    label=f"{label}  β({a:.2f},{b:.2f})")
        else:
            ax.plot([], [], color=color, lw=2.0, label=f"{label}  (fit failed)")

    # Capacity line at LF = 1.0
    ax.axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.5, label="Capacity")

    ax.set_title(od, fontsize=10, fontweight="bold")
    ax.set_xlabel("Load Factor", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} not found.")
        print("Run  python unconstraining/batch_unconstrain.py  first.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {IN_CSV}...")
    df = pd.read_csv(IN_CSV, parse_dates=["date"])
    od_pairs = sorted(df["od_pair"].unique())
    print(f"O-D pairs: {len(od_pairs)}")

    # ── Per-O-D individual plots ──────────────────────────────────────────────
    for od in od_pairs:
        df_od    = df[df["od_pair"] == od]
        capacity = float(df_od["total_capacity"].mode()[0])

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_od(ax, df_od, od, capacity)

        slug = od.replace("→", "_to_").replace(" ", "")
        out  = os.path.join(OUT_DIR, f"demand_dist_{slug}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")

    # ── Summary grid (all O-D pairs on one figure) ────────────────────────────
    n     = len(od_pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 6, nrows * 3.8))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, od in enumerate(od_pairs):
        df_od    = df[df["od_pair"] == od]
        capacity = float(df_od["total_capacity"].mode()[0])
        plot_od(axes_flat[i], df_od, od, capacity)

    # Hide any unused panels
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Demand Distribution per O-D Pair — Fitted Beta Curves\n"
        "(Constrained vs EM vs PD vs Oracle/Latent)",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    out_grid = os.path.join(OUT_DIR, "demand_dist_all.png")
    plt.savefig(out_grid, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n  Grid summary → {out_grid}")

    # ── Print fitted Beta params table ────────────────────────────────────────
    print("\n── Fitted Beta(α, β) parameters ─────────────────────────────────")
    header = f"{'O-D':<12}" + "".join(f"  {s[1]:<22}" for s in SIGNALS)
    print(header)
    print("─" * len(header))

    for od in od_pairs:
        df_od    = df[df["od_pair"] == od]
        capacity = float(df_od["total_capacity"].mode()[0])
        row      = f"{od:<12}"
        for col, label, _ in SIGNALS:
            if col not in df_od.columns:
                row += f"  {'N/A':<22}"
                continue
            lf     = to_lf(df_od[col], capacity)
            params = fit_beta(lf)
            if params:
                a, b = params
                mean = a / (a + b)
                row += f"  α={a:.2f} β={b:.2f} μ={mean:.2f}  "
            else:
                row += f"  {'fit failed':<22}"
        print(row)


if __name__ == "__main__":
    main()
