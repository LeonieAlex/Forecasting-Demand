"""
experiment.py
-------------
Batch forecasting experiment across all O-D pairs.

Compares four demand inputs — each representing a different level of
unconstraining quality — as the training signal for the same forecaster:

  constrained : raw observed seats (baseline, censored at capacity)
  naive       : naive unconstrained = constrained (no change)
  em          : EM rolling-window unconstrained
  pd          : PD rolling-window unconstrained
  oracle      : latent_seats (true simulated demand, upper bound)

All inputs are forecast with the same model (SARIMA by default) so
forecast accuracy differences reflect only the demand input quality.
Forecasts are evaluated against the oracle (latent_seats) as ground
truth, since that is what every unconstraining method is trying to
recover.

Train: 2015-01-01 → 2022-12-31
Test:  2023-01-01 → 2024-12-31

Outputs:
  forecasting/results/experiment_summary.csv
  forecasting/results/experiment_comparison.png

Usage (from Simulation root):
    python forecasting/experiment.py
    python forecasting/experiment.py --od "JP→US"   # one pair only
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from forecasting.models  import SARIMAForecaster
from forecasting.utils   import add_features, summary

# ── Config ────────────────────────────────────────────────────────────────────

BATCH_CSV   = os.path.join(ROOT, "unconstraining", "results", "batch_unconstrained.csv")
RESULTS_DIR = os.path.join(HERE, "results")
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"

# Demand signals to compare (column name → display label)
# Naive is omitted — identical to Constrained, adds no information.
SIGNALS = {
    "constrained_seats": "Constrained",
    "em_seats":          "EM",
    "pd_seats":          "PD",
    "latent_seats":      "Oracle",
}

COLORS = {
    "Constrained": "#777777",
    "EM":          "#3B6FA0",
    "PD":          "#2E8B57",
    "Oracle":      "#C05050",
}

SARIMA_ORDER          = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 7)   # simpler order for batch speed


# ── Per-O-D forecasting ───────────────────────────────────────────────────────

def forecast_od(df_od: pd.DataFrame, od: str) -> tuple[list[dict], pd.DataFrame]:
    """
    For one O-D pair, fit SARIMA on each demand signal (train period) and
    forecast the test period. Evaluate all forecasts against oracle.
    Returns (metric_rows, predictions_df).
    """
    df_od  = df_od.sort_values("date").reset_index(drop=True)
    train  = df_od[df_od["date"] <= TRAIN_END]
    test   = df_od[df_od["date"] >= TEST_START]

    oracle_test = test["latent_seats"].values.astype(float)
    n_test      = len(test)

    rows  = []
    preds = {"date": test["date"].values, "oracle": oracle_test}

    for col, label in SIGNALS.items():
        if col not in df_od.columns:
            continue

        y_train = train[col].values.astype(float)

        model = SARIMAForecaster(
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            maxiter=200,
        )
        try:
            model.fit(y_train)
            y_pred = np.clip(model.predict(n_test), 0, None)
        except Exception as e:
            print(f"    [{od}] {label} SARIMA failed: {e} — using last-value fallback")
            y_pred = np.full(n_test, y_train[-1])

        preds[label] = y_pred
        m = summary(oracle_test, y_pred, name=label)
        m["od_pair"] = od
        rows.append(m)

        print(f"    {label:<14}  MAE={m['MAE']:>7.1f}  "
              f"RMSE={m['RMSE']:>7.1f}  WAPE={m['WAPE']:>6.2f}%")

    return rows, pd.DataFrame(preds)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_od(pred_df: pd.DataFrame, od: str, metrics: pd.DataFrame, out_dir: str):
    """Time-series plot for one O-D pair: SARIMA forecasts vs true latent demand."""
    fig, ax = plt.subplots(figsize=(14, 5))

    dates = pred_df["date"]

    # True latent demand (ground truth) — prominent filled band
    ax.fill_between(dates, 0, pred_df["oracle"], alpha=0.12, color="black")
    ax.plot(dates, pred_df["oracle"], color="black", lw=2.0, alpha=0.7,
            label="Latent demand (oracle)", zorder=6)

    # Forecast lines for each demand signal
    for label, color in COLORS.items():
        if label not in pred_df.columns or label == "Oracle":
            continue
        m   = metrics[metrics["model"] == label].iloc[0] if label in metrics["model"].values else None
        lbl = f"{label}  (WAPE={m['WAPE']:.1f}%)" if m is not None else label
        lw  = 1.6 if label in ("EM", "PD") else 1.0
        ax.plot(dates, pred_df[label], color=color, lw=lw, alpha=0.85, label=lbl)

    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=20, ha="right", fontsize=8)
    ax.set_title(
        f"{od}  —  Passenger Demand Forecast vs True Latent Demand  (Test: 2023–2024)",
        fontsize=11,
    )
    ax.set_ylabel("Passengers per day")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, ncol=2, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.18)
    plt.tight_layout()

    slug = od.replace("→", "_to_").replace(" ", "")
    out  = os.path.join(out_dir, f"experiment_{slug}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary(summary_df: pd.DataFrame, out_dir: str):
    """Heatmap of WAPE by O-D pair × demand input."""
    pivot = summary_df.pivot(index="od_pair", columns="model", values="WAPE")
    col_order = [l for l in SIGNALS.values() if l in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(len(col_order) * 1.6 + 2, len(pivot) * 0.55 + 1.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="WAPE (%)")

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for r in range(len(pivot)):
        for c in range(len(col_order)):
            val = pivot.values[r, c]
            ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color="black")

    ax.set_title("Forecast WAPE (%) by O-D Pair × Demand Input\n"
                 "(lower = better; evaluated vs Oracle/true demand)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "experiment_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary heatmap → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--od", default=None, help="Run a single O-D pair only")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(BATCH_CSV):
        print(f"ERROR: {BATCH_CSV} not found.")
        print("Run  python unconstraining/batch_unconstrain.py  first.")
        sys.exit(1)

    print(f"Loading {BATCH_CSV}...")
    df = pd.read_csv(BATCH_CSV, parse_dates=["date"])

    od_pairs = [args.od] if args.od else sorted(df["od_pair"].unique())
    print(f"\nRunning experiment on {len(od_pairs)} O-D pair(s)...\n")

    all_metrics = []
    for od in od_pairs:
        df_od = df[df["od_pair"] == od]
        if len(df_od) < 400:
            print(f"  Skipping {od} (too few rows: {len(df_od)})")
            continue

        print(f"{'─'*55}")
        print(f"  {od}  ({len(df_od):,} days)")
        rows, pred_df = forecast_od(df_od.copy(), od)
        all_metrics.extend(rows)

        od_metrics = pd.DataFrame([r for r in rows], columns=["model","MAE","RMSE","MAPE","WAPE","od_pair"])
        plot_od(pred_df, od, od_metrics, RESULTS_DIR)

    summary_df = pd.DataFrame(all_metrics)
    out_csv = os.path.join(RESULTS_DIR, "experiment_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\n{'='*55}")
    print("EXPERIMENT SUMMARY — Average WAPE by demand input")
    print("="*55)
    avg = summary_df.groupby("model")["WAPE"].mean().reindex(list(SIGNALS.values())).dropna()
    for label, wape_val in avg.items():
        print(f"  {label:<14}  WAPE={wape_val:.2f}%")

    best = avg.idxmin()
    print(f"\n  Best input (avg WAPE): {best}")
    print(f"  Full results → {out_csv}")

    if len(od_pairs) > 1:
        plot_summary(summary_df, RESULTS_DIR)


if __name__ == "__main__":
    main()
