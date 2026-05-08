"""Differential-Evolution calibration with split-sample validation.

Calibration period:  1995-10-01 -> 2009-09-30  (14 water years)
Validation period:   2010-10-01 -> 2024-09-30  (14 water years)

Optimises Kling-Gupta Efficiency (KGE), which weights bias, variability,
and correlation equally — a robust default for streamflow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_acquisition import load_cached, RUSSIAN_RIVER_HOPLAND
from src.hydro_model import simulate, Parameters, PARAMETER_BOUNDS, PARAMETER_NAMES
from src.metrics import all_metrics, kge

CAL_START, CAL_END = "1995-10-01", "2009-09-30"
VAL_START, VAL_END = "2010-10-01", "2024-09-30"
WARMUP_DAYS = 365


def slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def neg_kge_factory(P, PET, Q_obs, warmup):
    def neg_kge(x):
        params = Parameters(*x)
        Q_sim = simulate(P, PET, params)
        return -kge(Q_sim[warmup:], Q_obs[warmup:])
    return neg_kge


def main():
    df = load_cached(RUSSIAN_RIVER_HOPLAND)
    print(f"Loaded {len(df):,} days from {df['date'].min().date()} -> "
          f"{df['date'].max().date()}")

    cal = slice_period(df, CAL_START, CAL_END)
    val = slice_period(df, VAL_START, VAL_END)
    print(f"Calibration: {len(cal):,} days  ({CAL_START} -> {CAL_END})")
    print(f"Validation:  {len(val):,} days  ({VAL_START} -> {VAL_END})")

    bounds = [PARAMETER_BOUNDS[n] for n in PARAMETER_NAMES]

    print("\nRunning Differential Evolution (target: maximise KGE) ...")
    obj = neg_kge_factory(cal["P_mm"].to_numpy(),
                          cal["PET_mm"].to_numpy(),
                          cal["Q_mm"].to_numpy(),
                          WARMUP_DAYS)
    result = differential_evolution(obj, bounds, seed=42, tol=1e-6,
                                     maxiter=200, polish=True, workers=1)

    p_opt = Parameters(*result.x)
    print(f"\nOptimal parameters (KGE = {-result.fun:.3f}):")
    for n, v in zip(PARAMETER_NAMES, result.x):
        print(f"  {n:8s} = {v:8.3f}")

    # Calibration metrics on full cal record (excluding warmup)
    Q_cal_sim = simulate(cal["P_mm"].to_numpy(), cal["PET_mm"].to_numpy(), p_opt)
    cal_metrics = all_metrics(Q_cal_sim[WARMUP_DAYS:], cal["Q_mm"].to_numpy()[WARMUP_DAYS:])

    # Validation metrics — independent period
    Q_val_sim = simulate(val["P_mm"].to_numpy(), val["PET_mm"].to_numpy(), p_opt)
    val_metrics = all_metrics(Q_val_sim[WARMUP_DAYS:], val["Q_mm"].to_numpy()[WARMUP_DAYS:])

    print("\nGoodness of fit:")
    print(f"  {'metric':8s}  {'calibration':>12s}  {'validation':>12s}")
    for k in ["NSE", "KGE", "logNSE", "PBIAS"]:
        print(f"  {k:8s}  {cal_metrics[k]:12.3f}  {val_metrics[k]:12.3f}")

    # Save calibrated parameters + metrics for downstream scripts
    out = {
        "parameters":   {n: float(v) for n, v in zip(PARAMETER_NAMES, result.x)},
        "calibration":  {"start": CAL_START, "end": CAL_END, "metrics": cal_metrics},
        "validation":   {"start": VAL_START, "end": VAL_END, "metrics": val_metrics},
        "warmup_days":  WARMUP_DAYS,
        "objective":    "KGE",
    }
    out_path = ROOT / "data" / "processed" / "calibration_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path.relative_to(ROOT)}")

    # ---- figure: cal/val time series with shaded periods ------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

    for ax, period_df, period_sim, label in [
        (axes[0], cal, Q_cal_sim, "Calibration: 1995-2009"),
        (axes[1], val, Q_val_sim, "Validation: 2010-2024"),
    ]:
        dates = pd.to_datetime(period_df["date"])
        ax.plot(dates, period_df["Q_mm"], color="black", lw=0.5, alpha=0.85,
                 label="Observed (USGS 11462500)")
        ax.plot(dates, period_sim, color="#D62728", lw=0.7, alpha=0.85,
                 label="Simulated (5-parameter two-bucket)")
        ax.axvspan(dates.iloc[0], dates.iloc[WARMUP_DAYS], color="grey", alpha=0.15,
                    label="Warm-up (excluded from metrics)")
        ax.set_ylabel("Streamflow (mm/day)")
        ax.set_title(f"{label}   |   "
                      f"NSE = {(cal_metrics if 'Calibration' in label else val_metrics)['NSE']:.3f}, "
                      f"KGE = {(cal_metrics if 'Calibration' in label else val_metrics)['KGE']:.3f}, "
                      f"PBIAS = {(cal_metrics if 'Calibration' in label else val_metrics)['PBIAS']:+.1f}%",
                      fontsize=11)
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 100)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Date")
    plt.suptitle(f"{RUSSIAN_RIVER_HOPLAND.name}  -  "
                  f"split-sample calibration / validation",
                  fontsize=13, y=0.995)
    plt.tight_layout()
    fig_path = ROOT / "figures" / "02_calibration_validation.png"
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"Figure -> {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
