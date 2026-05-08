"""GLUE - Generalised Likelihood Uncertainty Estimation (Beven & Binley, 1992).

Steps
-----
1. Load the LHS realisations + KGE scores from script 03.
2. Re-define behavioural threshold (KGE >= 0.5) and compute weights as
   the rescaled KGE-above-threshold (informal likelihood).
3. Re-run the behavioural set on the validation period to get an
   ensemble of streamflow time series.
4. Compute weighted 5/50/95 percentile bounds at every timestep.
5. Diagnostic: fraction of observed days that fall inside the 5-95% band
   (target ~ 0.90 for a well-specified ensemble).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_acquisition import load_cached, RUSSIAN_RIVER_HOPLAND
from src.hydro_model import simulate, Parameters, PARAMETER_NAMES

WARMUP = 365
GLUE_THRESHOLD = 0.5      # KGE
PLOT_PERIOD = ("2018-10-01", "2024-09-30")    # last 6 water years for clarity


def weighted_quantile(values: np.ndarray, weights: np.ndarray,
                       q: float) -> float:
    order = np.argsort(values)
    cw = np.cumsum(weights[order]) / weights.sum()
    idx = int(np.searchsorted(cw, q))
    idx = min(idx, len(values) - 1)
    return float(values[order][idx])


def main():
    # ---- load LHS results from script 03 --------------------------------
    arr = np.load(ROOT / "data" / "processed" / "rsa_realisations.npz")
    samples, kge_all = arr["samples"], arr["kge"]
    print(f"Loaded {len(samples):,} LHS realisations.")

    behave = kge_all >= GLUE_THRESHOLD
    print(f"GLUE threshold: KGE >= {GLUE_THRESHOLD}  -> "
          f"{int(behave.sum())} behavioural runs ({100*behave.mean():.1f}%)")
    if behave.sum() < 30:
        # Fallback: take the top 5% so we always have an ensemble.
        cutoff = np.quantile(kge_all, 0.95)
        behave = kge_all >= cutoff
        print(f"  (relaxed to KGE >= {cutoff:.2f}; {int(behave.sum())} runs)")

    # ---- informal likelihood weights ------------------------------------
    kge_b = kge_all[behave]
    likelihood = kge_b - kge_b.min()
    weights = likelihood / likelihood.sum()

    # ---- forcing data ---------------------------------------------------
    df = load_cached(RUSSIAN_RIVER_HOPLAND)
    df = df[(df["date"] >= PLOT_PERIOD[0]) & (df["date"] <= PLOT_PERIOD[1])]\
            .reset_index(drop=True)
    P, PET, Qobs = (df["P_mm"].to_numpy(),
                     df["PET_mm"].to_numpy(),
                     df["Q_mm"].to_numpy())

    # ---- run behavioural ensemble (vectorised) --------------------------
    p_behave = Parameters(*samples[behave].T)
    print(f"Simulating ensemble over {len(df):,} days ...")
    Q_ens = simulate(P, PET, p_behave)         # shape (N_b, T)

    # ---- weighted prediction band ---------------------------------------
    T = Q_ens.shape[1]
    q05 = np.array([weighted_quantile(Q_ens[:, t], weights, 0.05) for t in range(T)])
    q50 = np.array([weighted_quantile(Q_ens[:, t], weights, 0.50) for t in range(T)])
    q95 = np.array([weighted_quantile(Q_ens[:, t], weights, 0.95) for t in range(T)])

    # ---- coverage diagnostic --------------------------------------------
    inside = (Qobs >= q05) & (Qobs <= q95)
    coverage_frac = float(np.mean(inside[WARMUP:]))
    print(f"Observed-inside-band fraction (target ~0.90): {coverage_frac:.3f}")

    # Save ensemble bounds for the scenario step
    np.savez(ROOT / "data" / "processed" / "glue_predictions.npz",
              dates=df["date"].astype(str).to_numpy(),
              q05=q05, q50=q50, q95=q95, obs=Qobs)
    out = {"glue_threshold": GLUE_THRESHOLD,
           "n_behavioural":  int(behave.sum()),
           "coverage_5_95":  coverage_frac,
           "plot_period":    PLOT_PERIOD}
    out_path = ROOT / "data" / "processed" / "glue_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path.relative_to(ROOT)}")

    # ---- figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5))
    dates = pd.to_datetime(df["date"])
    ax.fill_between(dates, q05, q95, color="#1F77B4", alpha=0.30,
                     label="5 - 95 % prediction band")
    ax.plot(dates, q50, color="#0B3D91", lw=1.0, label="Weighted median")
    ax.plot(dates, Qobs, color="black", lw=0.7, alpha=0.8,
             label="Observed (USGS 11462500)")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 100)
    ax.set_ylabel("Streamflow (mm/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"GLUE prediction band  -  Russian River near Hopland, CA  "
                  f"(coverage of 5 - 95 % band: {coverage_frac*100:.1f} %)",
                  fontsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = ROOT / "figures" / "04_glue_uncertainty_band.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"Figure -> {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
