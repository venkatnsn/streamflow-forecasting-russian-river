"""Regional Sensitivity Analysis (Spear & Hornberger, 1980).

Steps
-----
1. Latin Hypercube sample of N parameter sets within the prior bounds.
2. Run the model for each, score every run with KGE.
3. Behavioural threshold: keep the top 10% (highest KGE).
4. Plot the cumulative distribution of each parameter inside the
   behavioural set vs. the prior. Parameters whose CDFs are bent away
   from the diagonal are sensitive; those that follow it are not.
5. Compute the KS distance between behavioural and prior for each
   parameter — the standard scalar sensitivity index.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_acquisition import load_cached, RUSSIAN_RIVER_HOPLAND
from src.hydro_model import simulate, Parameters, PARAMETER_BOUNDS, PARAMETER_NAMES
from src.metrics import kge

N_SAMPLES = 5000
WARMUP = 365
TOP_FRACTION = 0.10
PERIOD = ("1995-10-01", "2024-09-30")     # all 30 water years


def main():
    df = load_cached(RUSSIAN_RIVER_HOPLAND)
    df = df[(df["date"] >= PERIOD[0]) & (df["date"] <= PERIOD[1])].reset_index(drop=True)
    P, PET, Qobs = (df["P_mm"].to_numpy(),
                     df["PET_mm"].to_numpy(),
                     df["Q_mm"].to_numpy())

    # ---- LHS sample over the prior --------------------------------------
    sampler = qmc.LatinHypercube(d=len(PARAMETER_NAMES), seed=2026)
    u = sampler.random(N_SAMPLES)
    bounds = np.array([PARAMETER_BOUNDS[n] for n in PARAMETER_NAMES])
    samples = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    print(f"Sampled {N_SAMPLES} parameter sets via Latin Hypercube.")

    # ---- run model vectorised, score with KGE --------------------------
    p = Parameters(*samples.T)        # each attribute is a length-N array
    print(f"Running {N_SAMPLES} simulations (vectorised) ...")
    Q_all = simulate(P, PET, p)
    kge_scores = np.array([kge(Q_all[i, WARMUP:], Qobs[WARMUP:])
                            for i in range(N_SAMPLES)])

    # ---- behavioural set -------------------------------------------------
    cutoff = np.quantile(kge_scores, 1 - TOP_FRACTION)
    behave = kge_scores >= cutoff
    n_behave = int(behave.sum())
    print(f"Behavioural threshold: KGE >= {cutoff:.3f}  "
          f"({n_behave} of {N_SAMPLES} runs, {100*TOP_FRACTION:.0f}%)")

    # ---- KS distance per parameter --------------------------------------
    ks_distances = {}
    for j, name in enumerate(PARAMETER_NAMES):
        lo, hi = PARAMETER_BOUNDS[name]
        std_all  = (samples[:, j] - lo) / (hi - lo)
        std_b    = (samples[behave, j] - lo) / (hi - lo)
        # Compare both to a common grid
        grid = np.linspace(0, 1, 200)
        cdf_all = np.searchsorted(np.sort(std_all), grid) / len(std_all)
        cdf_b   = np.searchsorted(np.sort(std_b),   grid) / max(len(std_b), 1)
        ks_distances[name] = float(np.max(np.abs(cdf_b - cdf_all)))

    print("\nKS distance (behavioural vs prior, larger = more sensitive):")
    for n, d in sorted(ks_distances.items(), key=lambda kv: -kv[1]):
        print(f"  {n:8s}  {d:.3f}")

    # ---- save ------------------------------------------------------------
    out = {"n_samples": N_SAMPLES, "n_behavioural": n_behave,
           "kge_threshold": float(cutoff),
           "ks_distance":   ks_distances,
           "ranking":       [k for k, _ in sorted(ks_distances.items(),
                                                    key=lambda kv: -kv[1])]}
    out_path = ROOT / "data" / "processed" / "sensitivity_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path.relative_to(ROOT)}")

    # Persist behavioural set so GLUE can reuse the same realisations.
    np.savez(ROOT / "data" / "processed" / "rsa_realisations.npz",
              samples=samples, kge=kge_scores, behavioural=behave)
    print(f"Saved -> data/processed/rsa_realisations.npz")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, len(PARAMETER_NAMES),
                              figsize=(3.0 * len(PARAMETER_NAMES), 3.6),
                              sharey=True)
    for j, name in enumerate(PARAMETER_NAMES):
        lo, hi = PARAMETER_BOUNDS[name]
        std_all = np.sort((samples[:, j] - lo) / (hi - lo))
        std_b   = np.sort((samples[behave, j] - lo) / (hi - lo))
        axes[j].plot(std_all, np.linspace(0, 1, len(std_all)),
                     color="grey", lw=1.2, label="prior")
        axes[j].plot(std_b, np.linspace(0, 1, len(std_b)),
                     color="#D62728", lw=2.2, label="behavioural")
        axes[j].plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.5)
        axes[j].set_title(f"{name}\nKS = {ks_distances[name]:.2f}", fontsize=11)
        axes[j].set_xlabel("standardised value")
        axes[j].set_ylim(0, 1); axes[j].set_xlim(0, 1)
        axes[j].grid(alpha=0.3)
    axes[0].set_ylabel("Cumulative proportion")
    axes[-1].legend(loc="lower right", fontsize=9, frameon=False)
    plt.suptitle(f"Regional Sensitivity Analysis  -  "
                  f"top {int(100*TOP_FRACTION)}% of {N_SAMPLES} LHS runs (KGE >= {cutoff:.2f})",
                  fontsize=12, y=1.03)
    plt.tight_layout()
    fig_path = ROOT / "figures" / "03_sensitivity_cdf.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"Figure -> {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
