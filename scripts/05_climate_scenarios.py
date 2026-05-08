"""Climate-change scenario projections using the GLUE behavioural ensemble.

Each scenario perturbs the forcing time series (P, T) and runs the same
behavioural parameter set used in script 04. The annual mean streamflow
ensemble is summarised as a probability density per scenario.

Scenarios (delta-change applied to the historical 1995-2024 record):
    Baseline:          observed P, T
    Wetter +20%:       P x 1.20, T unchanged
    Drier  -20%:       P x 0.80, T unchanged
    Warmer +2 C:       P unchanged, T + 2 C  (PET recomputed)
    Drier + Warmer:    P x 0.80, T + 2 C    (a typical CMIP6 SSP3-7.0
                                              California outcome)

The point of the analysis is not the absolute numbers but the SPREAD
across scenarios relative to the parameter-uncertainty spread within
each scenario - which is more important for water managers.
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
from src.hydro_model import simulate, Parameters
from src.pet import hargreaves_pet

WARMUP = 365


SCENARIOS = {
    "Baseline":              {"p_factor": 1.00, "dT": 0.0},
    "Wetter (+20% P)":       {"p_factor": 1.20, "dT": 0.0},
    "Drier  (-20% P)":       {"p_factor": 0.80, "dT": 0.0},
    "Warmer (+2 C)":         {"p_factor": 1.00, "dT": 2.0},
    "Drier + Warmer":        {"p_factor": 0.80, "dT": 2.0},
}
COLOURS = ["#3B5BA5", "#1B9E77", "#D95F02", "#7570B3", "#E7298A"]


def perturb_forcings(df: pd.DataFrame, p_factor: float, dT: float,
                      latitude: float) -> tuple[np.ndarray, np.ndarray]:
    P    = df["P_mm"].to_numpy() * p_factor
    Tmax = df["T_max_C"].to_numpy() + dT
    Tmin = df["T_min_C"].to_numpy() + dT
    doy  = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy()
    PET  = hargreaves_pet(Tmax, Tmin, doy, latitude)
    return P, PET


def main():
    # Load behavioural ensemble
    arr = np.load(ROOT / "data" / "processed" / "rsa_realisations.npz")
    samples = arr["samples"]; kge_all = arr["kge"]
    behave = kge_all >= 0.5
    if behave.sum() < 30:
        cutoff = np.quantile(kge_all, 0.95)
        behave = kge_all >= cutoff
    behave_samples = samples[behave]
    weights = (kge_all[behave] - kge_all[behave].min())
    weights = weights / weights.sum()
    print(f"Behavioural ensemble: {behave.sum()} parameter sets")

    df = load_cached(RUSSIAN_RIVER_HOPLAND)
    print(f"Forcing record: {df['date'].min().date()} -> {df['date'].max().date()}")

    summary_rows = []
    annual_means = {}    # scenario -> array of N_behave annual means

    for s_name, cfg in SCENARIOS.items():
        P, PET = perturb_forcings(df, cfg["p_factor"], cfg["dT"],
                                    RUSSIAN_RIVER_HOPLAND.latitude)
        p_obj = Parameters(*behave_samples.T)
        Q_ens = simulate(P, PET, p_obj)
        # Mean annual flow (mm/yr) per ensemble member, ignoring warmup
        Q_post = Q_ens[:, WARMUP:]
        mean_per_member = Q_post.mean(axis=1) * 365.25
        annual_means[s_name] = mean_per_member

        med = float(np.average(mean_per_member, weights=weights))
        # Weighted 5/95 percentiles
        order = np.argsort(mean_per_member)
        cw    = np.cumsum(weights[order]) / weights.sum()
        q05   = float(mean_per_member[order][np.searchsorted(cw, 0.05)])
        q95   = float(mean_per_member[order][np.searchsorted(cw, 0.95)])
        summary_rows.append({"scenario": s_name, "p_factor": cfg["p_factor"],
                              "dT_C": cfg["dT"], "Q_mean_mm_yr": med,
                              "Q_5pct": q05, "Q_95pct": q95})

    summary = pd.DataFrame(summary_rows)
    print("\nMean annual streamflow per scenario (weighted ensemble):")
    print(summary.round(1).to_string(index=False))
    summary.to_csv(ROOT / "data" / "processed" / "climate_scenarios.csv", index=False)

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.2))
    positions = np.arange(len(SCENARIOS))

    for i, (s_name, vals) in enumerate(annual_means.items()):
        # Weighted KDE-like jitter via histogram
        ax.scatter(np.full_like(vals, positions[i]) + np.random.normal(0, 0.04, vals.size),
                    vals, s=10, alpha=0.30, color=COLOURS[i])
        med = np.average(vals, weights=weights)
        ax.scatter(positions[i], med, s=120, marker="D", color=COLOURS[i],
                    edgecolor="black", lw=0.7, zorder=5)
        # 5-95 band (weighted)
        order = np.argsort(vals)
        cw    = np.cumsum(weights[order]) / weights.sum()
        q05   = vals[order][np.searchsorted(cw, 0.05)]
        q95   = vals[order][np.searchsorted(cw, 0.95)]
        ax.plot([positions[i], positions[i]], [q05, q95],
                color=COLOURS[i], lw=2.0, alpha=0.85)

    # Reference: observed mean annual streamflow over the record
    obs_annual = (df["Q_mm"].to_numpy()[WARMUP:].mean() * 365.25)
    ax.axhline(obs_annual, ls="--", color="grey",
                label=f"Observed mean (1996-2024): {obs_annual:.0f} mm/yr")

    ax.set_xticks(positions)
    ax.set_xticklabels(list(SCENARIOS.keys()), rotation=15, ha="right")
    ax.set_ylabel("Mean annual streamflow (mm / yr)")
    ax.set_title(f"Climate-change scenarios for {RUSSIAN_RIVER_HOPLAND.name}\n"
                  f"Weighted GLUE ensemble (N = {int(behave.sum())} behavioural runs); "
                  f"diamonds = weighted median, bars = 5-95% range",
                  fontsize=12)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig_path = ROOT / "figures" / "05_climate_scenarios.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"Figure -> {fig_path.relative_to(ROOT)}")

    out = {"observed_mean_annual_mm_yr": obs_annual,
           "scenarios": summary.to_dict(orient="records")}
    (ROOT / "data" / "processed" / "scenarios_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
