"""Goodness-of-fit metrics for streamflow modelling.

Includes Nash-Sutcliffe Efficiency, Kling-Gupta Efficiency,
log-transformed NSE (sensitive to low flows), and percent bias.
"""
from __future__ import annotations

import numpy as np


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    return float(1.0 - np.nansum((sim - obs) ** 2)
                       / np.nansum((obs - np.nanmean(obs)) ** 2))


def log_nse(sim: np.ndarray, obs: np.ndarray, epsilon: float = 1e-2) -> float:
    """NSE on log-transformed flows; emphasises low-flow accuracy."""
    sim = np.log(np.asarray(sim, dtype=float) + epsilon)
    obs = np.log(np.asarray(obs, dtype=float) + epsilon)
    return float(1.0 - np.nansum((sim - obs) ** 2)
                       / np.nansum((obs - np.nanmean(obs)) ** 2))


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency (Gupta et al. 2009).

    Decomposes NSE into correlation, bias, and variability components,
    and recombines them so that all three are weighted equally.
    """
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[mask]; obs = obs[mask]
    if sim.size < 2:
        return float("nan")
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs) if np.std(obs) > 0 else 0
    beta  = np.mean(sim) / np.mean(obs) if np.mean(obs) > 0 else 0
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pbias(sim: np.ndarray, obs: np.ndarray) -> float:
    """Percent bias (sim - obs) / obs * 100."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    return float(100.0 * np.nansum(sim - obs) / np.nansum(obs))


def all_metrics(sim: np.ndarray, obs: np.ndarray) -> dict:
    return {
        "NSE":     nse(sim, obs),
        "logNSE":  log_nse(sim, obs),
        "KGE":     kge(sim, obs),
        "PBIAS":   pbias(sim, obs),
    }
