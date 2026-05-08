"""Differential Evolution against KGE, with split-sample validation."""
import numpy as np
from scipy.optimize import differential_evolution

WARMUP = 365  # days excluded from the metric

def kge(sim, obs):
    """Kling-Gupta Efficiency (Gupta et al. 2009)."""
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

# Bounds for the 5 parameters
BOUNDS = [(50, 600),     # S_max  (mm)
          (0.0, 1.0),    # f_quick (-)
          (0.001, 0.5),  # k_perc (1/day)
          (5, 200),      # tau_b  (days)
          (0.5, 3.0)]    # gamma  (-)

def neg_kge(x, P, PET, Qobs):
    Qsim = simulate(P, PET, *x)            # from your model module
    return -kge(Qsim[WARMUP:], Qobs[WARMUP:])

# Calibration period (water years 1996-2009)
result = differential_evolution(neg_kge, BOUNDS, args=(P_cal, PET_cal, Q_cal),
                                 seed=42, tol=1e-6, maxiter=200)
print(f"Calibration KGE: {-result.fun:.3f}")
print(f"Optimal params:  {dict(zip(['S_max','f_quick','k_perc','tau_b','gamma'], result.x))}")

# Validation on a held-out period (water years 2011-2024)
Qsim_val = simulate(P_val, PET_val, *result.x)
print(f"Validation KGE:  {kge(Qsim_val[WARMUP:], Q_val[WARMUP:]):.3f}")
