"""5-parameter two-bucket rainfall-runoff model, Numba-JIT'd.

Soil bucket -> ET (atm) + saturation excess -> Quickflow + Percolation
Groundwater bucket -> Baseflow

Total streamflow Q = Quickflow + Baseflow.
Five free parameters: S_max, f_quick, k_perc, tau_b, gamma.
"""
import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def simulate(P, PET, S_max, f_quick, k_perc, tau_b, gamma,
              S0_frac=0.5, G0_frac=0.1):
    T = P.size
    Q = np.empty(T)
    S = S0_frac * S_max
    G = G0_frac * S_max
    for t in range(T):
        S += P[t]
        sat = max(0.0, min(1.0, S / S_max))
        et = min(S, PET[t] * sat ** gamma)
        S -= et
        excess = max(0.0, S - S_max)
        S -= excess
        Qq = f_quick * excess
        S += (1.0 - f_quick) * excess
        perc = k_perc * S
        S -= perc
        G += perc
        Qb = G / tau_b
        G -= Qb
        Q[t] = Qq + Qb
    return Q


@njit(cache=True, parallel=True, fastmath=True)
def simulate_batch(P, PET, params, S0_frac=0.5, G0_frac=0.1):
    """params shape (N, 5): columns are [S_max, f_quick, k_perc, tau_b, gamma]."""
    N = params.shape[0]
    out = np.empty((N, P.size))
    for i in prange(N):
        out[i] = simulate(P, PET, params[i, 0], params[i, 1],
                           params[i, 2], params[i, 3], params[i, 4],
                           S0_frac, G0_frac)
    return out
