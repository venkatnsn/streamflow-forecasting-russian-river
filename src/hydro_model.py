"""A 5-parameter conceptual two-bucket rainfall-runoff model (Numba-JIT'd).

Structure:

    P (mm/day)  ->  [SOIL bucket]  -- ET --> atmosphere
                          |
                  saturation excess -> Q_quick (fast runoff)
                          |
                  percolation k_perc -> [GW bucket] -- Q_base --> stream
                                                        (recession 1/tau_b)

Parameters
----------
S_max   : soil moisture capacity (mm) -- bucket size
f_quick : fraction of saturation excess routed to fast runoff (-)
k_perc  : percolation rate to groundwater store (1/day)
tau_b   : groundwater recession time constant (days)
gamma   : non-linear ET shape exponent (-) -- ET = PET * (S/S_max)^gamma

Two entry points are exposed:

* ``simulate_one`` for a single parameter set (used by the optimiser).
* ``simulate_batch`` for an N x 5 parameter matrix (used by RSA / GLUE).

Both are JIT-compiled with Numba; the batch version uses prange for
multi-core parallelism.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numba import njit, prange

# --- Public API: bounds + names ---------------------------------------------
PARAMETER_BOUNDS = {
    "S_max":   (50.0, 600.0),
    "f_quick": (0.0,    1.0),
    "k_perc":  (0.001,  0.5),
    "tau_b":   (5.0,  200.0),
    "gamma":   (0.5,    3.0),
}
PARAMETER_NAMES = list(PARAMETER_BOUNDS.keys())


# --- Numba kernels ----------------------------------------------------------
@njit(cache=True, fastmath=True)
def _simulate_kernel(P, PET, S_max, f_quick, k_perc, tau_b, gamma,
                      S0_frac, G0_frac):
    T = P.size
    Q = np.empty(T, dtype=np.float64)
    S = S0_frac * S_max
    G = G0_frac * S_max
    for t in range(T):
        S = S + P[t]
        sat = S / S_max
        if sat < 0.0:
            sat = 0.0
        elif sat > 1.0:
            sat = 1.0
        et = PET[t] * (sat ** gamma)
        if et > S:
            et = S
        S = S - et

        excess = S - S_max
        if excess > 0.0:
            S = S - excess
            Qq = f_quick * excess
            S = S + (1.0 - f_quick) * excess
        else:
            Qq = 0.0
        perc = k_perc * S
        S = S - perc
        G = G + perc
        Qb = G / tau_b
        G = G - Qb
        Q[t] = Qq + Qb
    return Q


@njit(cache=True, parallel=True, fastmath=True)
def _simulate_batch_kernel(P, PET, params, S0_frac, G0_frac):
    """params is shape (N, 5) with columns [S_max, f_quick, k_perc, tau_b, gamma]."""
    N = params.shape[0]
    T = P.size
    out = np.empty((N, T), dtype=np.float64)
    for i in prange(N):
        out[i] = _simulate_kernel(P, PET, params[i, 0], params[i, 1],
                                    params[i, 2], params[i, 3], params[i, 4],
                                    S0_frac, G0_frac)
    return out


# --- Friendly wrappers ------------------------------------------------------
@dataclass
class Parameters:
    """Holds the five parameters; each may be a scalar or 1-D array.

    Use ``.to_array()`` to get an (N, 5) matrix for the batch simulator.
    """
    S_max:   float | np.ndarray
    f_quick: float | np.ndarray
    k_perc:  float | np.ndarray
    tau_b:   float | np.ndarray
    gamma:   float | np.ndarray

    def to_array(self) -> np.ndarray:
        arrs = [np.atleast_1d(self.S_max).astype(np.float64),
                np.atleast_1d(self.f_quick).astype(np.float64),
                np.atleast_1d(self.k_perc).astype(np.float64),
                np.atleast_1d(self.tau_b).astype(np.float64),
                np.atleast_1d(self.gamma).astype(np.float64)]
        N = max(a.size for a in arrs)
        out = np.empty((N, 5), dtype=np.float64)
        for j, a in enumerate(arrs):
            out[:, j] = np.broadcast_to(a, N)
        return out


def simulate(P: np.ndarray, PET: np.ndarray, params: Parameters,
              S0_frac: float = 0.5, G0_frac: float = 0.1) -> np.ndarray:
    """Run the model on one or many parameter sets.

    Returns a 1-D array of length T if all parameters are scalars,
    otherwise an (N, T) matrix.
    """
    P   = np.ascontiguousarray(P,   dtype=np.float64)
    PET = np.ascontiguousarray(PET, dtype=np.float64)
    arr = params.to_array()
    if arr.shape[0] == 1:
        return _simulate_kernel(P, PET, arr[0, 0], arr[0, 1],
                                  arr[0, 2], arr[0, 3], arr[0, 4],
                                  S0_frac, G0_frac)
    return _simulate_batch_kernel(P, PET, arr, S0_frac, G0_frac)
