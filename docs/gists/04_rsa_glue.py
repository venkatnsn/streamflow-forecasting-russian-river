"""Regional Sensitivity Analysis + GLUE uncertainty in one script."""
import numpy as np
from scipy.stats import qmc

# 5,000-sample Latin Hypercube over the prior parameter bounds
N = 5000
sampler = qmc.LatinHypercube(d=5, seed=2026)
u = sampler.random(N)
bounds = np.array(BOUNDS)
samples = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])

# Vectorised simulation, score every run
Q_all = simulate_batch(P, PET, samples)
kge_all = np.array([kge(Q_all[i, WARMUP:], Qobs[WARMUP:]) for i in range(N)])

# --- Regional Sensitivity: top 10% are behavioural -----------------------
behave = kge_all >= np.quantile(kge_all, 0.90)

ks_distance = {}
for j, name in enumerate(['S_max','f_quick','k_perc','tau_b','gamma']):
    lo, hi = bounds[j]
    grid = np.linspace(0, 1, 200)
    s_all = np.sort((samples[:, j] - lo) / (hi - lo))
    s_be  = np.sort((samples[behave, j] - lo) / (hi - lo))
    F_all = np.searchsorted(s_all, grid) / N
    F_be  = np.searchsorted(s_be,  grid) / behave.sum()
    ks_distance[name] = float(np.max(np.abs(F_be - F_all)))

print("Sensitivity ranking (KS distance, larger = more sensitive):")
for n, d in sorted(ks_distance.items(), key=lambda kv: -kv[1]):
    print(f"  {n:8s}  {d:.3f}")

# --- GLUE uncertainty: behavioural ensemble + likelihood weights --------
glue_thresh = 0.5
behave = kge_all >= glue_thresh                     # tighter threshold
likelihood = kge_all[behave] - kge_all[behave].min()
weights = likelihood / likelihood.sum()

# Run the behavioural set on any new forcing (here: a future scenario)
Q_ensemble = simulate_batch(P_future, PET_future, samples[behave])

def weighted_quantile(values, w, q):
    order = np.argsort(values)
    cw = np.cumsum(w[order]) / w.sum()
    return values[order][np.searchsorted(cw, q)]

q05 = np.array([weighted_quantile(Q_ensemble[:, t], weights, 0.05)
                 for t in range(Q_ensemble.shape[1])])
q50 = np.array([weighted_quantile(Q_ensemble[:, t], weights, 0.50)
                 for t in range(Q_ensemble.shape[1])])
q95 = np.array([weighted_quantile(Q_ensemble[:, t], weights, 0.95)
                 for t in range(Q_ensemble.shape[1])])
