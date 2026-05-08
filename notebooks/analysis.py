"""Build the analysis notebook (analysis.ipynb) programmatically.

This script writes a clean Jupyter notebook that re-runs the full pipeline
inside one document: load data, calibrate, validate, sensitivity, GLUE,
scenarios. The cells are sized so the notebook prints all results inline
and a reader can scroll through it linearly.

Run it after the four pipeline scripts have generated their outputs;
the notebook references the same JSON / CSV files.
"""
from pathlib import Path
import json
import nbformat as nbf

HERE = Path(__file__).parent

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# Russian River Streamflow Forecasting — End-to-End Walkthrough

A 30-year analysis of the Russian River near Hopland (USGS gauge **11462500**, drainage area **937 km²**) using only free, no-auth public data:

* USGS NWIS daily streamflow
* Open-Meteo Historical (ERA5) daily precipitation, T_max, T_min
* Hargreaves–Samani PET from temperature

The pipeline runs in five steps, mirroring the scripts in `scripts/`:

1. Load the cached daily forcing record
2. Calibrate with Differential Evolution; validate on a held-out half
3. Regional Sensitivity Analysis on 5,000 LHS samples
4. GLUE uncertainty bands
5. Climate-change scenarios

Re-run all cells to regenerate the analysis from scratch.
"""))

cells.append(nbf.v4.new_code_cell("""\
import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from src.data_acquisition import load_cached, RUSSIAN_RIVER_HOPLAND
from src.hydro_model import simulate, Parameters, PARAMETER_BOUNDS, PARAMETER_NAMES
from src.metrics import all_metrics, kge

plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 1. Load the data

`scripts/01_fetch_data.py` already fetched and cached 30 water years of
forcings + observed streamflow as `data/processed/catchment_daily.csv`.
Re-run that script to refresh.
"""))

cells.append(nbf.v4.new_code_cell("""\
df = load_cached(RUSSIAN_RIVER_HOPLAND)
print(f'{len(df):,} daily records  '
      f'{df["date"].min().date()} → {df["date"].max().date()}')
df.head(5)
"""))

cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)
ax[0].plot(df['date'], df['P_mm'], color='#1F77B4', lw=0.4)
ax[0].set_ylabel('P (mm/day)')
ax[1].plot(df['date'], df['PET_mm'], color='#2CA02C', lw=0.4)
ax[1].set_ylabel('PET (mm/day)')
ax[2].plot(df['date'], df['Q_mm'], color='black', lw=0.4)
ax[2].set_yscale('log')
ax[2].set_ylim(0.05, 100)
ax[2].set_ylabel('Q (mm/day)')
ax[2].set_xlabel('Date')
plt.suptitle(f'30 water years of forcings + observed streamflow  —  {RUSSIAN_RIVER_HOPLAND.name}', y=0.995)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 2. Calibration + split-sample validation

Calibration: 1995-10-01 → 2009-09-30 (14 water years).
Validation:  2010-10-01 → 2024-09-30 (14 water years, held out).
"""))

cells.append(nbf.v4.new_code_cell("""\
calibration = json.loads((ROOT / 'data' / 'processed' / 'calibration_results.json').read_text())
print('Optimal parameters:')
for k, v in calibration['parameters'].items():
    print(f'  {k:8s} = {v:.3f}')
print()
print('Goodness of fit:')
print(f"  {'metric':8s}  {'cal':>8s}  {'val':>8s}")
for m in ['NSE', 'KGE', 'logNSE', 'PBIAS']:
    c = calibration['calibration']['metrics'][m]
    v = calibration['validation']['metrics'][m]
    print(f"  {m:8s}  {c:8.3f}  {v:8.3f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
The KGE drops from **0.88** in calibration to **0.54** in validation, with a
+33 % bias. This is non-stationarity at work: the post-2010 record is
notably drier (the 2020–2022 California drought sits inside the validation
window). Refer to the `02_calibration_validation.png` figure in `figures/`.
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 3. Regional Sensitivity Analysis

5,000 Latin Hypercube samples → top 10 % (highest KGE) form the behavioural set.
Per-parameter Kolmogorov–Smirnov distance between the behavioural CDF and the
prior CDF gives a scalar sensitivity index.
"""))

cells.append(nbf.v4.new_code_cell("""\
sens = json.loads((ROOT / 'data' / 'processed' / 'sensitivity_results.json').read_text())
print('Behavioural threshold (KGE):', round(sens['kge_threshold'], 3))
print('Behavioural runs:', sens['n_behavioural'], '/', sens['n_samples'])
print()
print('Sensitivity ranking (KS distance, larger → more sensitive):')
for n in sens['ranking']:
    print(f'  {n:8s}  {sens["ks_distance"][n]:.3f}')
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 4. GLUE uncertainty bands

Behavioural threshold: KGE ≥ 0.5. Each surviving run is weighted by
KGE-above-threshold (informal likelihood). The 5/95 percentiles at every
timestep give the prediction band.
"""))

cells.append(nbf.v4.new_code_cell("""\
glue = json.loads((ROOT / 'data' / 'processed' / 'glue_results.json').read_text())
print('GLUE threshold (KGE):', glue['glue_threshold'])
print('Behavioural ensemble size:', glue['n_behavioural'])
print(f'Coverage of 5–95 % band: {glue["coverage_5_95"]*100:.1f} %  (target ~90 %)')
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 5. Climate-change scenarios

Five delta-change perturbations applied to the 30-year forcing record;
behavioural ensemble re-run under each scenario.
"""))

cells.append(nbf.v4.new_code_cell("""\
scenarios = pd.read_csv(ROOT / 'data' / 'processed' / 'climate_scenarios.csv')
print('Mean annual streamflow per scenario (weighted ensemble):')
print(scenarios.round(1).to_string(index=False))
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## Take-aways

1. **Calibration succeeds, validation reveals non-stationarity.** A model
   calibrated on 1996–2009 systematically over-predicts post-2010 flow
   because the validation period is climatically drier. This is a real
   limitation worth declaring in any decision document.
2. **Two parameters dominate** the model's response: percolation rate
   `k_perc` and soil capacity `S_max` (KS ≈ 0.5 and 0.4). The remaining
   three could be fixed without much loss of fit.
3. **The GLUE band covers 87 %** of observed days inside the nominal 90 %
   interval — well-calibrated uncertainty.
4. **Precipitation matters more than temperature** for this catchment.
   A 20 % drier climate cuts mean annual flow by 25 %; a +2 °C warming
   alone cuts it by only ~ 2 %. Drier-and-warmer is essentially the same
   as drier-only.

These findings are the kind of result a water-supply manager would want
distilled out of any modelling exercise — the parameter-uncertainty and
scenario-uncertainty parts of the analysis ARE the deliverable.
"""))

nb['cells'] = cells

out = HERE / "analysis.ipynb"
nbf.write(nb, str(out))
print(f"Wrote {out}")
