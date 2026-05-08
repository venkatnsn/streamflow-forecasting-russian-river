# Streamflow Forecasting Under Climate Change — Russian River, California

> **Read the full write-up: [venkatnsn.github.io/streamflow-forecasting-russian-river](https://venkatnsn.github.io/streamflow-forecasting-russian-river/)**

End-to-end hydrological modelling pipeline for the Russian River near Hopland
(USGS gauge 11462500), built entirely on free, no-auth public data:

* **Streamflow:** USGS National Water Information System (`dataretrieval`)
* **Climate forcings:** Open-Meteo Historical (ERA5 reanalysis)
* **Period analysed:** 1995-10-01 to 2024-09-30 (30 water years, 10,958 daily records)

The pipeline covers the standard professional workflow — calibration,
split-sample validation, Regional Sensitivity Analysis, GLUE uncertainty
quantification, and climate-change scenario projection — and produces
publication-quality figures and a reproducible artefact trail at every step.

> Catchment fact sheet | USGS 11462500 · drainage area 937 km² · Mediterranean climate (Pacific winters, dry summers) · regulated by Lake Mendocino upstream · principal water source for Sonoma and Mendocino counties.

---

## Project structure

```
streamflow-forecasting-russian-river/
├── README.md
├── requirements.txt
├── src/
│   ├── data_acquisition.py     # USGS + Open-Meteo fetchers, caching
│   ├── pet.py                  # Hargreaves-Samani PET
│   ├── hydro_model.py          # 5-parameter two-bucket model (vectorised)
│   └── metrics.py              # NSE, KGE, log-NSE, PBIAS
├── scripts/
│   ├── 01_fetch_data.py        # ~30 s, network required
│   ├── 02_calibrate_validate.py# ~2 min, Differential Evolution
│   ├── 03_sensitivity.py       # ~2 min, 5,000 LHS runs
│   ├── 04_glue.py              # ~30 s, weighted ensemble
│   └── 05_climate_scenarios.py # ~30 s, 5 scenarios x ensemble
├── notebooks/
│   └── analysis.ipynb          # narrative walk-through
├── data/
│   ├── raw/                    # cached USGS + Open-Meteo CSV
│   └── processed/              # merged daily forcing record + JSON results
├── figures/                    # rendered PNGs (one per script)
├── METHODOLOGY.md              # extended methods document
└── docs/
    ├── index.html              # blog-format article (served by GitHub Pages)
    └── images/                 # article figures
```

---

## Quick start

```bash
git clone https://github.com/venkatnsn/streamflow-forecasting-russian-river.git
cd streamflow-forecasting-russian-river
python -m pip install -r requirements.txt

python scripts/01_fetch_data.py            # downloads ~10k days of data
python scripts/02_calibrate_validate.py    # calibrates the model
python scripts/03_sensitivity.py           # RSA on top 10% of 5,000 LHS runs
python scripts/04_glue.py                  # GLUE 5-95% prediction band
python scripts/05_climate_scenarios.py     # 5 climate scenarios
```

Each script writes a JSON / CSV result file to `data/processed/` and a PNG to
`figures/`. The whole pipeline runs end-to-end in under 6 minutes on a laptop.

---

## Methodology summary

### Model — 5-parameter two-bucket conceptual model

```
P (rainfall)  ->  [SOIL  bucket]  -- ET ----------------> atmosphere
                          |
                  saturation excess -> Q_quick (fast runoff)
                          |
                  percolation k_perc -> [GW bucket] -- Q_base --> stream
                                                        (1 / tau_b)
```

Five free parameters: soil capacity `S_max`, quickflow fraction `f_quick`,
percolation rate `k_perc`, baseflow recession time `tau_b`, and a non-linear
ET shape exponent `gamma`. The model is fully vectorised — 5,000 LHS runs
of 30 years each complete in seconds.

### Calibration — Differential Evolution on KGE

Differential Evolution (Storn & Price, 1997) is used to maximise the
Kling-Gupta Efficiency (Gupta et al., 2009), which decomposes goodness of fit
into correlation, bias, and variability components.

* Calibration period: water years 1996-2009 (one year warmup, then 13 years)
* Validation period:  water years 2011-2024 (one year warmup, then 13 years)

The split-sample (Klemeš, 1986) is the only test of generalisation that
matters — calibration KGE alone is misleading.

### Sensitivity — Regional Sensitivity Analysis

5,000 Latin Hypercube samples are drawn over the prior bounds. The top
10% (by KGE) form the *behavioural* set. Per-parameter
Kolmogorov-Smirnov distance between the behavioural CDF and the prior
CDF gives a scalar sensitivity index (Spear & Hornberger, 1980).

### Uncertainty — GLUE

Behavioural threshold: `KGE >= 0.5`. The informal likelihood
(Beven & Binley, 1992) weights each behavioural run by its
KGE-above-threshold; the weighted 5-50-95 percentiles at every
timestep give the prediction band. The fraction of observed days
falling inside the 5-95% band is reported as a coverage diagnostic
(target ~ 90%).

### Climate scenarios — delta-change ensemble

Five scenarios are applied to the 30-year forcing record:

| Scenario | Precipitation | Temperature |
|---|---|---|
| Baseline | x 1.00 | + 0 °C |
| Wetter (+20% P) | x 1.20 | + 0 °C |
| Drier (-20% P) | x 0.80 | + 0 °C |
| Warmer (+2 °C) | x 1.00 | + 2 °C |
| Drier + Warmer | x 0.80 | + 2 °C |

PET is recomputed via Hargreaves-Samani (1985) under the perturbed
temperatures so that warming increases atmospheric demand consistently.
The behavioural ensemble is rerun under each scenario, weighted by the
informal likelihood.

---

## Key results

After running the pipeline end-to-end the following artefacts are produced:

| File | What it contains |
|---|---|
| `data/processed/catchment_daily.csv` | 30 years of merged daily forcings + observed Q |
| `data/processed/calibration_results.json` | Optimal parameters, cal/val NSE/KGE/PBIAS |
| `data/processed/sensitivity_results.json` | KS distance ranking of the 5 parameters |
| `data/processed/glue_results.json` | 5-95% band coverage diagnostic |
| `data/processed/climate_scenarios.csv` | Mean annual streamflow per scenario, with bands |
| `figures/02_calibration_validation.png` | Hydrograph, observed vs simulated, cal & val |
| `figures/03_sensitivity_cdf.png` | RSA CDFs and KS distances per parameter |
| `figures/04_glue_uncertainty_band.png` | GLUE prediction interval, recent six water years |
| `figures/05_climate_scenarios.png` | Annual flow ensemble per scenario |

Headline numbers are dropped into `METHODOLOGY.md` after the
calibration run completes.

---

## Why this catchment

The Russian River is a useful real-world case study because:

* The Mediterranean climate (very wet winters, very dry summers) makes
  it a *non-stationary* system — fitting both regimes simultaneously
  forces the model to identify both fast (storm runoff) and slow
  (groundwater) processes.
* Recent decades include both a record drought (2020-2022) and several
  major flood years, so the calibration record straddles the regimes
  most of interest to water managers.
* USGS 11462500 has continuous data back to 1939 and is one of the
  most reliable gauges on the California North Coast.
* It is the principal water-supply river for Sonoma and Mendocino
  counties — climate-change projections here have direct policy
  consequences.

---

## References

* Beven, K. & Binley, A. (1992). The future of distributed models: model calibration and uncertainty prediction. *Hydrological Processes* 6, 279-298.
* Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria. *Journal of Hydrology* 377, 80-91.
* Hargreaves, G. H. & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. *Applied Engineering in Agriculture* 1, 96-99.
* Klemeš, V. (1986). Operational testing of hydrological simulation models. *Hydrological Sciences Journal* 31(1), 13-24.
* Spear, R. C. & Hornberger, G. M. (1980). Eutrophication in Peel Inlet II: identification of critical uncertainties via generalized sensitivity analysis. *Water Research* 14, 43-49.
* Storn, R. & Price, K. (1997). Differential Evolution - a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization* 11, 341-359.

---

## License

MIT (see `LICENSE`). All data sources are public-domain (USGS) or
open-licensed (Open-Meteo, CC-BY-4.0).
