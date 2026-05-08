# Forecasting California's Russian River Under Climate Change

*A 30-year, end-to-end Python pipeline: real USGS streamflow, ERA5 climate forcings, calibration, sensitivity, GLUE uncertainty, and climate-change scenarios — every step reproducible from one repository.*

<<FIG_HERO>>

---

The Russian River keeps the lights on for half a million people in Sonoma and Mendocino counties, California. It also flooded twice in the past decade and emptied to a trickle during the 2020–2022 drought. Forecasting how it will behave under a warmer, drier future is a real engineering problem, not an academic exercise.

This walkthrough builds that forecast end-to-end in Python, using only free, no-auth public data. The whole pipeline runs in under six minutes on a laptop. Every artefact — code, cached data, results, figures — is in a single repository so the result is fully reproducible and any other catchment can swap in by changing two numbers.

Repository: **[github.com/venkatnsn/streamflow-forecasting-russian-river](https://github.com/venkatnsn/streamflow-forecasting-russian-river)**

By the end you will know how to:

1. Pull 30 years of daily streamflow + climate forcings from public APIs in a few lines of Python
2. Build a vectorised, numba-accelerated rainfall–runoff model that runs 5,000 simulations per second
3. Calibrate it with Differential Evolution and test it with a real split-sample validation
4. Identify which parameters actually matter through Regional Sensitivity Analysis
5. Put defensible error bars on the forecast with GLUE
6. Project streamflow under five climate-change scenarios and quantify which type of uncertainty dominates

---

## 1. The catchment

USGS gauge **11462500** sits on the Russian River near Hopland, CA. Three numbers define the system:

- **937 km²** drainage area
- **~1,000 mm/yr** mean precipitation, mostly Pacific winter storms
- **~1,300 mm/yr** mean potential evapotranspiration

The output of those three numbers — streamflow at the gauge — is what the model has to predict.

A Mediterranean climate makes this a particularly interesting test case. Winters are wet and cool (low atmospheric demand, lots of runoff). Summers are dry and hot (high atmospheric demand, baseflow only). A model that fits one season but not the other is a useless model. The split-sample test in §3 is what catches that.

---

## 2. Data — 30 water years from two free APIs

Two endpoints, neither of which requires authentication or a paid subscription:

- **USGS NWIS** via the `dataretrieval` Python package — daily mean discharge in cubic feet per second, converted to mm/day depth equivalent over the catchment area.
- **Open-Meteo Historical** — daily precipitation, T_max, T_min from the ERA5 reanalysis (~25 km grid). The catchment-average rainfall is a better forcing than any single rain gauge — point gauges miss orographic uplift.

Reference evapotranspiration is computed via Hargreaves & Samani (1985), which needs only T_max, T_min, day-of-year and latitude.

<<GIST_DATA_ACQUISITION>>

The merged record contains 10,958 daily rows (1995-01-01 → 2024-12-31). Mean precipitation 985 mm/yr, mean PET 1,292 mm/yr, mean streamflow 575 mm/yr — a runoff coefficient of 0.58, exactly what you would expect for a Mediterranean Pacific-coast catchment.

---

## 3. The model — five parameters, two buckets

A conceptual lumped-parameter model with five free parameters:

```
P (rainfall)  ->  [SOIL bucket]  -- ET --> atmosphere
                          |
                   saturation excess --> Q_quick (fast runoff)
                          |
                 percolation (k_perc) --> [GW bucket] -- Q_base --> stream
                                                          (1/tau_b)
```

| Parameter | Range | Physical meaning |
|---|---|---|
| `S_max` | 50 – 600 mm | Maximum soil moisture capacity |
| `f_quick` | 0 – 1 | Fraction of saturation excess routed to fast runoff |
| `k_perc` | 0.001 – 0.5 / day | Percolation rate from soil to groundwater |
| `tau_b` | 5 – 200 days | Groundwater (baseflow) recession time constant |
| `gamma` | 0.5 – 3 | Non-linear ET shape exponent (ET = PET · (S/S_max)^γ) |

The whole simulation loop is JIT-compiled with Numba so 5,000 30-year runs of 10,958 days complete in about 0.2 seconds on a single core, and the batch version uses `prange` for multi-core parallelism. This is what makes the sensitivity and GLUE analyses tractable.

<<GIST_TWO_BUCKET_MODEL>>

---

## 4. Calibration and split-sample validation

The 30-year record is split in half:

- **Calibration:** water years 1996 – 2009 (first year warmup, then 13 years)
- **Validation:** water years 2011 – 2024 (held out, also with first-year warmup)

Differential Evolution maximises the **Kling–Gupta Efficiency** (Gupta et al. 2009), which decomposes goodness-of-fit into correlation, bias, and variability terms — a more balanced metric than plain NSE for streamflow.

<<GIST_CALIBRATE_DE>>

The optimised parameters and goodness-of-fit:

| Parameter | Optimum |
|---|---|
| `S_max` | 248.0 mm |
| `f_quick` | 0.23 |
| `k_perc` | 0.0042 / day |
| `tau_b` | 17.4 days |
| `gamma` | 2.84 |

| Metric | Calibration | Validation |
|---|---|---|
| NSE | 0.770 | 0.539 |
| KGE | 0.885 | 0.537 |
| log-NSE | 0.724 | 0.518 |
| PBIAS | −0.1 % | **+33.3 %** |

<<FIG_CAL_VAL>>

The +33 % bias in validation is a real, honest finding — and the most important number in the whole analysis. The validation period contains the 2020–2022 California drought, which was substantially drier than anything in the calibration period. A model calibrated on 1996–2009 climate over-predicts post-2010 streamflow because the historical record it was tuned against is no longer representative.

This is **non-stationarity**, and it is the central problem of climate-change hydrology. No amount of better optimisation will fix it. The right responses are to use a regime-spanning calibration period whenever possible, and to put honest error bars on any forecast — which is what the next two sections do.

---

## 5. Sensitivity — which parameters actually drive the model?

Two questions to answer before trusting any model:

- Which parameter, if perturbed, would ruin the fit?
- Which one barely matters?

The professional answer is **Regional Sensitivity Analysis** (Spear & Hornberger 1980): draw a Latin Hypercube sample over the parameter bounds, run the model for each, classify by goodness-of-fit, and compare the cumulative distribution of each parameter inside the *behavioural* set against the prior. A parameter whose behavioural CDF is bent away from the diagonal is sensitive. One that follows the diagonal is not.

<<GIST_RSA_GLUE>>

5,000 LHS runs, top 10 % behavioural set, KS distance between behavioural CDF and prior:

| Parameter | KS distance | Interpretation |
|---|---|---|
| `k_perc` | **0.49** | Most sensitive — controls the soil-to-groundwater partition |
| `S_max` | **0.40** | Soil-bucket size — sets the threshold for runoff generation |
| `tau_b` | 0.21 | Baseflow recession — moderately constrained |
| `gamma` | 0.17 | ET shape — weakly constrained |
| `f_quick` | **0.04** | Almost unconstrained — the fast/slow split is invisible to KGE |

<<FIG_SENSITIVITY>>

Two parameters dominate the behaviour of this model. The other three could be fixed at literature defaults and the fit would barely change. That is a legitimate model-reduction signal — a real-world payoff of doing the sensitivity analysis instead of skipping straight to calibration.

---

## 6. Uncertainty bands — GLUE

A single best-fit forecast is a lie of omission. Many parameter sets fit the data almost equally well — this is **equifinality** (Beven & Binley 1992). They will all give different forecasts under future conditions. The honest forecast is an ensemble.

**GLUE — Generalised Likelihood Uncertainty Estimation:**

1. Take the LHS runs that exceed a behavioural threshold (here KGE ≥ 0.5)
2. Weight each surviving run by an *informal likelihood* — KGE-above-threshold normalised to sum to 1
3. The weighted 5/50/95 percentiles of the ensemble at every timestep give the prediction band
4. Diagnostic: the fraction of observed days that fall inside the 5–95 % band — a target of ~ 90 %

<<FIG_GLUE>>

The 5–95 % band covers **86.6 %** of observed days — close to the textbook 90 % target. The ensemble is well-calibrated: when the model says "this day will fall between Q_low and Q_high with 90 % confidence," it is right about 90 % of the time.

That is what makes GLUE a defensible engineering tool. A water manager reading this band can say "I have a 90 % chance of seeing a flow between X and Y on any given day" — a far more useful statement than the deterministic "the model predicts Z."

---

## 7. Climate-change scenarios

Five scenarios applied to the 30-year forcing record. PET is recomputed from perturbed temperatures so warming consistently increases atmospheric demand.

| Scenario | P factor | ΔT (°C) | Mean Q (mm/yr) | 5 – 95 % band |
|---|---|---|---|---|
| Baseline | 1.00 | 0 | 623 | 525 – 739 |
| Wetter (+20 % P) | 1.20 | 0 | 787 | 689 – 908 |
| Drier (−20 % P) | 0.80 | 0 | 466 | 367 – 573 |
| Warmer (+2 °C) | 1.00 | +2 | 611 | 511 – 728 |
| Drier + Warmer | 0.80 | +2 | 455 | 355 – 564 |

<<FIG_SCENARIOS>>

Two findings stand out:

- **Precipitation matters far more than temperature.** A 20 % cut in rainfall reduces mean annual flow by 25 %. A 2 °C warming alone reduces it by only ~ 2 %. Drier-and-warmer is essentially the same as drier-only — the temperature signal is buried in the precipitation signal.
- **Within-scenario uncertainty (parameter spread, ~ 200 mm/yr) is comparable to between-scenario uncertainty (~ 320 mm/yr from baseline to drier).** Reducing parameter uncertainty through better calibration would meaningfully sharpen the climate forecast — which is the case for investing in better data, not just more sophisticated models.

---

## 8. The general workflow

The same pipeline applies to any catchment, any model, any country. Change two numbers (the USGS gauge ID and the lat/lon) and re-run.

| Step | Question answered | Method |
|---|---|---|
| 1. Acquire data | What goes in? | Public APIs (USGS, Open-Meteo, BoM, IMD…) |
| 2. Build the model | What processes matter? | Conceptual reasoning + literature |
| 3. Calibrate | What parameter values fit observed data? | Differential Evolution on KGE |
| 4. Validate | Does the fit generalise? | Split-sample test |
| 5. Sensitivity | Which parameters matter? | RSA (LHS + behavioural CDFs) |
| 6. Uncertainty | How confident is the forecast? | GLUE (likelihood-weighted ensemble) |
| 7. Forecast | What does the future look like? | Ensemble run under perturbed forcings |

Master these seven steps and any environmental model becomes tractable.

---

## What's in the repository

The full code, data, and figures are at **[github.com/venkatnsn/streamflow-forecasting-russian-river](https://github.com/venkatnsn/streamflow-forecasting-russian-river)**. Quick start:

```bash
git clone https://github.com/venkatnsn/streamflow-forecasting-russian-river.git
cd streamflow-forecasting-russian-river
pip install -r requirements.txt

python scripts/01_fetch_data.py            # download 30 years of data
python scripts/02_calibrate_validate.py    # calibrate + split-sample test
python scripts/03_sensitivity.py           # 5,000-run RSA
python scripts/04_glue.py                  # GLUE prediction band
python scripts/05_climate_scenarios.py     # 5 climate scenarios
```

A Jupyter notebook in `notebooks/analysis.ipynb` re-runs the pipeline as a single narrative, and `METHODOLOGY.md` documents every assumption.

---

## References

- Beven, K. & Binley, A. (1992). The future of distributed models: model calibration and uncertainty prediction. *Hydrological Processes* 6, 279–298.
- Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria. *Journal of Hydrology* 377, 80–91.
- Hargreaves, G. H. & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. *Applied Engineering in Agriculture* 1, 96–99.
- Klemeš, V. (1986). Operational testing of hydrological simulation models. *Hydrological Sciences Journal* 31(1), 13–24.
- Spear, R. C. & Hornberger, G. M. (1980). Eutrophication in Peel Inlet II: identification of critical uncertainties via generalized sensitivity analysis. *Water Research* 14, 43–49.
- Storn, R. & Price, K. (1997). Differential Evolution — a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization* 11, 341–359.

---

If this article was useful, a clap and a follow help the work reach more readers. Topics worth covering next: Bayesian MCMC for environmental models, multi-catchment regionalisation, and CMIP6-driven hydrology under SSP scenarios.
