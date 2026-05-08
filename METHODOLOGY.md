# Methodology — Russian River Streamflow Pipeline

## Catchment

* USGS gauge **11462500** — Russian River near Hopland, CA
* Coordinates: 38.992 °N, 123.130 °W
* Drainage area: 937 km² (362 mi²)
* Climate: Mediterranean, ~ 1000 mm/yr precipitation, ~ 1300 mm/yr PET
* Regulation: minor upstream regulation by Lake Mendocino (Coyote Valley Dam)
  — the gauge sees a partly natural flow regime
* Period analysed: water years 1996 – 2024 (calendar dates 1995-10-01 to
  2024-09-30)

## Forcing data

| Variable | Source | Spatial reference | Temporal resolution |
|---|---|---|---|
| Streamflow Q (cfs → mm/day) | USGS NWIS via `dataretrieval` | Gauge 11462500 | Daily |
| Precipitation P (mm/day) | Open-Meteo Historical (ERA5 reanalysis) | Lat/lon at the gauge, ~ 25 km grid | Daily |
| Air temperature Tmax, Tmin (°C) | Open-Meteo Historical (ERA5) | Same grid cell | Daily |
| Reference ET (mm/day) | Hargreaves & Samani (1985) from Tmax, Tmin, day-of-year, latitude | Same grid cell | Daily |

A fully climatic forcing record is preferable to a single station because
ERA5 spatially-averaged precipitation matches the catchment-average
forcing better than a point gauge (which can miss orographic uplift).

The merged daily forcing CSV is committed to `data/processed/` so the
pipeline is reproducible without network access after the first run.

## Hydrologic model

A 5-parameter conceptual two-bucket model. The state vector is
(soil moisture S, groundwater G).

| Parameter | Symbol | Bounds | Physical meaning |
|---|---|---|---|
| Soil moisture capacity (mm) | `S_max` | 50 – 600 | Maximum soil water holding capacity |
| Quickflow fraction (-) | `f_quick` | 0 – 1 | Saturation excess routed to fast runoff |
| Percolation rate (1/day) | `k_perc` | 0.001 – 0.5 | Daily fraction of soil water percolating to GW |
| Baseflow recession (day) | `tau_b` | 5 – 200 | GW reservoir time constant |
| ET shape exponent (-) | `gamma` | 0.5 – 3.0 | Non-linearity of ET = PET · (S/Smax)^gamma |

Daily updates (per day t):

```
ET   = PET · (S / S_max) ^ gamma
S    = S + P - ET
Qq   = f_quick · max(0, S - S_max)            # quickflow
S    = S - Qq
perc = k_perc · S
S    = S - perc
G    = G + perc
Qb   = G / tau_b                              # baseflow
G    = G - Qb
Q    = Qq + Qb
```

The model is **vectorised** — `S_max, f_quick, k_perc, tau_b, gamma` may
each be a 1-D array of length N, in which case the simulator returns an
(N, T) matrix of streamflow. This is what makes the 5,000-run RSA tractable.

## Calibration

* **Objective:** maximise Kling-Gupta Efficiency (KGE).
* **Optimiser:** `scipy.optimize.differential_evolution`, default settings,
  `seed=42`, `tol=1e-6`, `maxiter=200`.
* **Calibration period:** water years 1996 – 2009 (1995-10-01 to 2009-09-30,
  with one-year warmup discarded from the metric calculation).
* **Validation period:** water years 2011 – 2024 (independent split-sample
  test; one-year warmup also discarded).

KGE is preferred over plain NSE because the lake / Mediterranean climate
makes pure squared-error metrics over-emphasise winter floods at the
expense of dry-season baseflow. KGE balances bias, variability, and
correlation.

## Sensitivity analysis (RSA)

* 5,000 Latin Hypercube samples drawn over the prior parameter bounds.
* Vectorised simulation; each realisation scored with KGE.
* Behavioural threshold: top 10 % (highest KGE).
* Per-parameter Kolmogorov-Smirnov distance between the behavioural
  CDF and the prior CDF gives the sensitivity index. Values close to 0
  mean the behavioural CDF is indistinguishable from the prior
  (the parameter is insensitive to the objective). Values close to 1
  mean the parameter is strongly informed by the data.

## Uncertainty quantification (GLUE)

* The same 5,000 LHS realisations are reused.
* Behavioural threshold: KGE >= 0.5 (relaxed to the top 5 % if too few
  members survive, so the ensemble is always at least 30 members).
* **Informal likelihood:** `L_i = (KGE_i - min KGE)`, then normalised so
  weights sum to 1.
* The weighted 5/50/95 percentiles of the ensemble at every timestep
  define the prediction band.
* **Coverage diagnostic:** fraction of observed days falling inside the
  5-95 % band. A well-specified ensemble should hit ~ 0.90.

## Climate-change scenarios

Five forcing perturbations applied to the 1995-2024 record:

| Scenario | P factor | ΔT (°C) |
|---|---|---|
| Baseline | 1.00 | 0 |
| Wetter (+20% P) | 1.20 | 0 |
| Drier (-20% P) | 0.80 | 0 |
| Warmer (+2 °C) | 1.00 | +2 |
| Drier + Warmer | 0.80 | +2 |

PET is recomputed from the perturbed Tmax, Tmin via Hargreaves so the
warming scenarios increase atmospheric demand consistently.

For each scenario the behavioural ensemble is rerun on the perturbed
forcing record, and the weighted 5/95 percentiles of mean annual flow
are reported. This produces a cross-scenario spread (climate
uncertainty) that can be compared to the within-scenario spread
(parameter uncertainty).

## References

* Beven, K. & Binley, A. (1992). *Hydrol. Process.* 6, 279.
* Gupta, H. V., Kling, H., Yilmaz, K. K., Martinez, G. F. (2009). *J. Hydrol.* 377, 80.
* Hargreaves, G. H. & Samani, Z. A. (1985). *Appl. Eng. Agric.* 1, 96.
* Klemeš, V. (1986). *Hydrol. Sci. J.* 31, 13.
* Spear, R. C. & Hornberger, G. M. (1980). *Water Res.* 14, 43.
* Storn, R. & Price, K. (1997). *J. Glob. Optim.* 11, 341.
