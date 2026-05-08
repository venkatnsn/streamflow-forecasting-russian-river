"""Real-time data acquisition for the Russian River near Hopland (USGS 11462500).

Two free, no-auth sources are used:

* USGS NWIS via the ``dataretrieval`` package for daily mean discharge.
* Open-Meteo Historical archive for daily precipitation, T_max, T_min
  (ERA5 reanalysis, ~25 km grid).

The merged daily forcing record is cached as CSV so downstream scripts
can be re-run without hitting either API.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests

# Suppress dataretrieval deprecation noise — the v2 transition is in 2027.
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Catchment metadata -----------------------------------------------------
@dataclass(frozen=True)
class Catchment:
    name: str
    site_no: str
    latitude: float
    longitude: float
    area_km2: float
    timezone: str

RUSSIAN_RIVER_HOPLAND = Catchment(
    name="Russian River near Hopland, CA",
    site_no="11462500",
    latitude=38.992,
    longitude=-123.130,
    area_km2=937.0,           # USGS reports 362 sq mi
    timezone="America/Los_Angeles",
)

CFS_TO_M3S = 0.0283168                       # cubic feet per second -> m^3/s
SECONDS_PER_DAY = 86400.0


def fetch_streamflow(catchment: Catchment, start: str, end: str) -> pd.DataFrame:
    """Pull daily mean discharge from USGS NWIS, return a tidy DataFrame.

    Returns columns ``date``, ``Q_cfs``, ``Q_mm`` (depth-equivalent over the
    catchment area). ``Q_mm`` is what the lumped hydrologic model actually
    fits, since rainfall is also in mm/day.
    """
    from dataretrieval import nwis
    df, _ = nwis.get_dv(sites=catchment.site_no, parameterCd="00060",
                          start=start, end=end)
    df = df.rename(columns={"00060_Mean": "Q_cfs"})[["Q_cfs"]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df.sort_index()

    # Depth conversion: Q_mm/day = Q_m3s * 86400 / (area_m2) * 1000
    area_m2 = catchment.area_km2 * 1e6
    df["Q_mm"] = df["Q_cfs"] * CFS_TO_M3S * SECONDS_PER_DAY / area_m2 * 1000.0
    return df.reset_index()


def fetch_meteorology(catchment: Catchment, start: str, end: str) -> pd.DataFrame:
    """Pull daily precip + T_max + T_min from Open-Meteo (ERA5 reanalysis).

    Returns columns ``date``, ``P_mm``, ``T_max_C``, ``T_min_C``, ``T_mean_C``.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": catchment.latitude,
        "longitude": catchment.longitude,
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": catchment.timezone,
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()["daily"]
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "time": "date",
        "precipitation_sum": "P_mm",
        "temperature_2m_max": "T_max_C",
        "temperature_2m_min": "T_min_C",
    })
    df["date"] = pd.to_datetime(df["date"])
    df["T_mean_C"] = (df["T_max_C"] + df["T_min_C"]) / 2.0
    return df


def merge_and_cache(catchment: Catchment, start: str, end: str,
                    cache_dir: Path | None = None) -> pd.DataFrame:
    """Fetch both sources, merge on date, save to cache, return the merged frame."""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    flow_path = cache_dir / f"streamflow_usgs_{catchment.site_no}.csv"
    meteo_path = cache_dir / "meteo_openmeteo.csv"
    merged_path = cache_dir.parent / "processed" / "catchment_daily.csv"

    print(f"  [1/3] streamflow ({catchment.name}, {start} -> {end}) ...")
    flow = fetch_streamflow(catchment, start, end)
    flow.to_csv(flow_path, index=False)
    print(f"        {len(flow):,} daily records -> {flow_path.name}")

    print(f"  [2/3] meteorology (Open-Meteo ERA5, lat={catchment.latitude}, "
          f"lon={catchment.longitude}) ...")
    met = fetch_meteorology(catchment, start, end)
    met.to_csv(meteo_path, index=False)
    print(f"        {len(met):,} daily records -> {meteo_path.name}")

    print("  [3/3] merging on date ...")
    merged = met.merge(flow, on="date", how="inner").sort_values("date")
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_path, index=False)
    print(f"        {len(merged):,} matched days -> {merged_path.name}")
    return merged


def load_cached(catchment: Catchment, cache_dir: Path | None = None) -> pd.DataFrame:
    """Load the merged daily record from disk."""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    df = pd.read_csv(cache_dir / "catchment_daily.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)
