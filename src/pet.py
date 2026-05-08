"""Reference evapotranspiration (PET) using the Hargreaves-Samani method.

Hargreaves & Samani (1985) is the de-facto standard when only T_max,
T_min, and date are available — it does not require humidity or wind.

Formula (FAO-56 reference):

    PET = 0.0023 * Ra * (T_mean + 17.8) * sqrt(T_max - T_min)

where ``Ra`` is the extra-terrestrial radiation in mm/day equivalent
(computed analytically from latitude and day-of-year).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def extraterrestrial_radiation_mm(latitude_deg: float, doy: np.ndarray) -> np.ndarray:
    """Daily extra-terrestrial radiation (mm/day equivalent) following FAO-56."""
    phi = np.radians(latitude_deg)
    doy = np.asarray(doy, dtype=float)

    # Inverse relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)

    # Solar declination
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)

    # Sunset hour angle, with safe clipping
    arg = -np.tan(phi) * np.tan(delta)
    arg = np.clip(arg, -1.0, 1.0)
    omega = np.arccos(arg)

    # Ra in MJ/m^2/day, then convert to mm/day equivalent
    Gsc = 0.0820   # MJ/m^2/min, solar constant
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        omega * np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.sin(omega)
    )
    # 1 mm of water requires 2.45 MJ/m^2 to evaporate
    return Ra / 2.45


def hargreaves_pet(t_max: np.ndarray, t_min: np.ndarray,
                   doy: np.ndarray, latitude_deg: float) -> np.ndarray:
    """Daily Hargreaves PET (mm/day)."""
    Ra_mm = extraterrestrial_radiation_mm(latitude_deg, doy)
    t_mean = (t_max + t_min) / 2.0
    delta_t = np.clip(t_max - t_min, 0.0, None)   # protect sqrt
    return 0.0023 * Ra_mm * (t_mean + 17.8) * np.sqrt(delta_t)


def add_pet_column(df: pd.DataFrame, latitude_deg: float) -> pd.DataFrame:
    """Append a ``PET_mm`` column to a daily forcing frame in place-ish."""
    df = df.copy()
    doy = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy()
    df["PET_mm"] = hargreaves_pet(df["T_max_C"].to_numpy(),
                                   df["T_min_C"].to_numpy(),
                                   doy, latitude_deg)
    return df
