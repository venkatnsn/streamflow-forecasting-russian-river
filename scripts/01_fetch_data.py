"""Acquire 30 years of daily streamflow + meteorology, merge, and cache.

Run once. Subsequent scripts read the cached CSV in data/processed/.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when this script runs from anywhere.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_acquisition import (RUSSIAN_RIVER_HOPLAND, merge_and_cache)
from src.pet import add_pet_column

START = "1995-01-01"
END   = "2024-12-31"

print(f"Fetching {START} -> {END} for {RUSSIAN_RIVER_HOPLAND.name}")
print(f"USGS site {RUSSIAN_RIVER_HOPLAND.site_no}, "
      f"area {RUSSIAN_RIVER_HOPLAND.area_km2:.0f} km^2")
print()

merged = merge_and_cache(RUSSIAN_RIVER_HOPLAND, START, END)
merged = add_pet_column(merged, RUSSIAN_RIVER_HOPLAND.latitude)
merged.to_csv(ROOT / "data" / "processed" / "catchment_daily.csv", index=False)

print()
print("Summary statistics:")
print(merged[["P_mm", "PET_mm", "T_mean_C", "Q_mm"]].describe().round(2)
      .to_string())
print()
print(f"Output: {ROOT / 'data' / 'processed' / 'catchment_daily.csv'}")
