"""Real-time daily data for any U.S. catchment, free and no-auth."""
import requests, pandas as pd
from dataretrieval import nwis

CFS_TO_M3S = 0.0283168
SECONDS_PER_DAY = 86400.0

def fetch_streamflow(site_no, start, end, area_km2):
    """USGS NWIS daily mean discharge -> mm/day (depth equivalent)."""
    df, _ = nwis.get_dv(sites=site_no, parameterCd="00060", start=start, end=end)
    df = df.rename(columns={"00060_Mean": "Q_cfs"})[["Q_cfs"]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    area_m2 = area_km2 * 1e6
    df["Q_mm"] = df["Q_cfs"] * CFS_TO_M3S * SECONDS_PER_DAY / area_m2 * 1000.0
    return df.reset_index().rename(columns={"index": "date"})

def fetch_meteorology(latitude, longitude, start, end, timezone="UTC"):
    """Open-Meteo Historical archive: ERA5 daily precip + Tmax + Tmin."""
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": latitude, "longitude": longitude,
        "start_date": start, "end_date": end,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": timezone,
    }, timeout=120)
    r.raise_for_status()
    data = r.json()["daily"]
    df = pd.DataFrame(data).rename(columns={
        "time": "date", "precipitation_sum": "P_mm",
        "temperature_2m_max": "T_max_C", "temperature_2m_min": "T_min_C",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df

# Russian River near Hopland, CA  (USGS 11462500, 937 km^2)
flow = fetch_streamflow("11462500", "1995-01-01", "2024-12-31", area_km2=937.0)
met  = fetch_meteorology(38.992, -123.130, "1995-01-01", "2024-12-31",
                           timezone="America/Los_Angeles")
df = met.merge(flow, on="date").sort_values("date").reset_index(drop=True)
print(f"{len(df):,} matched daily records.")
