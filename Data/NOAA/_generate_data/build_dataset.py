"""
Build tornado-environment dataset:
- Labels + event metadata from NOAA Storm Events (your CSV rows)
- Environmental features from ERA5 (CDS):
    single-levels: cape, cin, 2m T, 2m Td, surface pressure, tcwv, + 10m winds
    pressure-levels: u/v winds + geopotential -> shear_0_1km, shear_0_3km

Output: one row per tornado event with all features + ef_category/ef_binary.

Requirements:
pip install pandas numpy xarray netCDF4 cdsapi

Before running:
- Configure cdsapi key (~/.cdsapirc)
- Put your tornado rows in a CSV file (e.g., storm_events_tornadoes.csv)

Key change vs your previous script:
- Pressure-level downloads use a SAFE fallback:
    * Try monthly file: era5_pl_YYYY_MM.nc
    * If missing or CDS rejects due to cost, download day-by-day:
        era5_pl_YYYY_MM_DD.nc
  Then open ds_pl from both monthly and daily files.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Tuple, List, Dict

import numpy as np
import pandas as pd
import xarray as xr
import cdsapi


# =============================================================================
# Timezone handling
# =============================================================================
TZ_OFFSETS_HOURS = {
    # Standard
    "EST": -5, "CST": -6, "MST": -7, "PST": -8,
    "AKST": -9, "HST": -10,
    # Daylight
    "EDT": -4, "CDT": -5, "MDT": -6, "PDT": -7,
    "AKDT": -8,
    # Sometimes seen
    "GMT": 0, "UTC": 0,
}

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}


def parse_stormevents_datetime_local(dt_str: str) -> datetime:
    """
    Robust parsing for Storm Events 'DD-MON-YY HH:MM:SS'
    Forces 19xx for YY >= 40 (covers 1950 sample) and 20xx otherwise.
    """
    s = str(dt_str).strip().upper()
    date_part, time_part = s.split()
    dd, mon, yy = date_part.split("-")
    hh, mm, ss = time_part.split(":")

    year2 = int(yy)
    year = 1900 + year2 if year2 >= 40 else 2000 + year2
    month = MONTHS[mon]
    return datetime(year, month, int(dd), int(hh), int(mm), int(ss))


def to_utc_and_round_hour(local_dt: datetime, tz_abbrev: str) -> datetime:
    import re
    """
    Convert local_dt (naive) with Storm Events timezone field to UTC, then round to nearest hour.

    Handles:
      - Abbreviations: CST, CDT, EST, ...
      - Offset forms:  CST-6, CDT-5, MST-7, PST-8, UTC+0, GMT+0, etc.

    Convention used:
      local time = UTC + offset_hours
      so UTC = local - offset_hours

    For US zones:
      CST => -6, CDT => -5, etc. (matches TZ_OFFSETS_HOURS)
      CST-6 means offset_hours = -6
    """
    raw = (tz_abbrev or "").strip().upper()

    if not raw:
        raise ValueError("Empty CZ_TIMEZONE value.")

    # 1) Try explicit offset formats like "CST-6", "UTC+0", "GMT-3"
    m = re.match(r"^[A-Z]{2,5}([+-]\d{1,2})$", raw)
    if m:
        offset_hours = int(m.group(1))  # e.g., "-6"
    else:
        # 2) Plain abbreviation like "CST"
        if raw not in TZ_OFFSETS_HOURS:
            raise ValueError(
                f"Unknown CZ_TIMEZONE value: {raw!r}. "
                f"Add it to TZ_OFFSETS_HOURS or extend parsing."
            )
        offset_hours = TZ_OFFSETS_HOURS[raw]

    # Convert to UTC (UTC = local - offset)
    utc_dt = (local_dt - timedelta(hours=offset_hours)).replace(tzinfo=timezone.utc)

    # Round to nearest hour (>=30 mins rounds up)
    rounded = (utc_dt + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    return rounded



# =============================================================================
# EF/F handling
# =============================================================================
def parse_f_or_ef_scale(scale: str) -> int | None:
    """
    TOR_F_SCALE examples: 'F3', 'EF2', '0' (sometimes), ''.
    Returns integer category 0-5 when possible.
    """
    if scale is None:
        return None
    s = str(scale).strip().upper()
    if not s:
        return None
    if s.startswith("EF"):
        s = s[2:]
    elif s.startswith("F"):
        s = s[1:]
    try:
        return int(float(s))
    except Exception:
        return None


def ef_binary_from_category(cat: int | None) -> int | None:
    if cat is None:
        return None
    return 1 if cat >= 2 else 0


# =============================================================================
# ERA5 helpers
# =============================================================================
def bbox_with_margin(lats: Iterable[float], lons: Iterable[float], margin_deg: float = 2.0) -> List[float]:
    """
    CDS area format: [North, West, South, East]
    """
    lat_list = list(map(float, lats))
    lon_list = list(map(float, lons))

    lat_min = float(np.nanmin(lat_list))
    lat_max = float(np.nanmax(lat_list))
    lon_min = float(np.nanmin(lon_list))
    lon_max = float(np.nanmax(lon_list))

    north = min(90.0, lat_max + margin_deg)
    south = max(-90.0, lat_min - margin_deg)
    west = max(-180.0, lon_min - margin_deg)
    east = min(180.0, lon_max + margin_deg)
    return [north, west, south, east]


def unique_ymd_hours(times_utc: Iterable[datetime]) -> Dict[Tuple[str, str], Dict[str, List[str]]]:
    """
    Return {(year, month): {day: [HH:00, ...]}} for the *event hours only*.

    This is the "not overkill" fix: we don't request all 24 hours.
    Since we round to the nearest hour, each event contributes exactly one hour.
    """
    out: Dict[Tuple[str, str], Dict[str, set]] = {}
    for t in times_utc:
        y = f"{t.year:04d}"
        m = f"{t.month:02d}"
        d = f"{t.day:02d}"
        hh = f"{t.hour:02d}:00"
        out.setdefault((y, m), {}).setdefault(d, set()).add(hh)
    return {k: {d: sorted(list(v)) for d, v in daymap.items()} for k, daymap in out.items()}


# =============================================================================
# Shear computation
# =============================================================================
G = 9.80665  # m/s^2


def nearest_profile_wind_at_height(
    geopotential: np.ndarray,  # shape [levels]
    u: np.ndarray,             # shape [levels]
    v: np.ndarray,             # shape [levels]
    target_m: float
) -> Tuple[float, float]:
    """
    Choose the pressure level whose geopotential height (phi/g) is closest to target_m.
    """
    height_m = geopotential / G
    idx = int(np.nanargmin(np.abs(height_m - target_m)))
    return float(u[idx]), float(v[idx])


def shear_magnitude(u1: float, v1: float, u0: float, v0: float) -> float:
    return float(math.hypot(u1 - u0, v1 - v0))


# =============================================================================
# Main pipeline
# =============================================================================
def main(
    storm_events_csv: str,
    out_csv: str = "tornado_env_dataset.csv",
    cache_dir: str = "era5_cache",
):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"[1] Reading CSV: {storm_events_csv}")
    df = pd.read_csv(storm_events_csv)

    print("[2] Filtering tornado rows + required fields...")
    df = df[df["EVENT_TYPE"].astype(str).str.upper() == "TORNADO"].copy()
    df = df[df["BEGIN_DATE_TIME"].notna() & df["BEGIN_LAT"].notna() & df["BEGIN_LON"].notna()].copy()
    print(f"    tornado rows kept: {len(df)}")

    print("[3] Parsing EF/F labels...")
    df["ef_category"] = df["TOR_F_SCALE"].apply(parse_f_or_ef_scale)
    df["ef_binary"] = df["ef_category"].apply(ef_binary_from_category)

    print("[4] Converting begin times -> UTC (rounded hour)...")
    begin_utc: List[datetime] = []
    for dt_str, tz_abbrev in zip(df["BEGIN_DATE_TIME"], df["CZ_TIMEZONE"]):
        local_dt = parse_stormevents_datetime_local(dt_str)
        begin_utc.append(to_utc_and_round_hour(local_dt, str(tz_abbrev)))

    ERA5_LATEST_UTC = datetime(2025, 12, 10, 23, 0, 0, tzinfo=timezone.utc)
    df["begin_time_utc"] = begin_utc
    before = len(df)
    df = df[df["begin_time_utc"] <= ERA5_LATEST_UTC].copy()
    print(f"    dropped beyond ERA5_LATEST_UTC: {before - len(df)}; remaining: {len(df)}")

    print("[5] Building request footprint (bbox + needed days/hours)...")
    area = bbox_with_margin(df["BEGIN_LAT"].astype(float), df["BEGIN_LON"].astype(float), margin_deg=2.0)
    ym_day_hours = unique_ymd_hours(df["begin_time_utc"])
    print(f"    area (N,W,S,E): {area}")
    print(f"    year-month chunks: {len(ym_day_hours)}")

    c = cdsapi.Client()

    # -------------------------------------------------------------------------
    # Single-level downloads (monthly chunks; event-hours only)
    # -------------------------------------------------------------------------
    print("[6] Ensuring ERA5 single-levels exist...")
    single_files: List[str] = []
    for (year, month), day_hours in sorted(ym_day_hours.items()):
        target = cache_path / f"era5_single_{year}_{month}.nc"
        single_files.append(str(target))

        if target.exists():
            print(f"    cache hit: {target.name}")
            continue

        days = sorted(day_hours.keys())
        hours = sorted({h for hs in day_hours.values() for h in hs})
        print(f"    requesting single {year}-{month}: days={len(days)}, hours={len(hours)} -> {target.name}")

        req = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "surface_pressure",
                "total_column_water_vapour",
                "convective_available_potential_energy",
                "convective_inhibition",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": year,
            "month": month,
            "day": days,
            "time": hours,      # event-hours only
            "area": area,
        }
        c.retrieve("reanalysis-era5-single-levels", req, str(target))

    # -------------------------------------------------------------------------
    # Pressure-level downloads:
    # - prefer monthly file era5_pl_YYYY_MM.nc if it exists
    # - if missing, fetch day-by-day files era5_pl_YYYY_MM_DD.nc
    # This prevents CDS "cost limits exceeded" on monthly PL requests.
    # -------------------------------------------------------------------------
    print("[7] Ensuring ERA5 pressure-levels exist (monthly if possible, else daily fallback)...")

    pl_files: List[str] = []
    pressure_levels = ["1000", "975", "950", "925", "900", "875", "850", "800", "750", "700"]

    for (year, month), day_hours in sorted(ym_day_hours.items()):
        monthly_target = cache_path / f"era5_pl_{year}_{month}.nc"

        if monthly_target.exists():
            # Great: use the monthly file you already downloaded
            pl_files.append(str(monthly_target))
            print(f"    cache hit (monthly): {monthly_target.name}")
            continue

        # Monthly file missing -> fallback to daily downloads (only the needed days and hours)
        print(f"    missing monthly {monthly_target.name} -> downloading daily PL files for {year}-{month}")
        for day, hours in sorted(day_hours.items()):
            daily_target = cache_path / f"era5_pl_{year}_{month}_{day}.nc"
            pl_files.append(str(daily_target))

            if daily_target.exists():
                # already downloaded in a previous run
                continue

            print(f"      requesting PL {year}-{month}-{day}: hours={len(hours)} -> {daily_target.name}")
            req = {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "geopotential",
                ],
                "pressure_level": pressure_levels,
                "year": year,
                "month": month,
                "day": [day],       # single-day request
                "time": hours,      # event-hours only
                "area": area,       # keep same bbox; if you still hit cost, we shrink this next
            }
            c.retrieve("reanalysis-era5-pressure-levels", req, str(daily_target))

    # -------------------------------------------------------------------------
    # Open datasets
    # -------------------------------------------------------------------------
    print("[8] Opening NetCDFs with xarray...")
    ds_single = xr.open_mfdataset(single_files, combine="by_coords")
    ds_pl = xr.open_mfdataset(pl_files, combine="by_coords")
    
    # --- Normalize ERA5 time coordinate naming (CDS sometimes uses valid_time) ---
    if "valid_time" in ds_single.coords and "time" not in ds_single.coords:
        ds_single = ds_single.rename({"valid_time": "time"})
    if "valid_time" in ds_pl.coords and "time" not in ds_pl.coords:
        ds_pl = ds_pl.rename({"valid_time": "time"})


    # -------------------------------------------------------------------------
    # Build final rows
    # -------------------------------------------------------------------------
    print("[9] Extracting features per tornado...")
    out_rows: List[dict] = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        lat = float(row["BEGIN_LAT"])
        lon = float(row["BEGIN_LON"])
        t: datetime = row["begin_time_utc"]

        if i == 1 or i % 25 == 0 or i == len(df):
            print(f"    event {i}/{len(df)} time={t.isoformat()} lat={lat:.3f} lon={lon:.3f}")

        # Sample single-level vars nearest in time/space
        s = ds_single.sel(time=np.datetime64(t), latitude=lat, longitude=lon, method="nearest")

        cape = float(s["cape"].values)
        cin = float(s["cin"].values)
        t2m = float(s["t2m"].values)
        d2m = float(s["d2m"].values)
        sp = float(s["sp"].values)
        tcwv = float(s["tcwv"].values)
        u10 = float(s["u10"].values)
        v10 = float(s["v10"].values)

        # Sample pressure-level profile nearest in time/space
        p = ds_pl.sel(time=np.datetime64(t), latitude=lat, longitude=lon, method="nearest")

        geop = np.asarray(p["z"].values)
        u = np.asarray(p["u"].values)
        v = np.asarray(p["v"].values)

        # Wind closest to 1km and 3km (via geopotential height)
        u1, v1 = nearest_profile_wind_at_height(geop, u, v, target_m=1000.0)
        u3, v3 = nearest_profile_wind_at_height(geop, u, v, target_m=3000.0)

        shear_0_1km = shear_magnitude(u1, v1, u10, v10)
        shear_0_3km = shear_magnitude(u3, v3, u10, v10)

        out_rows.append({
            "event_id": row.get("EVENT_ID", None),
            "begin_time_utc": t.isoformat(),
            "begin_lat": lat,
            "begin_lon": lon,
            "cape": cape,
            "cin": cin,
            "temp_2m": t2m,
            "dewpoint_2m": d2m,
            "surface_pressure": sp,
            "tcwv": tcwv,
            "shear_0_1km": shear_0_1km,
            "shear_0_3km": shear_0_3km,
            "ef_category": row["ef_category"],
            "ef_binary": row["ef_binary"],
            "tor_f_scale_raw": row.get("TOR_F_SCALE", None),
            "cz_timezone": row.get("CZ_TIMEZONE", None),
            "begin_date_time_raw": row.get("BEGIN_DATE_TIME", None),
        })

    print("[10] Saving CSV...")
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"Cached ERA5 downloads in: {cache_dir}")


if __name__ == "__main__":
    # Put your NOAA Storm Events tornado rows in this file:
    
    files_to_load = ['StormEvents_details-ftp_v1.0_d1950_c20250520.csv',
'StormEvents_details-ftp_v1.0_d2011_c20250520.csv',
'StormEvents_details-ftp_v1.0_d2012_c20250520.csv',
'StormEvents_details-ftp_v1.0_d2013_c20250520.csv',
'StormEvents_details-ftp_v1.0_d2014_c20250520.csv',
'StormEvents_details-ftp_v1.0_d2016_c20250818.csv',
'StormEvents_details-ftp_v1.0_d2017_c20250520.csv',
'StormEvents_details-ftp_v1.0_d2018_c20250520.csv',
'StormEvents_details-ftp_v1.0_d2020_c20251118.csv',
'StormEvents_details-ftp_v1.0_d2021_c20250520.csv',
'StormEvents_details-ftp_v1.0_d2022_c20250721.csv',
'StormEvents_details-ftp_v1.0_d2023_c20251216.csv',
'StormEvents_details-ftp_v1.0_d2024_c20251204.csv',
'StormEvents_details-ftp_v1.0_d2025_c20251216.csv']
    for f, filename in enumerate(files_to_load):
        main(storm_events_csv=f"./{filename}", out_csv=f'tornado_env_dataset{f}.csv')
