"""
Build hourly weather inputs from ERA5 reanalysis (Open-Meteo archive API).

Why: the weather files originally bundled with METR-LA / PEMS-BAY were stamped
in UTC while traffic is local wall-clock time, PEMS-BAY weather stopped a month
before the traffic data ends (~85% of the test period had no weather), hourly
values were forward-filled for up to 72 h, and METR-LA precipitation never
dropped below 0.01 with no rain labels at all.

This module fetches hourly weather for a dataset's own period directly in the
traffic timezone (DST handled by the API), writes
``weather_<dataset>_era5_local.csv`` with the column names the data pipeline
expects, and prints coverage, the daily temperature cycle and (for the two
benchmarks) agreement with the old file once that file is shifted out of UTC.

Open-Meteo's free API is intended for non-commercial use; see
https://open-meteo.com/en/terms before using it commercially.
"""

import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# One representative point per sensor network (city-wide weather, broadcast to all
# sensors by the model). METR-LA uses the centroid of its own sensor coordinates.
SITES: dict[str, dict] = {
    "PEMS-BAY": {"lat": 37.33, "lon": -121.93, "tz": "America/Los_Angeles",
                 "label": "Santa Clara Valley (PEMS-BAY network centre)"},
    "METR-LA": {"lat": 34.15, "lon": -118.35, "tz": "America/Los_Angeles",
                "label": "Los Angeles (METR-LA sensor centroid)"},
}
OLD_FILES = {"PEMS-BAY": "clean_weather_data_pems_bay.csv", "METR-LA": "clean_weather_data_metr_la.csv"}

HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
          "rain", "snowfall", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
          "wind_direction_10m", "surface_pressure", "weather_code"]

WMO = {0: "Fair", 1: "Mostly Clear", 2: "Partly Cloudy", 3: "Cloudy", 45: "Fog", 48: "Freezing Fog",
       51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle", 56: "Light Freezing Drizzle",
       57: "Freezing Drizzle", 61: "Light Rain", 63: "Rain", 65: "Heavy Rain",
       66: "Light Freezing Rain", 67: "Freezing Rain", 71: "Light Snow", 73: "Snow", 75: "Heavy Snow",
       77: "Snow Grains", 80: "Light Rain Showers", 81: "Rain Showers", 82: "Heavy Rain Showers",
       85: "Light Snow Showers", 86: "Snow Showers", 95: "Thunderstorm",
       96: "Thunderstorm with Hail", 99: "Thunderstorm with Heavy Hail"}
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def sky_code(pct) -> str:
    if pct is None or np.isnan(pct):
        return "VV"
    for limit, code in ((5, "CLR"), (25, "FEW"), (50, "SCT"), (87, "BKN")):
        if pct < limit:
            return code
    return "OVC"


def compass(deg, speed) -> str:
    if speed is not None and speed < 2:
        return "CALM"
    if deg is None or np.isnan(deg):
        return "VRB"
    return COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def fetch(lat: float, lon: float, start: str, end: str, tz: str):
    """Hourly ERA5 weather for ``[start, end]`` in local time ``tz``."""
    q = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
         "hourly": ",".join(HOURLY), "timezone": tz}
    with urllib.request.urlopen(ARCHIVE + "?" + urllib.parse.urlencode(q), timeout=180) as r:
        d = json.load(r)
    h = pd.DataFrame(d["hourly"])
    h["time"] = pd.to_datetime(h["time"])          # already local wall-clock
    return h, d


def to_pipeline_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename API columns to the names the data pipeline reads."""
    out = pd.DataFrame({
        "datetime": raw["time"],
        "temperature": raw["temperature_2m"],
        "dew_point": raw["dew_point_2m"],
        "relative_humidity": raw["relative_humidity_2m"],
        "hourly_precipitation": raw["precipitation"],
        "rain": raw["rain"],
        "snowfall": raw["snowfall"],
        "wind_speed": raw["wind_speed_10m"],
        "wind_gust": raw["wind_gusts_10m"],
        "wind_direction_deg": raw["wind_direction_10m"],
        "cloud_cover_pct": raw["cloud_cover"],
        "surface_pressure": raw["surface_pressure"],
        "weather_code": raw["weather_code"],
    })
    out["weather_condition"] = [WMO.get(int(c), "Unknown") if pd.notna(c) else "Unknown"
                                for c in raw["weather_code"]]
    out["cloud_cover"] = [sky_code(p) for p in raw["cloud_cover"]]
    out["wind_direction"] = [compass(d, s) for d, s in zip(raw["wind_direction_10m"], raw["wind_speed_10m"],
                                                           strict=True)]
    return out


def build_weather(dataset: str, data_dir: str, out_dir: str | None = None,
                  lat: float | None = None, lon: float | None = None,
                  tz: str | None = None) -> str:
    """Write ``<out_dir>/weather_<dataset>_era5_local.csv`` covering ``<data_dir>/<dataset>.csv``.

    METR-LA and PEMS-BAY have built-in coordinates; any other dataset needs
    ``lat``, ``lon`` and the IANA timezone ``tz`` its timestamps are in.
    """
    out_dir = out_dir or data_dir
    site = dict(SITES.get(dataset, {}))
    if lat is not None and lon is not None:
        site.update(lat=lat, lon=lon, label="user-supplied point")
    if tz:
        site["tz"] = tz
    if not {"lat", "lon", "tz"} <= site.keys():
        raise ValueError(f"no built-in location for {dataset}: pass --lat, --lon and --tz")

    traffic = pd.read_csv(os.path.join(data_dir, f"{dataset}.csv"), index_col=0, usecols=[0])
    ts = pd.to_datetime(traffic.index)
    start, end = ts.min().date(), ts.max().date()
    print(f"\n=== {dataset}: traffic {start} -> {end} ({len(ts):,} timestamps), "
          f"weather at {site['lat']}, {site['lon']} ({site.get('label', '')})")

    raw, meta = fetch(site["lat"], site["lon"], str(start), str(end + pd.Timedelta(days=1)), site["tz"])
    print(f"    fetched {len(raw):,} hourly rows | grid point {meta['latitude']:.2f}, "
          f"{meta['longitude']:.2f} | elevation {meta['elevation']} m | {meta['timezone_abbreviation']}")

    out = to_pipeline_frame(raw)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"weather_{dataset}_era5_local.csv")
    out.to_csv(path, index=False)

    # ---- validation -------------------------------------------------------
    w = out.set_index("datetime")
    nan_cols = {c: int(w[c].isna().sum()) for c in w.columns if w[c].dtype.kind in "fi" and w[c].isna().any()}
    # causal coverage: a reading at or before each traffic timestamp, within 90 min
    idx = w.index.get_indexer(ts, method="ffill")
    gap = (ts - w.index[idx]).total_seconds() / 60
    covered = float(((idx >= 0) & (gap <= 90)).mean())
    prof = w["temperature"].groupby(w.index.hour).mean()
    rain_h = float((w["hourly_precipitation"] > 0).mean())
    print(f"    coverage of traffic period: {covered:.1%} | missing numeric values: {nan_cols or 'none'}")
    print(f"    daily cycle: warmest hour {int(prof.idxmax())}:00 local, coldest {int(prof.idxmin())}:00 | "
          f"range {prof.min():.1f}-{prof.max():.1f} C")
    print(f"    precipitation: {rain_h:.1%} of hours wet, {w['hourly_precipitation'].sum():.0f} mm total")

    # ---- cross-check against the old benchmark file -------------------------
    old_path = os.path.join(data_dir, OLD_FILES.get(dataset, ""))
    if dataset in OLD_FILES and os.path.exists(old_path):
        old = pd.read_csv(old_path, parse_dates=["datetime"]).set_index("datetime")
        old_h = old["temperature"].resample("1h").mean()
        best = None
        for shift in range(-12, 13):
            s = old_h.copy()
            s.index = s.index + pd.Timedelta(hours=shift)
            j = pd.concat([w["temperature"], s], axis=1, join="inner").dropna()
            if len(j) > 500:
                r = float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])
                if best is None or r > best[1]:
                    best = (shift, r)
        if best is not None:
            print(f"    old file agrees best when shifted {best[0]:+d} h (r={best[1]:.3f})")
    print(f"    -> {path}")
    return path
