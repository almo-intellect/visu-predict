"""
Benchmark-protocol data pipeline for node-level spatio-temporal models.

Differences from ``data_module.load_and_prepare_data`` (the legacy pipeline)
and why they matter:

* Targets stay RAW. Zeros in METR-LA / PEMS-BAY are sensor faults; they are
  kept in the targets and masked out of loss and metrics (DCRNN protocol),
  instead of being overwritten with the sensor mean, which made the legacy test
  metrics incomparable with published results.
* Windows follow the DCRNN convention (x = 12 past steps, y = next 12 steps)
  and are split 70/10/20 by window index, so the test windows are the same
  ones used in the literature.
* Inputs are z-scored with statistics of the training portion only.
* Calendar information is given as integer indices at the data resolution:
  time-of-day slot (288 per day for 5-minute data) and day-of-week (optionally
  "holiday" as an 8th day type), consumed by embedding tables in the model.
  The legacy features (hour/23, week-of-year, month) collapsed 12 slots per hour
  into one value and took values in the test months never seen in training.
* Weather (optional) is converted from its source timezone to the traffic
  timezone, matched causally (latest observation at or before t), z-scored on
  the training period, and paired with an availability flag, so periods with no
  weather data are explicit instead of silently repeating the last row.

Batches are produced by :class:`WindowBatcher`, which keeps the whole series on
the target device and gathers windows with index arithmetic (no DataLoader
workers, no per-sample Python overhead).
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .spatial import load_adjacency_matrix

try:  # optional
    import holidays as _holidays

    HOLIDAYS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _holidays = None
    HOLIDAYS_AVAILABLE = False


# Weather metadata per dataset. ``weather_file`` is the rebuilt hourly series
# (local wall-clock time, covering the whole traffic period; see fix_weather.py).
# ``weather_file_legacy`` is the original export, kept only as a fallback: it is
# stamped in UTC (daily temperature peaks at ~22:00 / ~20:00 "local" otherwise),
# PEMS-BAY stops a month before the traffic data ends, hourly values were
# forward-filled for up to 72 h and METR-LA precipitation never drops below 0.01.
DATASET_META: Dict[str, Dict[str, str]] = {
    "PEMS-BAY": {"traffic_tz": "US/Pacific", "weather_tz": "US/Pacific",
                 "weather_file": "weather_PEMS-BAY_era5_local.csv",
                 "weather_file_legacy": "clean_weather_data_pems_bay.csv",
                 "legacy_weather_tz": "UTC", "country": "US"},
    "METR-LA": {"traffic_tz": "US/Pacific", "weather_tz": "US/Pacific",
                "weather_file": "weather_METR-LA_era5_local.csv",
                "weather_file_legacy": "clean_weather_data_metr_la.csv",
                "legacy_weather_tz": "UTC", "country": "US"},
}

# Columns used as numeric weather inputs when present (the rebuilt files add
# cloud cover, pressure and gusts; the legacy files carry visibility instead).
WEATHER_NUMERIC = [
    "temperature", "hourly_precipitation", "visibility",
    "wind_speed", "wind_gust", "relative_humidity", "dew_point",
    "cloud_cover_pct", "surface_pressure",
]


# =============================================================================
# Scaler
# =============================================================================

class ZScoreScaler:
    """Global z-score scaler (single mean/std, as in the benchmark lineage)."""

    def __init__(self, mean: float, std: float) -> None:
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}


# =============================================================================
# Loading helpers
# =============================================================================

def load_traffic_frame(input_dir: str, dataset_name: str) -> pd.DataFrame:
    """Read ``<input_dir>/<dataset_name>.csv`` (index = timestamps)."""
    path = os.path.join(input_dir, f"{dataset_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.astype(np.float32)


def calendar_indices(
    timestamps: pd.DatetimeIndex,
    steps_per_day: int,
    holiday_country: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time-of-day slot, day-of-week, holiday flag) per timestamp."""
    ts = pd.DatetimeIndex(timestamps)
    minutes = ts.hour * 60 + ts.minute
    slot_minutes = 1440 // steps_per_day
    tod = (np.asarray(minutes) // slot_minutes).astype(np.int64)
    dow = np.asarray(ts.dayofweek, dtype=np.int64)
    hol = np.zeros(len(ts), dtype=np.int64)
    if holiday_country:
        if HOLIDAYS_AVAILABLE:
            years = sorted(set(ts.year))
            cal = _holidays.country_holidays(holiday_country, years=years)
            dates = np.array(ts.date)
            hol = np.array([d in cal for d in dates], dtype=np.int64)
        else:
            warnings.warn("holidays package missing; holiday flag disabled")
    return tod, dow, hol


def dcrnn_window_splits(
    num_timesteps: int,
    in_steps: int,
    out_steps: int,
    ratios: Sequence[float] = (0.7, 0.1, 0.2),
) -> Dict[str, np.ndarray]:
    """Window start indices split like DCRNN's ``generate_training_data.py``.

    Window ``s`` uses x = [s, s+in) and y = [s+in, s+in+out).
    """
    num_samples = num_timesteps - in_steps - out_steps + 1
    num_test = round(num_samples * ratios[2])
    num_train = round(num_samples * ratios[0])
    num_val = num_samples - num_test - num_train
    starts = np.arange(num_samples, dtype=np.int64)
    return {
        "train": starts[:num_train],
        "val": starts[num_train:num_train + num_val],
        "test": starts[num_train + num_val:],
    }


def load_weather_features(
    path: str,
    timestamps: pd.DatetimeIndex,
    fit_end: int,
    traffic_tz: str = "US/Pacific",
    weather_tz: str = "UTC",
    tolerance: str = "90min",
) -> Tuple[np.ndarray, List[str]]:
    """City-level weather aligned to traffic timestamps.

    Returns an array ``(T, E)`` of z-scored features plus an availability flag
    (last column) and the list of feature names. Matching is causal
    (latest observation at or before each traffic timestamp).
    """
    w = pd.read_csv(path)
    time_col = "datetime" if "datetime" in w.columns else w.columns[0]
    wt = pd.to_datetime(w[time_col])
    if weather_tz and traffic_tz and weather_tz != traffic_tz:
        wt = wt.dt.tz_localize(weather_tz).dt.tz_convert(traffic_tz).dt.tz_localize(None)
    w = w.assign(_t=wt.values).drop(columns=[time_col]).sort_values("_t")

    feats = pd.DataFrame({"_t": w["_t"].values})
    names: List[str] = []
    for col in WEATHER_NUMERIC:
        if col in w.columns:
            v = pd.to_numeric(w[col], errors="coerce").astype(float)
            if col == "hourly_precipitation":
                v = np.log1p(v.clip(lower=0))
            feats[col] = v.values
            names.append(col)
    if "weather_condition" in w.columns:
        cond = w["weather_condition"].astype(str).str.lower()
        flags = {
            "is_rain": cond.str.contains("rain|drizzle|shower|storm|thunder"),
            "is_fog": cond.str.contains("fog|haze|mist|smoke"),
            "is_cloudy": cond.str.contains("cloud|overcast"),
        }
        for k, v in flags.items():
            feats[k] = v.astype(float).values
            names.append(k)

    left = pd.DataFrame({"_t": pd.DatetimeIndex(timestamps).values})
    merged = pd.merge_asof(
        left, feats, on="_t", direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    arr = merged[names].to_numpy(dtype=np.float64)
    available = ~np.isnan(arr).any(axis=1)

    # z-score each column with training-period statistics
    fit = arr[:fit_end][available[:fit_end]]
    mu = np.nanmean(fit, axis=0)
    sd = np.nanstd(fit, axis=0)
    sd[sd < 1e-6] = 1.0
    arr = (arr - mu) / sd
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.concatenate([arr, available[:, None].astype(np.float64)], axis=1)
    names = names + ["weather_available"]
    return arr.astype(np.float32), names


# =============================================================================
# Batching
# =============================================================================

class WindowBatcher:
    """Yields batches of windows gathered on-device.

    Each batch is a dict:
      ``x``   (B, in, N, 1 + L)  z-scored traffic (+ L history-lag channels)
      ``tod`` (B, in)            time-of-day slot (long)
      ``dow`` (B, in)            day type (long; 7 = holiday if enabled)
      ``exo`` (B, in, E)         exogenous features (only if present)
      ``y``   (B, out, N)        raw targets (zeros = missing)

    ``history_lags`` (e.g. ``(288, 2016)`` = one day / one week at 5-min
    resolution) add, for input token ``i``, the observation ``lag`` steps
    before the *target* step ``i`` - i.e. what happened at the same time of
    the forecast period yesterday / last week. Only past data is used
    (lag >= out_steps), so there is no leakage.
    """

    def __init__(
        self,
        tensors: Dict[str, torch.Tensor],
        starts: np.ndarray,
        in_steps: int,
        out_steps: int,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        history_lags: Sequence[int] = (),
    ) -> None:
        self.t = tensors
        self.device = tensors["x"].device
        self.starts = torch.as_tensor(starts, dtype=torch.long, device=self.device)
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self._in_off = torch.arange(in_steps, device=self.device)
        self._out_off = torch.arange(in_steps, in_steps + out_steps, device=self.device)
        self.history_lags = tuple(int(lag) for lag in history_lags)
        if self.history_lags:
            if in_steps != out_steps:
                raise ValueError("history lags are aligned to target steps; need in_steps == out_steps")
            if min(self.history_lags) < in_steps + out_steps:
                raise ValueError("history lags must be >= in_steps + out_steps to use past data only")
        self._lag_offs = [self._out_off - lag for lag in self.history_lags]

    def __len__(self) -> int:
        n = len(self.starts)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    @property
    def num_samples(self) -> int:
        return len(self.starts)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        n = len(self.starts)
        if self.shuffle:
            order = torch.randperm(n, generator=self.generator).to(self.device)
        else:
            order = torch.arange(n, device=self.device)
        for b in range(len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            s = self.starts[idx]
            xi = s[:, None] + self._in_off          # (B, in)
            yi = s[:, None] + self._out_off         # (B, out)
            x = self.t["x"][xi]                     # (B, in, N, 1)
            if self._lag_offs:
                lags = [self.t["x"][s[:, None] + off] for off in self._lag_offs]
                x = torch.cat([x] + lags, dim=-1)   # (B, in, N, 1 + L)
            batch = {
                "x": x,
                "tod": self.t["tod"][xi],
                "dow": self.t["dow"][xi],
                "y": self.t["y"][yi],               # (B, out, N)
            }
            if "exo" in self.t:
                batch["exo"] = self.t["exo"][xi]
            yield batch


# =============================================================================
# Bundle
# =============================================================================

@dataclass
class STDataBundle:
    train: WindowBatcher
    val: WindowBatcher
    test: WindowBatcher
    scaler: ZScoreScaler
    num_nodes: int
    steps_per_day: int
    num_day_types: int
    adj: Optional[np.ndarray]
    exo_dim: int
    input_dim: int = 1
    exo_names: List[str] = field(default_factory=list)
    timestamps: Optional[pd.DatetimeIndex] = None
    splits: Dict[str, np.ndarray] = field(default_factory=dict)
    info: Dict[str, object] = field(default_factory=dict)


def find_adjacency_file(input_dir: str, dataset_name: str) -> Optional[str]:
    candidates = [
        f"adj_{dataset_name}.pkl",
        f"adj_mx_{dataset_name.lower().replace('-', '_')}.pkl",
        f"adj_{dataset_name.replace('-', '')}.pkl",
    ]
    for c in candidates:
        p = os.path.join(input_dir, c)
        if os.path.exists(p):
            return p
    return None


def load_st_benchmark(
    input_dir: str,
    dataset_name: str,
    batch_size: int = 16,
    in_steps: int = 12,
    out_steps: int = 12,
    steps_per_day: Optional[int] = None,
    use_weather: bool = False,
    weather_file: Optional[str] = None,
    use_holidays: bool = False,
    holiday_country: Optional[str] = None,
    device: str = "cpu",
    seed: int = 42,
    eval_batch_size: Optional[int] = None,
    split_ratios: Sequence[float] = (0.7, 0.1, 0.2),
    history_lags: Sequence[int] = (),
) -> STDataBundle:
    """Load a dataset under the benchmark protocol and build batchers.

    With ``history_lags`` the earliest training windows (those without a full
    lag history) are dropped; validation and test windows are unchanged, so
    test metrics stay comparable with the literature.
    """
    df = load_traffic_frame(input_dir, dataset_name)
    values = df.to_numpy(dtype=np.float32)            # (T, N) raw, zeros kept
    timestamps = pd.DatetimeIndex(df.index)
    T, N = values.shape

    if steps_per_day is None:
        step = pd.Series(timestamps).diff().dropna().mode().iloc[0]
        steps_per_day = int(pd.Timedelta("1D") / step)

    meta = DATASET_META.get(dataset_name, {})
    country = (holiday_country or meta.get("country")) if use_holidays else None
    tod, dow, hol = calendar_indices(timestamps, steps_per_day, country)
    num_day_types = 7
    if use_holidays:
        dow = np.where(hol > 0, 7, dow)
        num_day_types = 8

    splits = dcrnn_window_splits(T, in_steps, out_steps, split_ratios)
    fit_end = int(splits["train"][-1]) + in_steps     # raw steps seen as train inputs
    train_vals = values[:fit_end]
    scaler = ZScoreScaler(train_vals.mean(), train_vals.std())

    x = scaler.transform(values)[..., None].astype(np.float32)   # (T, N, 1)

    tensors = {
        "x": torch.from_numpy(x),
        "tod": torch.from_numpy(tod),
        "dow": torch.from_numpy(dow),
        "y": torch.from_numpy(values),
    }

    exo_names: List[str] = []
    if use_weather:
        wf = weather_file or meta.get("weather_file")
        wpath = wf if (wf and os.path.isabs(wf)) else os.path.join(input_dir, wf or "")
        wtz = meta.get("weather_tz", "UTC")
        if not (wf and os.path.exists(wpath)):
            # fall back to the original export (UTC stamps, partial coverage)
            legacy = meta.get("weather_file_legacy")
            if legacy and os.path.exists(os.path.join(input_dir, legacy)):
                warnings.warn(f"{wf} not found; falling back to {legacy}, which is stamped in "
                              "UTC and does not cover the whole period")
                wpath, wtz = os.path.join(input_dir, legacy), meta.get("legacy_weather_tz", "UTC")
        if wf and os.path.exists(wpath):
            exo, exo_names = load_weather_features(
                wpath, timestamps, fit_end,
                traffic_tz=meta.get("traffic_tz", "US/Pacific"),
                weather_tz=wtz,
            )
            tensors["exo"] = torch.from_numpy(exo)
            cover = float(exo[:, -1].mean())
            test_cover = float(exo[splits["test"][0]:, -1].mean())
            print(f"Weather: {len(exo_names)} features, coverage {cover:.1%} "
                  f"(test period {test_cover:.1%})")
        else:
            warnings.warn(f"weather file not found ({wpath}); continuing without weather")

    tensors = {k: v.to(device) for k, v in tensors.items()}

    adj = None
    adj_path = find_adjacency_file(input_dir, dataset_name)
    if adj_path is not None:
        a, _, _ = load_adjacency_matrix(adj_path, fallback_size=N)
        if a.shape == (N, N) and not np.allclose(a, np.eye(N)):
            adj = a.astype(np.float32)
        else:
            warnings.warn(f"adjacency at {adj_path} unusable (shape {a.shape})")

    history_lags = tuple(int(lag) for lag in history_lags)
    batch_splits = dict(splits)
    if history_lags:
        min_start = max(history_lags) - in_steps
        batch_splits = {k: v[v >= min_start] for k, v in splits.items()}
        if len(batch_splits["val"]) != len(splits["val"]) or len(batch_splits["test"]) != len(splits["test"]):
            raise ValueError("history lags longer than the training period would drop val/test windows")

    ebs = eval_batch_size or max(batch_size, 64)
    # training drops the ragged last batch (PEMS-BAY: a single sample), which
    # also keeps shapes static for torch.compile
    mk = lambda name, bs, shuffle: WindowBatcher(  # noqa: E731
        tensors, batch_splits[name], in_steps, out_steps, bs, shuffle=shuffle,
        drop_last=shuffle, seed=seed, history_lags=history_lags)
    bundle = STDataBundle(
        train=mk("train", batch_size, True),
        val=mk("val", ebs, False),
        test=mk("test", ebs, False),
        scaler=scaler,
        num_nodes=N,
        steps_per_day=steps_per_day,
        num_day_types=num_day_types,
        adj=adj,
        exo_dim=int(tensors["exo"].shape[-1]) if "exo" in tensors else 0,
        input_dim=1 + len(history_lags),
        exo_names=exo_names,
        timestamps=timestamps,
        splits=batch_splits,
        info={
            "dataset": dataset_name, "timesteps": T, "nodes": N,
            "zero_fraction": float((values == 0).mean()),
            "train_windows": len(batch_splits["train"]), "val_windows": len(batch_splits["val"]),
            "test_windows": len(batch_splits["test"]),
            "test_start": str(timestamps[splits["test"][0] + in_steps]),
            "history_lags": list(history_lags),
            "scaler": scaler.state_dict(),
        },
    )
    return bundle
