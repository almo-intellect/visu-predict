"""Traffic dataset and preprocessing pipeline."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import Dataset

from visu_predict.config import TrainingConfig
from visu_predict.features.weather import WeatherIntegration

logger = logging.getLogger(__name__)

try:
    import holidays as _holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    _holidays = None
    HOLIDAYS_AVAILABLE = False


def _make_scaler(kind: str):
    return {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler(),
    }[kind]


def prepare_data(
    df: pd.DataFrame, config: TrainingConfig,
) -> tuple[np.ndarray, pd.DatetimeIndex, Any, int]:
    """Apply missing-value handling and scaling. Returns (data, timestamps, scaler, num_features)."""
    if config.missing_value_strategy == "ffill_bfill":
        df = df.replace(0.0, np.nan).ffill().bfill()
    elif config.missing_value_strategy == "zero":
        df = df.fillna(0.0)
    elif config.missing_value_strategy == "mean":
        df = df.replace(0.0, np.nan).fillna(df.mean(numeric_only=True))
    elif config.missing_value_strategy == "median":
        df = df.replace(0.0, np.nan).fillna(df.median(numeric_only=True))
    elif config.missing_value_strategy == "interpolate":
        df = df.replace(0.0, np.nan).interpolate(method="time").ffill().bfill()

    timestamps = df.index
    sensor_data = df.to_numpy(dtype=np.float32)
    num_features = sensor_data.shape[1]

    scaler = _make_scaler(config.data_scaler_type)
    data_normalized = scaler.fit_transform(sensor_data).astype(np.float32)
    return data_normalized, timestamps, scaler, num_features


def load_traffic_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a traffic CSV with timestamp index parsed as datetime."""
    path = Path(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _create_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    t = pd.to_datetime(timestamps)
    return np.stack(
        [
            t.hour.values / 23.0,
            t.dayofweek.values / 6.0,
            (t.dayofyear // 7).values / 51.0,
            t.month.values / 11.0,
        ],
        axis=1,
    ).astype(np.float32)


def _create_holiday_feature(timestamps: pd.DatetimeIndex, country_code: str) -> np.ndarray:
    if not HOLIDAYS_AVAILABLE:
        warnings.warn("holidays package not installed; holiday feature is zeros.", stacklevel=2)
        return np.zeros((len(timestamps), 1), dtype=np.float32)

    dates = pd.to_datetime(timestamps).date
    years = {d.year for d in dates}
    try:
        cal = _holidays.country_holidays(country_code, years=years)
    except (KeyError, NotImplementedError):
        warnings.warn(f"Unknown country code {country_code!r}; defaulting to US.", stacklevel=2)
        cal = _holidays.country_holidays("US", years=years)
    return np.array([(1.0 if d in cal else 0.0) for d in dates], dtype=np.float32).reshape(-1, 1)


def _create_lagged_features(data: np.ndarray, num_lags: int) -> np.ndarray:
    lags = []
    for i in range(1, num_lags + 1):
        rolled = np.roll(data, shift=i, axis=0)
        rolled[:i] = 0.0
        lags.append(rolled)
    return np.concatenate(lags, axis=1).astype(np.float32)


class TrafficDataset(Dataset):
    """Windowed traffic dataset with optional time/holiday/weather/lag features.

    Returns ``(features_dict, target_tensor)`` per sample. Keys present in
    ``features_dict`` depend on the config flags. A ``concatenated`` key is
    always provided for backward-compatible tensor consumers.
    """

    def __init__(
        self,
        data: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        config: Optional[TrainingConfig] = None,
        weather: Optional[WeatherIntegration] = None,
    ) -> None:
        if config is None:
            config = TrainingConfig(base_output_dir="./output")

        self.config = config
        self.data = data.astype(np.float32, copy=False)
        self.timestamps = timestamps
        self.seq_length = config.seq_length
        self.pred_length = config.pred_length

        self.feature_groups: dict[str, np.ndarray] = {"traffic": self.data}

        if config.use_time_features and timestamps is not None:
            self.feature_groups["time"] = _create_time_features(timestamps)

        if config.use_holiday_feature and timestamps is not None:
            self.feature_groups["holiday"] = _create_holiday_feature(
                timestamps, config.holiday_country_code,
            )

        if config.use_weather_feature and timestamps is not None:
            if weather is None and config.weather_data_file:
                weather = WeatherIntegration(config.weather_data_file)
            if weather is not None:
                try:
                    self.feature_groups["weather"] = weather.align_to_timestamps(
                        timestamps, feature_name=config.weather_feature_type,
                    ).astype(np.float32)
                except Exception as exc:
                    logger.warning("Failed to attach weather features: %s", exc)

        if config.use_lagged_features:
            self.feature_groups["lagged"] = _create_lagged_features(self.data, config.num_lags)

        self.feature_dims = {name: arr.shape[-1] for name, arr in self.feature_groups.items()}
        self._concatenated = np.concatenate(list(self.feature_groups.values()), axis=1).astype(np.float32)
        self.feature_dims["concatenated"] = self._concatenated.shape[1]

    def __len__(self) -> int:
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        end = idx + self.seq_length
        target_end = end + self.pred_length

        features: dict[str, torch.Tensor] = {
            name: torch.from_numpy(arr[idx:end]) for name, arr in self.feature_groups.items()
        }
        features["concatenated"] = torch.from_numpy(self._concatenated[idx:end])
        target = torch.from_numpy(self.data[end:target_end])
        return features, target

    @property
    def total_feature_dim(self) -> int:
        return self.feature_dims["concatenated"]
