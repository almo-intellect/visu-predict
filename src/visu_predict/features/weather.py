"""Weather data integration for traffic prediction.

Loads, normalises, and aligns weather observations against traffic timestamps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

WEATHER_CONDITION_MAP: dict[str, int] = {
    "Fair": 0, "Partly Cloudy": 1, "Mostly Cloudy": 2, "Cloudy": 3,
    "Rain": 4, "Snow": 5, "Thunderstorm": 6, "Fog": 7,
}
CLOUD_COVER_MAP: dict[str, int] = {"CLR": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}
WIND_DIRECTION_MAP: dict[str, int] = {
    "CALM": 0, "N": 1, "NNE": 2, "NE": 3, "ENE": 4, "E": 5, "ESE": 6, "SE": 7,
    "SSE": 8, "S": 9, "SSW": 10, "SW": 11, "WSW": 12, "W": 13, "WNW": 14,
    "NW": 15, "NNW": 16, "VAR": 17,
}

NUMERICAL_FEATURES: tuple[str, ...] = (
    "temperature", "visibility", "wind_speed", "relative_humidity", "dew_point",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "weather_condition_code", "wind_direction_code", "cloud_cover_code",
)
ALL_WEATHER_COLUMNS: tuple[str, ...] = (
    "temperature", "weather_condition_code", "visibility", "wind_speed",
    "wind_direction_code", "relative_humidity", "dew_point", "cloud_cover_code",
)


def _map_categorical(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    return series.map(lambda x: mapping.get(x, 0) if pd.notnull(x) else 0).astype(float)


class WeatherIntegration:
    """Load weather CSV and align observations with traffic timestamps."""

    def __init__(self, weather_file_path: Optional[str | Path] = None) -> None:
        self.weather_df: Optional[pd.DataFrame] = None
        self.feature_arrays: Optional[dict[str, np.ndarray]] = None
        if weather_file_path is not None:
            self.load(weather_file_path)

    def load(self, filepath: str | Path) -> pd.DataFrame:
        """Load and preprocess weather CSV. Requires a 'datetime' column."""
        path = Path(filepath)
        logger.info("Loading weather data from %s", path)
        df = pd.read_csv(path)
        if "datetime" not in df.columns:
            raise ValueError(f"Weather CSV must contain a 'datetime' column (got {list(df.columns)})")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").replace("", np.nan)

        if "weather_condition" in df.columns:
            df["weather_condition_code"] = _map_categorical(df["weather_condition"], WEATHER_CONDITION_MAP)
        if "cloud_cover" in df.columns:
            df["cloud_cover_code"] = _map_categorical(df["cloud_cover"], CLOUD_COVER_MAP)
        if "wind_direction" in df.columns:
            df["wind_direction_code"] = _map_categorical(df["wind_direction"], WIND_DIRECTION_MAP)

        df = df.ffill().bfill()
        self.weather_df = df
        self._build_feature_arrays()
        logger.info(
            "Loaded %d weather records (%s → %s)",
            len(df), df.index.min(), df.index.max(),
        )
        return df

    def _build_feature_arrays(self) -> None:
        assert self.weather_df is not None
        df = self.weather_df.copy()

        for col in ALL_WEATHER_COLUMNS:
            if col not in df.columns:
                logger.warning("Weather column %s missing; filling with zeros", col)
                df[col] = 0.0

        features = df[list(ALL_WEATHER_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        features = features.fillna(features.mean(numeric_only=True)).fillna(0.0)

        numerical = list(NUMERICAL_FEATURES)
        scaler = MinMaxScaler()
        if not features[numerical].empty:
            features[numerical] = scaler.fit_transform(features[numerical])

        max_vals = {
            "weather_condition_code": max(WEATHER_CONDITION_MAP.values()),
            "wind_direction_code": max(WIND_DIRECTION_MAP.values()),
            "cloud_cover_code": max(CLOUD_COVER_MAP.values()),
        }
        for col, mv in max_vals.items():
            if mv > 0:
                features[col] = features[col] / mv

        self.feature_arrays = {
            "temperature": features[["temperature"]].to_numpy(),
            "weather_condition": features[["weather_condition_code"]].to_numpy(),
            "visibility": features[["visibility"]].to_numpy(),
            "wind": features[["wind_speed", "wind_direction_code"]].to_numpy(),
            "wind_speed": features[["wind_speed"]].to_numpy(),
            "wind_direction_code": features[["wind_direction_code"]].to_numpy(),
            "humidity": features[["relative_humidity"]].to_numpy(),
            "dew_point": features[["dew_point"]].to_numpy(),
            "cloud_cover_code": features[["cloud_cover_code"]].to_numpy(),
            "all_features": features[list(ALL_WEATHER_COLUMNS)].to_numpy(),
        }

    def align_to_timestamps(
        self,
        traffic_timestamps: pd.DatetimeIndex,
        feature_name: str = "all_features",
    ) -> np.ndarray:
        """Return weather features matched to traffic timestamps (nearest-time join)."""
        if self.weather_df is None or self.feature_arrays is None:
            raise RuntimeError("Call load() before align_to_timestamps().")
        if feature_name not in self.feature_arrays:
            raise KeyError(
                f"Unknown weather feature {feature_name!r}; "
                f"available: {sorted(self.feature_arrays)}"
            )

        selected = self.feature_arrays[feature_name]
        weather_index = self.weather_df.index
        nearest = weather_index.get_indexer(traffic_timestamps, method="nearest")
        return selected[nearest]

    def feature_dim(self, feature_name: str = "all_features") -> int:
        if self.feature_arrays is None:
            raise RuntimeError("Call load() first.")
        return self.feature_arrays[feature_name].shape[1]
