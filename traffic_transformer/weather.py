"""
Weather integration module for the traffic prediction transformer.

Provides weather data loading, preprocessing, and matching to traffic timestamps.
Fully independent module with no internal package imports.
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

WEATHER_CONDITION_MAP = {
    'Fair': 0, 'Partly Cloudy': 1, 'Mostly Cloudy': 2, 'Cloudy': 3,
    'Rain': 4, 'Snow': 5, 'Thunderstorm': 6, 'Fog': 7
}

CLOUD_COVER_MAP = {
    'CLR': 0, 'FEW': 1, 'SCT': 2, 'BKN': 3, 'OVC': 4
}

WIND_DIRECTION_MAP = {
    'CALM': 0, 'N': 1, 'NNE': 2, 'NE': 3, 'ENE': 4,
    'E': 5, 'ESE': 6, 'SE': 7, 'SSE': 8,
    'S': 9, 'SSW': 10, 'SW': 11, 'WSW': 12,
    'W': 13, 'WNW': 14, 'NW': 15, 'NNW': 16, 'VAR': 17
}

# PATCH: real-world observation feeds (e.g. Weather Underground exports) use
# many label variants that the 8-entry WEATHER_CONDITION_MAP does not cover
# ('Light Rain', 'Mostly Cloudy / Windy', 'Haze', ...). Previously these
# mapped to NaN and were silently forward-filled with neighbouring values.
# We now normalise common variants onto the base categories and warn about
# whatever remains unmapped.
WEATHER_CONDITION_VARIANTS = {
    'Light Rain': 'Rain', 'Heavy Rain': 'Rain', 'Rain Shower': 'Rain',
    'Light Rain Shower': 'Rain', 'Showers in the Vicinity': 'Rain',
    'Drizzle': 'Rain', 'Light Drizzle': 'Rain', 'Heavy Drizzle': 'Rain',
    'Light Snow': 'Snow', 'Heavy Snow': 'Snow', 'Snow Shower': 'Snow',
    'Wintry Mix': 'Snow', 'Sleet': 'Snow', 'Light Sleet': 'Snow',
    'T-Storm': 'Thunderstorm', 'Heavy T-Storm': 'Thunderstorm',
    'Thunder': 'Thunderstorm', 'Thunder in the Vicinity': 'Thunderstorm',
    'Haze': 'Fog', 'Mist': 'Fog', 'Patches of Fog': 'Fog',
    'Shallow Fog': 'Fog', 'Smoke': 'Fog',
    'Clear': 'Fair', 'Sunny': 'Fair', 'Mostly Sunny': 'Partly Cloudy',
    'Overcast': 'Cloudy',
}


def _normalize_condition(value):
    """Map a raw weather-condition label onto a base category."""
    if not isinstance(value, str):
        return value
    v = value.strip()
    # Strip qualifier suffixes such as 'Mostly Cloudy / Windy'
    for suffix in (' / Windy', ' and Windy', ' / Wind'):
        if v.endswith(suffix):
            v = v[: -len(suffix)]
            break
    return WEATHER_CONDITION_VARIANTS.get(v, v)


class WeatherIntegration:
    """Loads, preprocesses, and serves weather features aligned to traffic timestamps."""

    def __init__(self):
        self.weather_df: Optional[pd.DataFrame] = None
        self.feature_arrays: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Data loading and preprocessing
    # ------------------------------------------------------------------

    def load_weather_data(self, filepath: str) -> None:
        """Load weather CSV, encode categoricals, normalise numericals, and
        build the ``feature_arrays`` dictionary.

        Parameters
        ----------
        filepath : str
            Path to the weather CSV file.  Must contain a datetime-parseable
            column that will be used as the index.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weather data file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Parse the first column as datetime and set as index
        datetime_col = df.columns[0]
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)

        # ----- Map categorical features to numeric codes -----
        # PATCH: normalise label variants first, then warn about anything
        # that still fails to map instead of silently forward-filling it.
        def _map_and_report(col: str, mapping: Dict[str, int], normalizer=None) -> None:
            raw = df[col]
            values = raw.map(normalizer) if normalizer is not None else raw
            mapped = values.map(mapping)
            unmapped_mask = mapped.isna() & raw.notna()
            n_unmapped = int(unmapped_mask.sum())
            if n_unmapped:
                examples = sorted(set(raw[unmapped_mask].astype(str)))[:5]
                warnings.warn(
                    f"weather: {n_unmapped}/{len(raw)} "
                    f"({100.0 * n_unmapped / max(1, len(raw)):.1f}%) values in "
                    f"'{col}' were unrecognized and will be forward-filled. "
                    f"Examples: {examples}"
                )
            df[col] = mapped

        if 'weather_condition' in df.columns:
            _map_and_report('weather_condition', WEATHER_CONDITION_MAP, _normalize_condition)
        if 'cloud_cover' in df.columns:
            _map_and_report('cloud_cover', CLOUD_COVER_MAP)
        if 'wind_direction' in df.columns:
            _map_and_report('wind_direction', WIND_DIRECTION_MAP)

        # Forward-fill then backward-fill remaining NaN values
        df = df.ffill().bfill()

        # ----- Normalise numerical features with MinMaxScaler -----
        numerical_features = [
            'temperature', 'visibility', 'wind_speed',
            'relative_humidity', 'dew_point',
        ]
        numerical_present = [f for f in numerical_features if f in df.columns]

        if numerical_present:
            scaler = MinMaxScaler()
            df[numerical_present] = scaler.fit_transform(df[numerical_present])

        # ----- Normalise categorical codes by their max values -----
        categorical_max = {
            'weather_condition': max(WEATHER_CONDITION_MAP.values()),
            'cloud_cover': max(CLOUD_COVER_MAP.values()),
            'wind_direction': max(WIND_DIRECTION_MAP.values()),
        }
        for col, max_val in categorical_max.items():
            if col in df.columns and max_val > 0:
                df[col] = df[col] / max_val

        self.weather_df = df

        # ----- Build feature_arrays dict -----
        self.feature_arrays = {}

        if 'temperature' in df.columns:
            self.feature_arrays['temperature'] = df[['temperature']].values

        if 'weather_condition' in df.columns:
            self.feature_arrays['weather_condition'] = df[['weather_condition']].values

        if 'visibility' in df.columns:
            self.feature_arrays['visibility'] = df[['visibility']].values

        # Wind: combine wind_speed and wind_direction if both present
        wind_cols = [c for c in ['wind_speed', 'wind_direction'] if c in df.columns]
        if wind_cols:
            self.feature_arrays['wind'] = df[wind_cols].values

        if 'relative_humidity' in df.columns:
            self.feature_arrays['humidity'] = df[['relative_humidity']].values

        if 'dew_point' in df.columns:
            self.feature_arrays['dew_point'] = df[['dew_point']].values

        if 'cloud_cover' in df.columns:
            self.feature_arrays['cloud_cover'] = df[['cloud_cover']].values

        # Concatenate every individual feature into 'all_features'
        if self.feature_arrays:
            self.feature_arrays['all_features'] = np.hstack(
                [v for k, v in self.feature_arrays.items()]
            )

    # ------------------------------------------------------------------
    # Matching weather to traffic timestamps
    # ------------------------------------------------------------------

    def match_weather_to_traffic(
        self,
        traffic_timestamps,
        feature_name: str = 'all_features',
    ) -> np.ndarray:
        """Return weather features aligned to *traffic_timestamps* using a
        nearest-time matching approach.

        Parameters
        ----------
        traffic_timestamps : array-like of datetime-like
            Timestamps from the traffic dataset.
        feature_name : str, optional
            Key into ``self.feature_arrays`` (default ``'all_features'``).

        Returns
        -------
        np.ndarray
            Weather feature array with one row per traffic timestamp.
        """
        if self.weather_df is None or self.feature_arrays is None:
            raise RuntimeError("Weather data has not been loaded yet. Call load_weather_data() first.")

        if feature_name not in self.feature_arrays:
            raise KeyError(
                f"Feature '{feature_name}' not found. "
                f"Available features: {list(self.feature_arrays.keys())}"
            )

        traffic_timestamps = pd.DatetimeIndex(traffic_timestamps)
        weather_index = self.weather_df.index

        # Use get_indexer with nearest method for time-based matching
        indices = weather_index.get_indexer(traffic_timestamps, method='nearest')

        feature_data = self.feature_arrays[feature_name]
        matched_features = feature_data[indices]

        return matched_features

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_feature_dimension(self, feature_name: str = 'all_features') -> int:
        """Return the number of columns for the specified feature.

        Parameters
        ----------
        feature_name : str, optional
            Key into ``self.feature_arrays`` (default ``'all_features'``).

        Returns
        -------
        int
            Number of feature columns.
        """
        if self.feature_arrays is None:
            raise RuntimeError("Weather data has not been loaded yet. Call load_weather_data() first.")

        if feature_name not in self.feature_arrays:
            raise KeyError(
                f"Feature '{feature_name}' not found. "
                f"Available features: {list(self.feature_arrays.keys())}"
            )

        return self.feature_arrays[feature_name].shape[1]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def create_weather_feature_for_dataset(
    dataset,
    weather_file_path: str,
    feature_name: str = 'all_features',
) -> np.ndarray:
    """Create weather features matched to a traffic dataset's timestamps.

    Parameters
    ----------
    dataset
        A traffic dataset object that exposes a ``timestamps`` attribute.
    weather_file_path : str
        Path to the weather CSV file.
    feature_name : str, optional
        Which weather feature set to use (default ``'all_features'``).

    Returns
    -------
    np.ndarray
        Weather features aligned to the dataset's timestamps.
    """
    weather = WeatherIntegration()
    weather.load_weather_data(weather_file_path)
    matched_features = weather.match_weather_to_traffic(
        dataset.timestamps, feature_name=feature_name
    )
    return matched_features
