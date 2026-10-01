"""
Data module for the traffic prediction transformer.

Provides the TrafficDataset class (torch Dataset) and the prepare_data
utility for loading, cleaning, and scaling raw traffic DataFrames.
"""

import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics as sklearn_metrics

from .config import TrainingConfig, get_adjacency_matrix_path
from .weather import WeatherIntegration
from .spatial import SpatialIntegration, load_adjacency_matrix, normalize_adj

# Optional holidays support
HOLIDAYS_AVAILABLE = False
try:
    import holidays

    HOLIDAYS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# TrafficDataset
# ---------------------------------------------------------------------------

class TrafficDataset(Dataset):
    """A PyTorch Dataset that wraps traffic time-series data and generates
    input/target windows together with optional auxiliary features (time,
    weather, holidays, lags, spatial embeddings).
    """

    def __init__(
        self,
        data: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        sequence_length: int = 12,
        prediction_window: int = 12,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.data = data
        self.seq_length = sequence_length
        self.pred_window = prediction_window
        self.timestamps = timestamps
        self.config = config

        # ------------------------------------------------------------------
        # Feature groups
        # ------------------------------------------------------------------
        self.feature_groups: Dict[str, Dict[str, Any]] = {
            'traffic': {'data': self.data, 'dim': data.shape[1]},
        }

        # Time features
        if config is not None and getattr(config, 'use_time_features', False) and timestamps is not None:
            self.create_time_features(timestamps)
            self.feature_groups['time'] = {
                'data': self.time_features,
                'dim': self.time_features.shape[1],
            }

        # Holiday features
        if config is not None and getattr(config, 'use_holiday_feature', False) and timestamps is not None:
            self.create_holiday_feature(timestamps)
            self.feature_groups['holiday'] = {
                'data': self.holiday_feature,
                'dim': self.holiday_feature.shape[1],
            }

        # Weather features
        if config is not None and getattr(config, 'use_weather_feature', False) and timestamps is not None:
            try:
                self.create_weather_feature(timestamps)
                if hasattr(self, 'weather_feature') and self.weather_feature is not None:
                    self.feature_groups['weather'] = {
                        'data': self.weather_feature,
                        'dim': self.weather_feature.shape[1],
                    }
            except Exception as e:
                warnings.warn(f"Failed to create weather features: {e}")

        # Lagged features
        if config is not None and getattr(config, 'use_lagged_features', False):
            self.create_lagged_features()
            self.feature_groups['lagged'] = {
                'data': self.lagged_features,
                'dim': self.lagged_features.shape[1],
            }

        # Spatial features
        if config is not None and getattr(config, 'use_spatial_features', False):
            try:
                self.create_spatial_features()
                if hasattr(self, 'spatial_features') and self.spatial_features is not None:
                    self.feature_groups['spatial'] = {
                        'data': self.spatial_features,
                        'dim': self.spatial_features.shape[-1],
                    }
            except Exception as e:
                warnings.warn(f"Failed to create spatial features: {e}")

        # ------------------------------------------------------------------
        # Derived bookkeeping
        # ------------------------------------------------------------------
        self.feature_dims: Dict[str, int] = {
            name: group['dim'] for name, group in self.feature_groups.items()
        }

        self._prepare_concatenated_features()

        self.total_feature_dim = sum(self.feature_dims.values())

    # ------------------------------------------------------------------
    # Spatial features  (FIX #4 VECTORIZED)
    # ------------------------------------------------------------------

    def create_spatial_features(self) -> None:
        """Create spatial embedding features using SpatialIntegration.

        FIX #4: Uses broadcasting instead of nested loops for efficiency.
        """
        # PATCH: the old call ``SpatialIntegration(self.config)`` passed the
        # config as the adjacency path (TypeError -> silently disabled), and
        # the subsequent broadcast materialised a (samples, sensors, dim)
        # array — ~45 GB for PEMS-BAY. We now use the fixed factory and keep
        # the compact per-sensor embedding; it is static across time, so
        # nothing is gained by copying it per sample.
        spatial_integration = SpatialIntegration.create_spatial_integration_from_config(
            self.config
        )
        projected_embeddings = spatial_integration.get_projected_embeddings()

        num_sensors = self.data.shape[1]
        self.spatial_features = np.asarray(
            projected_embeddings[:num_sensors, :], dtype=np.float32
        )  # shape: (num_sensors, spatial_dim) — static, per-sensor

    # ------------------------------------------------------------------
    # Concatenated features  (FIX #10)
    # ------------------------------------------------------------------

    def _prepare_concatenated_features(self) -> None:
        """Build a single 2-D concatenated feature array from all feature
        groups.  This is consumed by downstream code that expects a flat
        ``(n_samples, total_dim)`` representation.

        FIX #10: When spatial features are 3-D we take the mean across the
        sensor axis so that concatenation into the 2-D array works.  The
        full 3-D tensor is still kept in ``feature_groups['spatial']`` for
        callers that need per-sensor spatial information.
        """
        num_samples = len(self.data)
        arrays: List[np.ndarray] = []
        for name, group in self.feature_groups.items():
            arr = group['data']
            if arr.ndim == 3:
                # FIX #10: (samples, sensors, dim) -> average over sensors.
                arr = arr.mean(axis=1)
            # PATCH: static per-sensor arrays (e.g. spatial embeddings of
            # shape (num_sensors, dim)) are not time-aligned and cannot be
            # concatenated with per-timestep features; skip them here.
            if arr.shape[0] != num_samples:
                continue
            arrays.append(arr)

        self.concatenated_features = np.concatenate(arrays, axis=1)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data) - self.seq_length - self.pred_window + 1

    def __getitem__(self, idx: int):
        x_data = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_window]

        # PATCH (feature-wise attention): when enabled, yield a dict of
        # per-group tensors. PyTorch's default collate batches dicts of
        # tensors natively, _process_batch moves them to device, and
        # TrafficTransformer.encode routes dict inputs through the
        # FeatureAttention module.
        if self.config is not None and getattr(self.config, 'use_feature_attention', False):
            sl = slice(idx, idx + self.seq_length)
            x_dict = {'traffic': torch.FloatTensor(x_data)}
            if hasattr(self, 'time_features') and self.config.use_time_features:
                x_dict['time'] = torch.FloatTensor(self.time_features[sl])
            if hasattr(self, 'weather_feature') and self.config.use_weather_feature:
                x_dict['weather'] = torch.FloatTensor(self.weather_feature[sl])
            if hasattr(self, 'holiday_feature') and getattr(self.config, 'use_holiday_feature', False):
                x_dict['holiday'] = torch.FloatTensor(self.holiday_feature[sl])
            if hasattr(self, 'lagged_features') and getattr(self.config, 'use_lagged_features', False):
                x_dict['lagged'] = torch.FloatTensor(self.lagged_features[sl])
            return x_dict, torch.FloatTensor(y)

        x_features: List[np.ndarray] = []

        if hasattr(self, 'time_features') and self.config.use_time_features:
            time_feat = self.time_features[idx:idx + self.seq_length]
            x_features.append(time_feat)

        if hasattr(self, 'weather_feature') and self.config.use_weather_feature:
            weather_feat = self.weather_feature[idx:idx + self.seq_length]
            x_features.append(weather_feat)

        # PATCH: holiday and lagged feature groups were built but never
        # concatenated into the input; they now enter the model when enabled.
        if hasattr(self, 'holiday_feature') and getattr(self.config, 'use_holiday_feature', False):
            x_features.append(self.holiday_feature[idx:idx + self.seq_length])

        if hasattr(self, 'lagged_features') and getattr(self.config, 'use_lagged_features', False):
            x_features.append(self.lagged_features[idx:idx + self.seq_length])

        if x_features:
            x = np.concatenate([x_data] + x_features, axis=1)
        else:
            x = x_data

        return torch.FloatTensor(x), torch.FloatTensor(y)

    # ------------------------------------------------------------------
    # Auxiliary feature creation helpers
    # ------------------------------------------------------------------

    def create_time_features(self, timestamps: pd.DatetimeIndex) -> None:
        """Create normalised time-of-day / calendar features.

        Features (all in [0, 1]):
            - hour / 23
            - day_of_week / 6
            - week_of_year (day_of_year // 7) / 51
            - month / 11
        """
        timestamps = pd.DatetimeIndex(timestamps)
        hour = timestamps.hour / 23.0
        dayofweek = timestamps.dayofweek / 6.0
        weekofyear = (timestamps.dayofyear // 7) / 51.0
        month = (timestamps.month - 1) / 11.0

        self.time_features = np.column_stack([hour, dayofweek, weekofyear, month])

    def create_holiday_feature(self, timestamps: pd.DatetimeIndex) -> None:
        """Create a binary holiday indicator feature.

        If the ``holidays`` package is not installed, a zero-vector is used
        as a fallback.
        """
        timestamps = pd.DatetimeIndex(timestamps)
        if HOLIDAYS_AVAILABLE:
            # PATCH: config defines ``holiday_country_code``; the old lookup
            # key 'holiday_country' never existed, so the setting was ignored.
            country = getattr(
                self.config, 'holiday_country_code',
                getattr(self.config, 'holiday_country', 'MZ'),
            )
            holiday_cal = holidays.country_holidays(country)
            holiday_flags = np.array(
                [1.0 if ts.date() in holiday_cal else 0.0 for ts in timestamps]
            )
        else:
            warnings.warn(
                "holidays package not installed; using dummy zeros for holiday feature."
            )
            holiday_flags = np.zeros(len(timestamps))

        self.holiday_feature = holiday_flags.reshape(-1, 1)

    def create_weather_feature(self, timestamps: pd.DatetimeIndex) -> None:
        """Load weather data and align it to the traffic timestamps."""
        # PATCH: config sets ``weather_data_file``; the old code read
        # ``weather_data_path`` (which never exists), so weather features were
        # silently skipped even with use_weather_feature=True.
        weather_file = (
            getattr(self.config, 'weather_data_file', None)
            or getattr(self.config, 'weather_data_path', None)
        )
        if weather_file is None or not os.path.exists(weather_file):
            warnings.warn(
                f"Weather data file not set or not found "
                f"(weather_data_file={weather_file!r}); skipping weather features."
            )
            return

        weather = WeatherIntegration()
        weather.load_weather_data(weather_file)
        self.weather_feature = weather.match_weather_to_traffic(
            timestamps, feature_name='all_features'
        )

    def create_lagged_features(self) -> None:
        """Create lagged copies of the traffic data.

        Lag steps are taken from ``config.lag_steps`` (default ``[1, 2, 3]``).
        Leading positions that have no history are filled with NaN.
        """
        lag_steps: List[int] = getattr(self.config, 'lag_steps', [1, 2, 3])
        lagged: List[np.ndarray] = []
        for lag in lag_steps:
            # PATCH: leading positions were NaN-filled, which would poison the
            # model input; replicate the first observation instead.
            lagged_data = np.empty_like(self.data)
            if lag < len(self.data):
                lagged_data[lag:] = self.data[:-lag]
                lagged_data[:lag] = self.data[0]
            else:
                lagged_data[:] = self.data[0]
            lagged.append(lagged_data)

        self.lagged_features = np.concatenate(lagged, axis=1)

    # ------------------------------------------------------------------
    # Adjacency matrix
    # ------------------------------------------------------------------

    def get_adjacency_matrix(self) -> torch.Tensor:
        """Load, normalise, and return the adjacency matrix as a tensor.

        The result is cached per dataset instance.

        PATCH: three fixes. (1) The cache was a *class* attribute shared
        across datasets (a stale PEMS-BAY matrix could leak into a Maputo
        run); it is now per-instance. (2) ``load_adjacency_matrix`` returns a
        ``(matrix, sensor_ids, node_ids)`` tuple, which the old code passed
        whole into ``normalize_adj`` (AttributeError -> adjacency silently
        None). (3) ``get_adjacency_matrix_path`` now accepts the config alone.
        """
        cached = getattr(self, '_adj_matrix_cache', None)
        if cached is not None:
            return cached

        adj_path = get_adjacency_matrix_path(self.config)
        if adj_path is None:
            raise FileNotFoundError(
                f"No adjacency matrix found for dataset "
                f"'{getattr(self.config, 'dataset_name', '?')}'."
            )
        adj, _sensor_ids, _node_ids = load_adjacency_matrix(
            adj_path, fallback_size=self.data.shape[1]
        )
        adj_norm = normalize_adj(adj)

        self._adj_matrix_cache = torch.FloatTensor(adj_norm)
        return self._adj_matrix_cache


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

def prepare_data(
    df: pd.DataFrame,
    config: TrainingConfig,
    scaler_fit_fraction: Optional[float] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex, Any, int]:
    """Clean, impute, and scale a raw traffic DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw traffic data whose index should be a ``DatetimeIndex`` (or
        convertible to one) and whose columns are sensor readings.
    config : TrainingConfig
        Experiment configuration that controls the missing-value strategy
        (``config.missing_value_strategy``) and the scaler type
        (``config.data_scaler_type``).

    Returns
    -------
    data : np.ndarray
        Scaled sensor data of shape ``(n_timesteps, n_sensors)``.
    timestamps : pd.DatetimeIndex
        The corresponding timestamps.
    scaler : sklearn scaler
        The fitted scaler instance (useful for inverse-transforming
        predictions later).
    n_sensors : int
        Number of sensor columns.
    """

    # PATCH: operate on a copy — the old in-place fills/sorts mutated the
    # caller's DataFrame as a side effect.
    df = df.copy()

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df.set_index(first_col, inplace=True)
    df.sort_index(inplace=True)

    timestamps = df.index

    # ------------------------------------------------------------------
    # Missing-value handling  (FIX #9)
    # ------------------------------------------------------------------
    strategy = getattr(config, 'missing_value_strategy', 'ffill_bfill')

    if strategy == 'ffill_bfill':
        # Original behaviour: treat zeros as missing values
        df.replace(0.0, np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
    elif strategy == 'zero':
        df.fillna(0.0, inplace=True)
    elif strategy == 'mean':
        # FIX #9: Only fill actual NaN, don't treat zeros as missing
        df.fillna(df.mean(), inplace=True)
    elif strategy == 'median':
        # FIX #9: Only fill actual NaN, don't treat zeros as missing
        df.fillna(df.median(), inplace=True)
    elif strategy == 'interpolate':
        df.interpolate(method='time', inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
    elif strategy == 'mean_replace_zeros':
        # Legacy behavior: treat zeros as missing
        df.replace(0.0, np.nan, inplace=True)
        df.fillna(df.mean(), inplace=True)
    else:
        warnings.warn(
            f"Unknown missing_value_strategy '{strategy}'; "
            "falling back to ffill/bfill without zero replacement."
        )
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    n_sensors = df.shape[1]

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    scaler_type = getattr(config, 'data_scaler_type', 'minmax')

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()

    # PATCH (data leakage): previously the scaler was fitted on the FULL
    # series before the temporal split, letting val/test statistics leak into
    # normalisation. When *scaler_fit_fraction* is given (the pipeline passes
    # the training fraction, 0.7), the scaler is fitted on that leading slice
    # only and then applied to the whole series.
    values = df.values
    if scaler_fit_fraction is not None and 0.0 < scaler_fit_fraction < 1.0:
        fit_end = max(1, int(len(values) * scaler_fit_fraction))
        scaler.fit(values[:fit_end])
        data = scaler.transform(values)
    else:
        data = scaler.fit_transform(values)

    return data, timestamps, scaler, n_sensors


# ---------------------------------------------------------------------------
# High-level pipeline: load_and_prepare_data
# ---------------------------------------------------------------------------

def load_and_prepare_data(
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, Any, Optional[torch.Tensor]]:
    """End-to-end data pipeline: load CSV, clean, scale, split, build loaders.

    This is the single-call entry point used by the orchestrator notebook.

    Parameters
    ----------
    config : TrainingConfig
        Must have ``input_dir`` already set (e.g. via ``setup_directories``).

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    scaler : fitted sklearn scaler (for inverse-transforming predictions)
    adj_matrix : torch.Tensor or None
    """
    input_dir = config.input_dir
    if input_dir is None:
        raise ValueError(
            "config.input_dir is not set. Call setup_directories(config) first."
        )

    # --- Load CSV ----------------------------------------------------------
    dataset_filename = f"{config.dataset_name}.csv"
    csv_path = os.path.join(input_dir, dataset_filename)
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{dataset_filename} not found in {input_dir}. "
            "Place the dataset CSV there or set config.dataset_name."
        )
    print(f"Loaded {config.dataset_name} data with shape: {df.shape}")

    # --- Clean & scale -----------------------------------------------------
    # PATCH: scaler is now fitted on the training fraction only (see
    # prepare_data) to remove val/test leakage from normalisation.
    train_fraction = 0.7
    data_normalized, timestamps, data_scaler, num_features = prepare_data(
        df, config, scaler_fit_fraction=train_fraction
    )

    # --- Adjacency matrix --------------------------------------------------
    # PATCH: adjacency is required by the GNN pre-encoder alone (spatial
    # embedding features are a separate mechanism), and is loaded directly
    # instead of via a throwaway TrafficDataset.
    adj_matrix: Optional[torch.Tensor] = None
    if config.use_gnn_pre_transformer:
        try:
            adj_path = get_adjacency_matrix_path(config, input_dir)
            if adj_path is not None:
                adj_raw, _ids, _nodes = load_adjacency_matrix(
                    adj_path, fallback_size=num_features
                )
                if adj_raw.shape[0] != num_features:
                    warnings.warn(
                        f"Adjacency matrix shape {adj_raw.shape} does not match "
                        f"the {num_features} sensor columns; ignoring it."
                    )
                else:
                    adj_matrix = torch.FloatTensor(normalize_adj(adj_raw))
                    print(f"Loaded adjacency matrix for {config.dataset_name}")
            else:
                warnings.warn(
                    "use_gnn_pre_transformer=True but no adjacency matrix was "
                    "found; the GNN pre-encoder will be skipped."
                )
        except Exception as e:
            warnings.warn(f"Could not load adjacency matrix: {e}")

    # --- Temporal split 70 / 10 / 20 --------------------------------------
    n = len(data_normalized)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    splits = {
        "train": (data_normalized[:train_end], timestamps[:train_end]),
        "val":   (data_normalized[train_end:val_end], timestamps[train_end:val_end]),
        "test":  (data_normalized[val_end:], timestamps[val_end:]),
    }
    print(
        f"Split — Train: {train_end}, Val: {val_end - train_end}, "
        f"Test: {n - val_end} timesteps"
    )

    # --- Datasets & DataLoaders --------------------------------------------
    loaders = {}
    for name, (data_split, ts_split) in splits.items():
        ds = TrafficDataset(
            data_split, ts_split,
            config.seq_length, config.pred_length, config,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=2 if config.num_workers > 0 else None,
            drop_last=(name == "train"),
            # PATCH: avoid respawning workers every epoch on Colab.
            persistent_workers=(config.num_workers > 0),
        )

    return loaders["train"], loaders["val"], loaders["test"], data_scaler, adj_matrix
