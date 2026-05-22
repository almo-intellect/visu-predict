from __future__ import annotations

import numpy as np
import pandas as pd

from visu_predict.config import TrainingConfig
from visu_predict.data import TrafficDataset, prepare_data


def _make_df(rows: int = 96, sensors: int = 8) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="5min")
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.uniform(10, 70, size=(rows, sensors)), index=timestamps)


def test_prepare_data_normalises_and_returns_shape():
    cfg = TrainingConfig(base_output_dir="./tmp", missing_value_strategy="zero")
    df = _make_df()
    data, ts, _scaler, n_features = prepare_data(df, cfg)
    assert data.shape == df.shape
    assert n_features == df.shape[1]
    assert len(ts) == len(df)


def test_traffic_dataset_iteration_shapes():
    cfg = TrainingConfig(
        base_output_dir="./tmp", seq_length=8, pred_length=4,
        missing_value_strategy="zero", use_time_features=True,
    )
    df = _make_df(rows=48, sensors=5)
    data, ts, _, _ = prepare_data(df, cfg)
    ds = TrafficDataset(data, ts, cfg)
    assert len(ds) == len(data) - cfg.seq_length - cfg.pred_length + 1

    features, target = ds[0]
    assert "traffic" in features
    assert "time" in features
    assert "concatenated" in features
    assert features["traffic"].shape == (cfg.seq_length, df.shape[1])
    assert target.shape == (cfg.pred_length, df.shape[1])


def test_traffic_dataset_lagged_features():
    cfg = TrainingConfig(
        base_output_dir="./tmp", seq_length=6, pred_length=3,
        missing_value_strategy="zero", use_lagged_features=True, num_lags=2,
    )
    df = _make_df(rows=24, sensors=3)
    data, ts, _, _ = prepare_data(df, cfg)
    ds = TrafficDataset(data, ts, cfg)
    features, _ = ds[0]
    assert "lagged" in features
    assert features["lagged"].shape == (cfg.seq_length, df.shape[1] * cfg.num_lags)
