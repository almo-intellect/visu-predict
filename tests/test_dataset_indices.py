from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from visu_predict.config import TrainingConfig
from visu_predict.data import TrafficDataset, _infer_steps_per_day, prepare_data


def _make_df(rows: int = 96, sensors: int = 5, freq: str = "5min") -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq=freq)
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.uniform(10, 70, size=(rows, sensors)), index=ts)


def test_infer_steps_per_day_5min():
    ts = pd.date_range("2024-01-01", periods=300, freq="5min")
    assert _infer_steps_per_day(ts) == 288


def test_infer_steps_per_day_15min():
    ts = pd.date_range("2024-01-01", periods=200, freq="15min")
    assert _infer_steps_per_day(ts) == 96


def test_discrete_time_indices_emitted_when_enabled():
    cfg = TrainingConfig(
        base_output_dir="./tmp", seq_length=4, pred_length=2,
        missing_value_strategy="zero", use_discrete_time_embeddings=True,
    )
    df = _make_df()
    data, ts, _, _ = prepare_data(df, cfg)
    ds = TrafficDataset(data, ts, cfg)

    features, _ = ds[0]
    assert "time_of_day_idx" in features
    assert "day_of_week_idx" in features
    assert features["time_of_day_idx"].dtype == torch.int64
    assert features["day_of_week_idx"].dtype == torch.int64
    assert features["time_of_day_idx"].shape == (cfg.seq_length,)
    assert features["day_of_week_idx"].shape == (cfg.seq_length,)
    assert (features["time_of_day_idx"] >= 0).all()
    assert (features["time_of_day_idx"] < ds.steps_per_day).all()
    assert (features["day_of_week_idx"] >= 0).all()
    assert (features["day_of_week_idx"] < 7).all()


def test_discrete_indices_replace_continuous_time_group():
    cfg = TrainingConfig(
        base_output_dir="./tmp", seq_length=4, pred_length=2,
        missing_value_strategy="zero",
        use_discrete_time_embeddings=True, use_time_features=True,
    )
    df = _make_df()
    data, ts, _, _ = prepare_data(df, cfg)
    ds = TrafficDataset(data, ts, cfg)
    features, _ = ds[0]
    # discrete mode wins: continuous "time" group should be absent
    assert "time" not in features


def test_legacy_time_features_present_when_discrete_disabled():
    cfg = TrainingConfig(
        base_output_dir="./tmp", seq_length=4, pred_length=2,
        missing_value_strategy="zero", use_time_features=True,
        use_discrete_time_embeddings=False,
    )
    df = _make_df()
    data, ts, _, _ = prepare_data(df, cfg)
    ds = TrafficDataset(data, ts, cfg)
    features, _ = ds[0]
    assert "time" in features
    assert "time_of_day_idx" not in features
