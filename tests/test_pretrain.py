from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from visu_predict.config import TrainingConfig
from visu_predict.data import prepare_data
from visu_predict.models.transformer import TrafficTransformer
from visu_predict.training.pretrain import (
    MaskedSTDataset,
    load_pretrained_encoder,
    masked_reconstruction_loss,
    save_pretrained_encoder,
)


def _make_df(rows: int = 96, sensors: int = 4) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="5min")
    rng = np.random.default_rng(7)
    return pd.DataFrame(rng.uniform(10, 70, size=(rows, sensors)), index=ts)


def _stae_config(**overrides) -> TrainingConfig:
    base = {
        "base_output_dir": "./tmp",
        "seq_length": 6,
        "pred_length": 3,
        "hidden_dim": 24,
        "num_heads": 4,
        "num_layers": 1,
        "model_pipeline": "stae",
        "use_discrete_time_embeddings": True,
        "d_input": 6, "d_tod": 6, "d_dow": 6, "d_adaptive": 6, "d_node": 0,
        "missing_value_strategy": "zero",
        "mask_ratio": 0.3,
    }
    base.update(overrides)
    return TrainingConfig(**base)


def test_masked_dataset_emits_mask_and_zeros_traffic():
    cfg = _stae_config()
    df = _make_df()
    data, ts, _, _ = prepare_data(df, cfg)
    ds = MaskedSTDataset(data, ts, cfg, rng=np.random.default_rng(0))
    features, target = ds[0]
    assert "mask" in features
    assert features["mask"].dtype == torch.bool
    assert features["mask"].shape == (cfg.seq_length, df.shape[1])
    # Where mask is True, the traffic feature must be zeroed.
    assert (features["traffic"][features["mask"]] == 0).all()
    # Target equals the original (pre-masking) traffic window.
    assert target.shape == features["traffic"].shape


def test_masked_dataset_rejects_invalid_ratio():
    cfg = _stae_config()
    df = _make_df()
    data, ts, _, _ = prepare_data(df, cfg)
    with pytest.raises(ValueError, match="mask_ratio"):
        MaskedSTDataset(data, ts, cfg, mask_ratio=0.0)
    with pytest.raises(ValueError, match="mask_ratio"):
        MaskedSTDataset(data, ts, cfg, mask_ratio=1.0)


def test_masked_loss_only_over_masked_cells():
    pred = torch.zeros(1, 4, 3)
    target = torch.ones(1, 4, 3)
    mask = torch.zeros(1, 4, 3, dtype=torch.bool)
    mask[0, 0, 0] = True  # single cell
    loss = masked_reconstruction_loss(pred, target, mask)
    # Only one masked position with diff^2 = 1
    assert torch.allclose(loss, torch.tensor(1.0))


def test_masked_loss_zero_when_no_mask():
    pred = torch.randn(1, 4, 3)
    target = torch.randn(1, 4, 3)
    mask = torch.zeros_like(pred, dtype=torch.bool)
    loss = masked_reconstruction_loss(pred, target, mask)
    assert loss.item() == 0.0


def test_forward_reconstruct_shape_and_gradient():
    batch, seq_len, num_sensors = 1, 6, 4
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=24, num_heads=4, num_layers=1, pred_len=3,
        model_pipeline="stae", seq_length=seq_len,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
    )
    traffic = torch.randn(batch, seq_len, num_sensors)
    mask = torch.zeros(batch, seq_len, num_sensors, dtype=torch.bool)
    mask[0, 2, 1] = True
    masked_traffic = traffic.masked_fill(mask, 0.0)
    features = {
        "traffic": masked_traffic,
        "time_of_day_idx": torch.randint(0, 288, (batch, seq_len)),
        "day_of_week_idx": torch.randint(0, 7, (batch, seq_len)),
    }
    pred = model.forward_reconstruct(features, mask)
    assert pred.shape == (batch, seq_len, num_sensors)
    loss = masked_reconstruction_loss(pred, traffic, mask)
    loss.backward()
    assert model.mask_token.grad is not None
    assert model.reconstruction_head.weight.grad is not None


def test_forward_reconstruct_fails_in_legacy_pipeline():
    model = TrafficTransformer(
        input_dim=5, num_features=5, hidden_dim=32, num_heads=4, num_layers=2,
        pred_len=3, model_pipeline="legacy",
    )
    with pytest.raises(RuntimeError, match="forward_reconstruct"):
        model.forward_reconstruct({}, torch.zeros(1, 1, 5, dtype=torch.bool))


def test_save_and_load_pretrained_encoder_roundtrip(tmp_path):
    model_a = TrafficTransformer(
        input_dim=4, num_features=4, hidden_dim=24, num_heads=4, num_layers=1,
        pred_len=2, model_pipeline="stae", seq_length=4,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
    )
    ckpt = tmp_path / "encoder.pth"
    save_pretrained_encoder(model_a, ckpt)

    model_b = TrafficTransformer(
        input_dim=4, num_features=4, hidden_dim=24, num_heads=4, num_layers=1,
        pred_len=2, model_pipeline="stae", seq_length=4,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
    )
    load_pretrained_encoder(model_b, ckpt)
    # Composer weights should now match.
    a_weight = model_a.stae_composer.input_proj.weight
    b_weight = model_b.stae_composer.input_proj.weight
    assert torch.equal(a_weight, b_weight)


def test_pretrain_subcommand_parses():
    from visu_predict.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["pretrain", "-c", "x.yaml", "-d", "y.csv"])
    assert args.command == "pretrain"
    assert str(args.config) == "x.yaml"
    assert str(args.data) == "y.csv"
