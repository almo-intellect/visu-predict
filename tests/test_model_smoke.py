from __future__ import annotations

import torch

from visu_predict.models.positional import PositionalEncoding
from visu_predict.models.transformer import TrafficTransformer


def test_positional_encoding_adds_signal():
    pe = PositionalEncoding(d_model=32, dropout=0.0, max_len=100)
    x = torch.zeros(10, 4, 32)
    out = pe(x)
    assert out.shape == x.shape
    assert torch.allclose(out, pe.pe[:10].expand_as(x))


def test_positional_encoding_rejects_odd_d_model():
    import pytest
    with pytest.raises(ValueError):
        PositionalEncoding(d_model=33)


def test_traffic_transformer_linear_decoder_forward():
    batch, seq_len, num_sensors = 2, 6, 5
    feature_dims = {"traffic": num_sensors, "concatenated": num_sensors}
    model = TrafficTransformer(
        input_dim=num_sensors,
        num_features=num_sensors,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        pred_len=3,
        decoder_type="linear",
        feature_dims=feature_dims,
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "concatenated": torch.randn(batch, seq_len, num_sensors),
    }
    out = model(src)
    assert out.shape == (batch, 3, num_sensors)


def test_traffic_transformer_mlp_decoder_forward():
    batch, seq_len, num_sensors = 2, 6, 5
    feature_dims = {"traffic": num_sensors, "concatenated": num_sensors}
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=32, num_heads=4, num_layers=2, pred_len=4,
        decoder_type="mlp", feature_dims=feature_dims,
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "concatenated": torch.randn(batch, seq_len, num_sensors),
    }
    out = model(src)
    assert out.shape == (batch, 4, num_sensors)


def test_freeze_layers_marks_params_non_trainable():
    model = TrafficTransformer(
        input_dim=5, num_features=5, hidden_dim=32, num_heads=4, num_layers=3,
        pred_len=3, feature_dims={"traffic": 5, "concatenated": 5},
    )
    model.freeze_layers(freeze_encoder=True, num_layers=2)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert any(n.startswith("embedding") for n in frozen)
    assert any(n.startswith("encoder.0") for n in frozen)
    assert any(n.startswith("encoder.1") for n in frozen)
    assert not any(n.startswith("encoder.2") for n in frozen)
