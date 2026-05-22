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


def test_stae_pipeline_forward():
    batch, seq_len, num_sensors = 2, 6, 5
    # d_input + d_tod + d_dow + d_adaptive + d_node must equal hidden_dim
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=32, num_heads=4, num_layers=2, pred_len=3,
        decoder_type="linear",
        feature_dims={"traffic": num_sensors, "concatenated": num_sensors},
        model_pipeline="stae",
        steps_per_day=288,
        d_input=8, d_tod=8, d_dow=8, d_adaptive=8, d_node=0,
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "time_of_day_idx": torch.randint(0, 288, (batch, seq_len)),
        "day_of_week_idx": torch.randint(0, 7, (batch, seq_len)),
        "concatenated": torch.randn(batch, seq_len, num_sensors),
    }
    out = model(src)
    assert out.shape == (batch, 3, num_sensors)


def test_stae_pipeline_requires_index_keys():
    import pytest
    model = TrafficTransformer(
        input_dim=5, num_features=5, hidden_dim=32, num_heads=4, num_layers=2,
        pred_len=3, model_pipeline="stae",
        d_input=8, d_tod=8, d_dow=8, d_adaptive=8, d_node=0,
    )
    src = {"traffic": torch.randn(2, 6, 5), "concatenated": torch.randn(2, 6, 5)}
    with pytest.raises(KeyError, match="time_of_day_idx"):
        model(src)


def test_stae_pipeline_dim_sum_must_match_hidden_dim():
    import pytest
    with pytest.raises(ValueError, match="d_input"):
        TrafficTransformer(
            input_dim=5, num_features=5, hidden_dim=32, num_heads=4, num_layers=2,
            pred_len=3, model_pipeline="stae",
            d_input=8, d_tod=8, d_dow=8, d_adaptive=8, d_node=4,  # sums to 36, not 32
        )


def test_stae_adaptive_embedding_receives_gradient():
    model = TrafficTransformer(
        input_dim=4, num_features=4, hidden_dim=24, num_heads=4, num_layers=1,
        pred_len=2, model_pipeline="stae",
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
    )
    src = {
        "traffic": torch.randn(1, 3, 4),
        "time_of_day_idx": torch.randint(0, 288, (1, 3)),
        "day_of_week_idx": torch.randint(0, 7, (1, 3)),
    }
    out = model(src)
    out.sum().backward()
    adaptive_weight = model.stae_composer.adaptive_embed.weight
    assert adaptive_weight.grad is not None
    assert (adaptive_weight.grad != 0).any()


def test_legacy_pipeline_unchanged_default():
    # Default pipeline must remain "legacy" and produce the same model
    # construction as before this PR.
    model = TrafficTransformer(
        input_dim=5, num_features=5, hidden_dim=32, num_heads=4, num_layers=2,
        pred_len=3, feature_dims={"traffic": 5, "concatenated": 5},
    )
    assert model.model_pipeline == "legacy"
    assert model.stae_composer is None
    assert model.feature_attention is not None


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
