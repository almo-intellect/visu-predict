from __future__ import annotations

import pytest
import torch

from visu_predict.models.patching import PatchEmbed
from visu_predict.models.transformer import TrafficTransformer


def test_patch_count_math_non_overlapping():
    p = PatchEmbed(d_model=8, patch_length=4)
    assert p.num_patches(12) == 3
    assert p.num_patches(16) == 4


def test_patch_count_math_strided():
    p = PatchEmbed(d_model=8, patch_length=4, patch_stride=2)
    # (12 - 4) // 2 + 1 = 5
    assert p.num_patches(12) == 5


def test_patch_embed_shape_non_overlapping():
    p = PatchEmbed(d_model=8, patch_length=3)
    x = torch.randn(2, 12, 5, 8)
    out = p(x)
    assert out.shape == (2, 4, 5, 8)


def test_patch_embed_d_out_override():
    p = PatchEmbed(d_model=8, patch_length=4, d_out=16)
    x = torch.randn(1, 8, 3, 8)
    out = p(x)
    assert out.shape == (1, 2, 3, 16)


def test_patch_embed_rejects_d_model_mismatch():
    p = PatchEmbed(d_model=8, patch_length=4)
    with pytest.raises(ValueError, match="channel dim"):
        p(torch.randn(1, 8, 3, 10))


def test_patch_embed_rejects_too_short_sequence():
    p = PatchEmbed(d_model=8, patch_length=8)
    with pytest.raises(ValueError, match="too short"):
        p(torch.randn(1, 4, 3, 8))


def test_stae_pipeline_with_patching_forward():
    batch, seq_len, num_sensors = 1, 12, 4
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=24, num_heads=4, num_layers=1, pred_len=3,
        model_pipeline="stae", seq_length=seq_len,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
        use_temporal_patching=True, patch_length=3,
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "time_of_day_idx": torch.randint(0, 288, (batch, seq_len)),
        "day_of_week_idx": torch.randint(0, 7, (batch, seq_len)),
    }
    out = model(src)
    assert out.shape == (batch, 3, num_sensors)


def test_stae_pipeline_patching_plus_adaptive_adjacency():
    batch, seq_len, num_sensors = 1, 8, 4
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=24, num_heads=4, num_layers=1, pred_len=2,
        model_pipeline="stae", seq_length=seq_len,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
        use_temporal_patching=True, patch_length=4,
        use_adaptive_adjacency=True, adaptive_adj_dim=8,
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "time_of_day_idx": torch.randint(0, 288, (batch, seq_len)),
        "day_of_week_idx": torch.randint(0, 7, (batch, seq_len)),
    }
    out = model(src)
    out.sum().backward()
    assert out.shape == (batch, 2, num_sensors)
    assert model.adaptive_adj.E1.grad is not None
