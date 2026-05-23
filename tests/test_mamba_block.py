from __future__ import annotations

import pytest
import torch

from visu_predict.models.mamba_block import MAMBA_AVAILABLE
from visu_predict.models.transformer import TrafficTransformer


def test_mamba_block_unavailable_raises_on_import_use():
    if MAMBA_AVAILABLE:
        pytest.skip("mamba-ssm is installed; this test only covers the missing-dep path")
    with pytest.raises(ImportError, match="mamba-ssm"):
        TrafficTransformer(
            input_dim=4, num_features=4, hidden_dim=24, num_heads=4, num_layers=1,
            pred_len=2, model_pipeline="stae", seq_length=4,
            d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
            temporal_block_type="mamba",
        )


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
def test_mamba_block_forward_shape():
    from visu_predict.models.mamba_block import MambaTemporalBlock
    block = MambaTemporalBlock(d_model=16)
    x = torch.randn(1, 6, 3, 16).cuda()  # mamba requires CUDA
    out = block(x)
    assert out.shape == x.shape


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
def test_stae_mamba_pipeline_forward():
    batch, seq_len, num_sensors = 1, 6, 4
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=24, num_heads=4, num_layers=1, pred_len=3,
        model_pipeline="stae", seq_length=seq_len,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
        temporal_block_type="mamba",
    ).cuda()
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors).cuda(),
        "time_of_day_idx": torch.randint(0, 288, (batch, seq_len)).cuda(),
        "day_of_week_idx": torch.randint(0, 7, (batch, seq_len)).cuda(),
    }
    out = model(src)
    assert out.shape == (batch, 3, num_sensors)
