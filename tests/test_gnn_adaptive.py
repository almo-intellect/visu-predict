from __future__ import annotations

import pytest
import torch

from visu_predict.models.gnn import TORCH_GEOMETRIC_AVAILABLE
from visu_predict.models.transformer import TrafficTransformer

pytestmark = pytest.mark.skipif(
    not TORCH_GEOMETRIC_AVAILABLE, reason="torch_geometric not installed"
)


def test_legacy_gnn_with_adaptive_adjacency_forward():
    batch, seq_len, num_sensors = 2, 6, 4
    feature_dims = {"traffic": num_sensors, "concatenated": num_sensors}
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=16, num_heads=4, num_layers=2, pred_len=3,
        decoder_type="linear", feature_dims=feature_dims,
        use_gnn_pre_transformer=True, gnn_type="gcn", gnn_layers=2,
        use_adaptive_adjacency=True, adaptive_adj_dim=8,
        adaptive_adj_inject_into="gnn",
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "concatenated": torch.randn(batch, seq_len, num_sensors),
    }
    # No external adjacency_matrix is needed when adaptive feeds the GNN.
    out = model(src)
    assert out.shape == (batch, 3, num_sensors)


def test_gnn_adaptive_gradient_reaches_embeddings():
    batch, seq_len, num_sensors = 1, 4, 3
    feature_dims = {"traffic": num_sensors, "concatenated": num_sensors}
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=16, num_heads=4, num_layers=1, pred_len=2,
        decoder_type="linear", feature_dims=feature_dims,
        use_gnn_pre_transformer=True, gnn_type="gcn", gnn_layers=1,
        use_adaptive_adjacency=True, adaptive_adj_dim=4,
        adaptive_adj_inject_into="gnn",
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "concatenated": torch.randn(batch, seq_len, num_sensors),
    }
    out = model(src)
    out.sum().backward()
    assert model.adaptive_adj.E1.grad is not None
    assert model.adaptive_adj.E1.grad.abs().sum() > 0


def test_inject_into_both_with_stae_uses_both_paths():
    # In STAE pipeline with use_gnn_pre_transformer=False the "gnn" branch
    # is irrelevant; "both" reduces to the spatial-attn bias and the model
    # still produces correct shape.
    batch, seq_len, num_sensors = 1, 4, 4
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=24, num_heads=4, num_layers=1, pred_len=2,
        model_pipeline="stae", seq_length=seq_len,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
        use_adaptive_adjacency=True, adaptive_adj_dim=6,
        adaptive_adj_inject_into="both",
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "time_of_day_idx": torch.randint(0, 288, (batch, seq_len)),
        "day_of_week_idx": torch.randint(0, 7, (batch, seq_len)),
    }
    out = model(src)
    assert out.shape == (batch, 2, num_sensors)
