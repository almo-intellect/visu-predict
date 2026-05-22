from __future__ import annotations

import torch

from visu_predict.models.adaptive_graph import AdaptiveAdjacency
from visu_predict.models.transformer import TrafficTransformer


def test_adjacency_shape_and_rows_sum_to_one():
    adj_mod = AdaptiveAdjacency(num_sensors=8, d_emb=4)
    adj = adj_mod.adjacency()
    assert adj.shape == (8, 8)
    row_sums = adj.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(8), atol=1e-5)


def test_adjacency_gradient_flows_to_embeddings():
    # Use a larger init so ReLU rarely zeros everything out; checks grad exists
    # rather than asserting any element is non-zero (which is flaky with tiny inits).
    torch.manual_seed(0)
    adj_mod = AdaptiveAdjacency(num_sensors=6, d_emb=4, init_scale=0.5)
    target = torch.randn(6, 6)
    loss = ((adj_mod.adjacency() - target) ** 2).mean()
    loss.backward()
    assert adj_mod.E1.grad is not None
    assert adj_mod.E2.grad is not None
    assert adj_mod.E1.grad.abs().sum() > 0
    assert adj_mod.E2.grad.abs().sum() > 0


def test_as_attention_bias_shape_and_broadcasts():
    adj_mod = AdaptiveAdjacency(num_sensors=5, d_emb=4)
    bias = adj_mod.as_attention_bias(num_heads=4)
    assert bias.shape == (1, 4, 5, 5)


def test_stae_pipeline_with_adaptive_adjacency_forward():
    batch, seq_len, num_sensors = 1, 4, 6
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=24, num_heads=4, num_layers=1, pred_len=3,
        model_pipeline="stae", seq_length=seq_len,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
        use_adaptive_adjacency=True, adaptive_adj_dim=8,
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "time_of_day_idx": torch.randint(0, 288, (batch, seq_len)),
        "day_of_week_idx": torch.randint(0, 7, (batch, seq_len)),
    }
    out = model(src)
    assert out.shape == (batch, 3, num_sensors)


def test_stae_pipeline_adaptive_adjacency_gets_gradient():
    batch, seq_len, num_sensors = 1, 3, 4
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=24, num_heads=4, num_layers=1, pred_len=2,
        model_pipeline="stae", seq_length=seq_len,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
        use_adaptive_adjacency=True, adaptive_adj_dim=8,
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "time_of_day_idx": torch.randint(0, 288, (batch, seq_len)),
        "day_of_week_idx": torch.randint(0, 7, (batch, seq_len)),
    }
    out = model(src)
    out.sum().backward()
    assert model.adaptive_adj.E1.grad is not None
    assert (model.adaptive_adj.E1.grad != 0).any()


def test_adaptive_adjacency_lives_in_both_pipelines():
    # As of PR #5, AdaptiveAdjacency is no longer STAE-exclusive — it can
    # feed the GNN encoder in the legacy pipeline too. Whether it's actually
    # *consumed* depends on adaptive_adj_inject_into and the pipeline.
    model = TrafficTransformer(
        input_dim=5, num_features=5, hidden_dim=32, num_heads=4, num_layers=2,
        pred_len=3, model_pipeline="legacy",
        use_adaptive_adjacency=True, adaptive_adj_inject_into="gnn",
    )
    assert model.adaptive_adj is not None
    assert model.adaptive_adj_inject_into == "gnn"


def test_adaptive_adjacency_off_by_default():
    model = TrafficTransformer(
        input_dim=5, num_features=5, hidden_dim=32, num_heads=4, num_layers=2,
        pred_len=3, model_pipeline="legacy",
    )
    assert model.adaptive_adj is None
    assert model.use_adaptive_adjacency is False
