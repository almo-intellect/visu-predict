from __future__ import annotations

import pytest
import torch

from visu_predict.models.adaptive_graph import AdaptiveAdjacency
from visu_predict.models.moe import (
    AdaptiveExpert,
    IdentityExpert,
    STMoE,
    STMoEGate,
    _mix_with_adjacency,
)
from visu_predict.models.transformer import TrafficTransformer


def test_mix_with_adjacency_shape():
    x = torch.randn(2, 4, 5, 8)
    adj = torch.eye(5)
    out = _mix_with_adjacency(x, adj)
    assert out.shape == x.shape
    # With identity adjacency, output equals input
    assert torch.allclose(out, x)


def test_identity_expert_default_is_identity():
    e = IdentityExpert(num_sensors=5)
    x = torch.randn(1, 3, 5, 4)
    out = e(x)
    assert torch.allclose(out, x)


def test_identity_expert_set_adjacency():
    e = IdentityExpert(num_sensors=4)
    # Build an adjacency that swaps node 0 and 1
    adj = torch.eye(4)
    adj[0, 0], adj[0, 1] = 0.0, 1.0
    adj[1, 1], adj[1, 0] = 0.0, 1.0
    e.set_adjacency(adj)
    x = torch.arange(4 * 6, dtype=torch.float32).view(1, 1, 4, 6)
    out = e(x)
    assert torch.allclose(out[0, 0, 0], x[0, 0, 1])
    assert torch.allclose(out[0, 0, 1], x[0, 0, 0])


def test_identity_expert_rejects_wrong_shape():
    e = IdentityExpert(num_sensors=4)
    with pytest.raises(ValueError, match="shape"):
        e.set_adjacency(torch.eye(5))


def test_adaptive_expert_uses_shared_adjacency():
    adj = AdaptiveAdjacency(num_sensors=4, d_emb=4, init_scale=0.5)
    e = AdaptiveExpert(adj)
    x = torch.randn(1, 3, 4, 6)
    out = e(x)
    assert out.shape == x.shape


def test_stmoe_gate_softmax_sums_to_one():
    g = STMoEGate(d_model=8, num_experts=2)
    x = torch.randn(1, 3, 4, 8)
    gates = g(x)
    assert gates.shape == (1, 3, 4, 2)
    sums = gates.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_stmoe_forward_shape_and_aux_loss_set():
    adj = AdaptiveAdjacency(num_sensors=5, d_emb=4, init_scale=0.5)
    moe = STMoE(d_model=8, num_sensors=5, adaptive_adj=adj)
    x = torch.randn(1, 3, 5, 8)
    out = moe(x)
    assert out.shape == x.shape
    assert moe.last_aux_loss is not None
    assert moe.last_aux_loss.item() >= 0


def test_stmoe_gradient_flows_to_both_experts():
    adj = AdaptiveAdjacency(num_sensors=4, d_emb=4, init_scale=0.5)
    moe = STMoE(d_model=8, num_sensors=4, adaptive_adj=adj)
    moe.set_static_adjacency(torch.eye(4) * 0.5)  # non-trivial static adj
    x = torch.randn(1, 3, 4, 8, requires_grad=True)
    out = moe(x)
    out.sum().backward()
    # Gradient flows to the adaptive embeddings (E1, E2) via the AdaptiveExpert.
    assert adj.E1.grad is not None
    assert adj.E1.grad.abs().sum() > 0
    # Gradient flows to the gate's projection weights.
    assert moe.gate.proj[0].weight.grad is not None


def test_traffic_transformer_with_moe_forward():
    batch, seq_len, num_sensors = 1, 4, 4
    model = TrafficTransformer(
        input_dim=num_sensors, num_features=num_sensors,
        hidden_dim=24, num_heads=4, num_layers=1, pred_len=2,
        model_pipeline="stae", seq_length=seq_len,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
        use_adaptive_adjacency=True, adaptive_adj_dim=6,
        use_moe=True,
    )
    src = {
        "traffic": torch.randn(batch, seq_len, num_sensors),
        "time_of_day_idx": torch.randint(0, 288, (batch, seq_len)),
        "day_of_week_idx": torch.randint(0, 7, (batch, seq_len)),
    }
    out = model(src)
    assert out.shape == (batch, 2, num_sensors)
    aux = model.collect_moe_aux_loss()
    assert aux is not None
    assert aux.item() >= 0


def test_moe_requires_adaptive_adjacency():
    with pytest.raises(ValueError, match="use_adaptive_adjacency"):
        TrafficTransformer(
            input_dim=4, num_features=4, hidden_dim=24, num_heads=4, num_layers=1,
            pred_len=2, model_pipeline="stae", seq_length=4,
            d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
            use_moe=True,  # no adaptive adjacency
        )


def test_set_static_adjacency_no_op_when_moe_disabled():
    model = TrafficTransformer(
        input_dim=4, num_features=4, hidden_dim=24, num_heads=4, num_layers=1,
        pred_len=2, model_pipeline="stae", seq_length=4,
        d_input=6, d_tod=6, d_dow=6, d_adaptive=6, d_node=0,
    )
    # Should not raise.
    model.set_static_adjacency(torch.eye(4))
    assert model.collect_moe_aux_loss() is None
