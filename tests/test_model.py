import numpy as np
import torch

from visu_predict.data import load_st_benchmark
from visu_predict.model import STTransformer, build_model, hop_distance_matrix

TINY = dict(num_temporal_layers=1, num_spatial_layers=1)


def _batch(syn, **kwargs):
    root, _ = syn
    data = load_st_benchmark(root, "SYN", batch_size=8, **kwargs)
    return data, next(iter(data.train))


def test_output_shape_with_graph_bias_and_exogenous_inputs(syn):
    data, b = _batch(syn)
    model = STTransformer(num_nodes=6, adj=data.adj, graph_bias=True, exo_dim=3, exo_embedding_dim=8, **TINY)
    out = model(b["x"], b["tod"], b["dow"], torch.randn(8, 12, 3))
    assert out.shape == (8, 12, 6)
    gb = model.graph_bias()
    assert gb.shape == (4, 6, 6) and gb.abs().sum() == 0       # starts as plain attention
    out.mean().backward()
    assert model.graph_bias.fwd.weight.grad.abs().sum() > 0


def test_history_lag_channels_are_accepted(syn):
    data, b = _batch(syn, history_lags=(288, 576))
    model = STTransformer(num_nodes=6, input_dim=data.input_dim, **TINY)
    assert model(b["x"], b["tod"], b["dow"]).shape == (8, 12, 6)


def test_default_configuration_matches_the_trained_models():
    """Same sizes as the published V19 runs (results.json "params")."""
    def count(model):
        return sum(p.numel() for p in model.parameters())

    assert STTransformer(num_nodes=207).model_dim == 152                        # STAEformer configuration
    assert count(STTransformer(num_nodes=207)) == 1_258_932                     # METR-LA
    assert count(STTransformer(num_nodes=325)) == 1_372_212                     # PEMS-BAY
    assert count(STTransformer(num_nodes=325, input_dim=3)) == 1_372_260        # PEMS-BAY + history lags


def test_hop_distances_on_a_chain():
    adj = np.eye(6, dtype=np.float32) + np.eye(6, k=1, dtype=np.float32)
    hops = hop_distance_matrix(adj, max_hops=6)
    assert hops[0, 1] == 1 and hops[0, 3] == 3
    assert hops[3, 0] == 7            # unreachable -> max_hops + 1


def test_padded_fused_attention_equals_the_explicit_path(syn):
    _, b = _batch(syn)
    model = STTransformer(num_nodes=6, **TINY).eval()   # head size 38 -> padded to 40 in SDPA
    with torch.no_grad():
        fused = model(b["x"], b["tod"], b["dow"])
        model.set_attention_capture(True)
        explicit = model(b["x"], b["tod"], b["dow"])
    assert torch.allclose(fused, explicit, atol=1e-5)


def test_adapt_to_graph_keeps_shared_weights(syn):
    data, b = _batch(syn)
    model = STTransformer(num_nodes=6, adj=data.adj, graph_bias=True, exo_dim=3, exo_embedding_dim=8, **TINY)
    with torch.no_grad():
        model.graph_bias.fwd.weight.fill_(0.5)
    shared = model.temporal_layers[0].attn.qkv.weight.clone()
    adj9 = np.eye(9, dtype=np.float32) + np.eye(9, k=1, dtype=np.float32)
    model.adapt_to_graph(9, adj=adj9, freeze_shared=True)
    out = model(torch.randn(2, 12, 9, 1), b["tod"][:2], b["dow"][:2], torch.randn(2, 12, 3))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert out.shape == (2, 12, 9)
    assert torch.equal(model.temporal_layers[0].attn.qkv.weight, shared)
    assert float(model.graph_bias.fwd.weight.mean().detach()) == 0.5
    assert trainable == {"adaptive_embedding", "graph_bias.fwd.weight", "graph_bias.bwd.weight"}
    assert model.config["num_nodes"] == 9


def test_build_model_restores_the_same_network(syn):
    data, b = _batch(syn)
    model = STTransformer(num_nodes=6, adj=data.adj, graph_bias=True, **TINY).eval()
    clone = build_model("STTransformer", model.config, adj=data.adj).eval()
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.equal(model(b["x"], b["tod"], b["dow"]), clone(b["x"], b["tod"], b["dow"]))
