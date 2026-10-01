"""
Node-level spatio-temporal transformer (STTransformer).

The earlier ``TrafficTransformer`` (V18, kept in :mod:`visu_predict.legacy`)
embeds all N sensors of a timestep into a single token, attends over the 12
timestep tokens, mean-pools over time and maps one vector to all 12 x N outputs.
It therefore has no notion of sensor identity, shares no weights across sensors,
cannot model which sensors influence which, and its input/output layers are tied
to N (so transfer to a different network needs re-heading).

STTransformer instead treats every (timestep, sensor) pair as a token:

    token(t, n) = [ value proj | time-of-day emb | day-type emb |
                    node emb (opt) | spatio-temporal adaptive emb |
                    exogenous emb (opt, e.g. weather, broadcast over nodes) ]

followed by temporal self-attention (over the 12 steps of each sensor) and
spatial self-attention (over all sensors at each step), and a "mixed"
output projection that maps each sensor's (T x D) history to its 12-step
forecast. This is the design family of STAEformer (Liu et al., CIKM 2023),
re-implemented here with two optional extensions:

* ``graph_bias``: Graphormer-style structural prior for spatial attention -
  a learnable per-head bias indexed by shortest-path hop distance on the
  road graph (both directions), initialised to zero so the model starts as
  plain full attention and learns how much topology to use.
* ``exo_dim``: exogenous city-level covariates (weather, events) embedded per
  timestep and broadcast to all sensors.

All weights except the node / adaptive embeddings are shared across sensors,
so a trained backbone can be moved to a network with a different number of
sensors by re-initialising only those embeddings (:meth:`STTransformer.adapt_to_graph`).
"""

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Building blocks
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Self-attention over the second-to-last axis using fused SDPA kernels.

    PyTorch's fused (flash / memory-efficient) kernels need the head size to be
    a multiple of 8; otherwise SDPA silently falls back to the slow "math"
    path that materialises every L x L score matrix (325 x 325 per head and
    timestep for PEMS-BAY). Odd head sizes (STAEformer: 152 / 4 = 38) are
    therefore zero-padded to the next multiple of 8 with the original
    1/sqrt(head_dim) scaling - mathematically identical output.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pad = (-self.head_dim) % 8
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.store_attention = False
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        *lead, L, D = x.shape
        x2 = x.reshape(-1, L, D)
        qkv = self.qkv(x2).view(x2.shape[0], L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)          # (B', H, L, dh)
        if bias is not None:
            bias = bias.to(q.dtype)
        if self.store_attention:
            scores = (q @ k.transpose(-1, -2)) * self.scale
            if bias is not None:
                scores = scores + bias
            attn = scores.softmax(dim=-1)
            self.last_attention = attn.detach()
            out = attn @ v
        else:
            if self.pad:
                q, k, v = (F.pad(t, (0, self.pad)) for t in (q, k, v))
            if bias is not None:
                bias = bias.expand(q.shape[0], -1, -1, -1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
            if self.pad:
                out = out[..., :self.head_dim]
        out = out.transpose(1, 2).reshape(*lead, L, D)
        return self.proj(out)


class STBlock(nn.Module):
    """Transformer layer applied along one axis of a (B, T, N, D) tensor."""

    def __init__(self, dim: int, ff_dim: int, num_heads: int,
                 dropout: float = 0.1, norm_first: bool = False) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, axis: int, bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x.transpose(axis, -2)
        if self.norm_first:
            x = x + self.drop1(self.attn(self.ln1(x), bias))
            x = x + self.drop2(self.ff(self.ln2(x)))
        else:
            x = self.ln1(x + self.drop1(self.attn(x, bias)))
            x = self.ln2(x + self.drop2(self.ff(x)))
        return x.transpose(axis, -2)


def hop_distance_matrix(adj: np.ndarray, max_hops: int) -> np.ndarray:
    """Directed shortest-path hop counts, clipped to ``max_hops + 1``."""
    from scipy.sparse.csgraph import shortest_path

    a = (np.asarray(adj) > 0).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    d = shortest_path(a, directed=True, unweighted=True)
    d[~np.isfinite(d)] = max_hops + 1
    return np.minimum(d, max_hops + 1).astype(np.int64)


class GraphDistanceBias(nn.Module):
    """Per-head attention bias from road-graph hop distance (Graphormer-style).

    ``bias[h, i, j] = fwd[h, hops(i -> j)] + bwd[h, hops(j -> i)]``. Both tables
    start at zero, so at initialisation spatial attention is unconstrained.
    """

    def __init__(self, adj: np.ndarray, num_heads: int, max_hops: int = 6) -> None:
        super().__init__()
        hops = torch.from_numpy(hop_distance_matrix(adj, max_hops))
        self.register_buffer("fwd_idx", hops, persistent=False)
        self.register_buffer("bwd_idx", hops.t().contiguous(), persistent=False)
        self.fwd = nn.Embedding(max_hops + 2, num_heads)
        self.bwd = nn.Embedding(max_hops + 2, num_heads)
        nn.init.zeros_(self.fwd.weight)
        nn.init.zeros_(self.bwd.weight)

    def forward(self) -> torch.Tensor:
        b = self.fwd(self.fwd_idx) + self.bwd(self.bwd_idx)      # (N, N, H)
        return b.permute(2, 0, 1).contiguous()                   # (H, N, N)


# =============================================================================
# STTransformer
# =============================================================================

class STTransformer(nn.Module):
    """Node-level spatio-temporal transformer.

    Inputs (as produced by :class:`visu_predict.data.WindowBatcher`):
        x   (B, T, N, input_dim)  scaled traffic
        tod (B, T)                time-of-day slot index
        dow (B, T)                day-type index
        exo (B, T, exo_dim)       optional exogenous covariates
    Output: (B, out_steps, N) in the scaled space (inverse-transform outside).

    The constructor arguments (except ``adj``) are kept in ``self.config`` and
    stored in checkpoints, so :func:`build_model` can rebuild the model.
    """

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        num_day_types: int = 7,
        input_dim: int = 1,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        node_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        exo_dim: int = 0,
        exo_embedding_dim: int = 0,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_temporal_layers: int = 3,
        num_spatial_layers: int = 3,
        dropout: float = 0.1,
        norm_first: bool = False,
        adj: np.ndarray | None = None,
        graph_bias: bool = False,
        graph_max_hops: int = 6,
    ) -> None:
        super().__init__()
        self.config: dict[str, Any] = {
            "num_nodes": num_nodes, "in_steps": in_steps, "out_steps": out_steps,
            "steps_per_day": steps_per_day, "num_day_types": num_day_types, "input_dim": input_dim,
            "output_dim": output_dim, "input_embedding_dim": input_embedding_dim,
            "tod_embedding_dim": tod_embedding_dim, "dow_embedding_dim": dow_embedding_dim,
            "node_embedding_dim": node_embedding_dim, "adaptive_embedding_dim": adaptive_embedding_dim,
            "exo_dim": exo_dim, "exo_embedding_dim": exo_embedding_dim,
            "feed_forward_dim": feed_forward_dim, "num_heads": num_heads,
            "num_temporal_layers": num_temporal_layers, "num_spatial_layers": num_spatial_layers,
            "dropout": dropout, "norm_first": norm_first, "graph_bias": graph_bias,
            "graph_max_hops": graph_max_hops,
        }
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.output_dim = output_dim
        self.steps_per_day = steps_per_day

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.tod_embedding = (nn.Embedding(steps_per_day, tod_embedding_dim)
                              if tod_embedding_dim > 0 else None)
        self.dow_embedding = (nn.Embedding(num_day_types, dow_embedding_dim)
                              if dow_embedding_dim > 0 else None)
        self.node_embedding = None
        if node_embedding_dim > 0:
            self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
            nn.init.xavier_uniform_(self.node_embedding)
        self.adaptive_embedding = None
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)
        self.exo_proj = None
        if exo_dim > 0 and exo_embedding_dim > 0:
            self.exo_proj = nn.Sequential(
                nn.Linear(exo_dim, exo_embedding_dim), nn.ReLU(),
                nn.Linear(exo_embedding_dim, exo_embedding_dim),
            )
        else:
            exo_embedding_dim = 0

        self.model_dim = (input_embedding_dim + tod_embedding_dim + dow_embedding_dim
                          + node_embedding_dim + adaptive_embedding_dim + exo_embedding_dim)

        self.temporal_layers = nn.ModuleList([
            STBlock(self.model_dim, feed_forward_dim, num_heads, dropout, norm_first)
            for _ in range(num_temporal_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            STBlock(self.model_dim, feed_forward_dim, num_heads, dropout, norm_first)
            for _ in range(num_spatial_layers)
        ])
        self.graph_bias = None
        if graph_bias:
            if adj is None:
                raise ValueError("graph_bias=True requires an adjacency matrix")
            self.graph_bias = GraphDistanceBias(adj, num_heads, graph_max_hops)
        self.final_norm = nn.LayerNorm(self.model_dim) if norm_first else nn.Identity()
        self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)

    # ------------------------------------------------------------------
    def embed(self, x: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor,
              exo: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, _ = x.shape
        feats = [self.input_proj(x)]
        if self.tod_embedding is not None:
            feats.append(self.tod_embedding(tod)[:, :, None, :].expand(B, T, N, -1))
        if self.dow_embedding is not None:
            feats.append(self.dow_embedding(dow)[:, :, None, :].expand(B, T, N, -1))
        if self.node_embedding is not None:
            feats.append(self.node_embedding.expand(B, T, N, -1))
        if self.adaptive_embedding is not None:
            feats.append(self.adaptive_embedding.expand(B, T, N, -1))
        if self.exo_proj is not None:
            if exo is None:
                raise ValueError("model was built with exo_dim > 0 but no exo input given")
            feats.append(self.exo_proj(exo)[:, :, None, :].expand(B, T, N, -1))
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor,
                exo: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, _ = x.shape
        h = self.embed(x, tod, dow, exo)                          # (B, T, N, D)
        for layer in self.temporal_layers:
            h = layer(h, axis=1)
        bias = self.graph_bias() if self.graph_bias is not None else None
        for layer in self.spatial_layers:
            h = layer(h, axis=2, bias=bias)
        h = self.final_norm(h)
        out = h.transpose(1, 2).reshape(B, N, T * self.model_dim)
        # Output head in fp32 even under bf16/fp16 autocast: predictions are
        # scaled back by std (~10-20 mph), so bf16 rounding (~0.4 %) would add
        # visible noise to MAE.
        with torch.autocast(device_type=out.device.type, enabled=False):
            out = self.output_proj(out.float())
        out = out.view(B, N, self.out_steps, self.output_dim)
        return out.transpose(1, 2).squeeze(-1)                   # (B, out, N)

    def set_attention_capture(self, enabled: bool = True) -> None:
        for m in self.modules():
            if isinstance(m, MultiHeadSelfAttention):
                m.store_attention = bool(enabled)

    # ------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------
    def adapt_to_graph(self, num_nodes: int, adj: np.ndarray | None = None,
                       freeze_shared: bool = False) -> "STTransformer":
        """Prepare a trained model for a different road network (transfer).

        Only the sensor-specific parameters depend on N: the node embedding
        and the spatio-temporal adaptive embedding are re-initialised for the
        new sensors. Everything else - value/time embeddings, all attention
        and feed-forward layers, the output head and the hop-distance bias
        tables (indexed by graph distance, not by sensor) - is kept. With
        ``freeze_shared=True`` only the new sensor embeddings (and the graph
        bias tables) remain trainable, for few-shot fine-tuning.
        """
        device = self.output_proj.weight.device
        self.num_nodes = num_nodes
        self.config["num_nodes"] = num_nodes
        if self.node_embedding is not None:
            d = self.node_embedding.shape[-1]
            self.node_embedding = nn.Parameter(torch.empty(num_nodes, d, device=device))
            nn.init.xavier_uniform_(self.node_embedding)
        if self.adaptive_embedding is not None:
            d = self.adaptive_embedding.shape[-1]
            self.adaptive_embedding = nn.Parameter(
                torch.empty(self.in_steps, num_nodes, d, device=device))
            nn.init.xavier_uniform_(self.adaptive_embedding)
        if self.graph_bias is not None:
            if adj is None:
                raise ValueError("model uses graph_bias: pass the target adjacency matrix")
            old = self.graph_bias
            max_hops = old.fwd.num_embeddings - 2
            new = GraphDistanceBias(adj, old.fwd.embedding_dim, max_hops).to(device)
            new.fwd.weight.data.copy_(old.fwd.weight.data)
            new.bwd.weight.data.copy_(old.bwd.weight.data)
            self.graph_bias = new
        if freeze_shared:
            keep = {id(p) for p in (self.node_embedding, self.adaptive_embedding) if p is not None}
            if self.graph_bias is not None:
                keep |= {id(p) for p in self.graph_bias.parameters()}
            for p in self.parameters():
                p.requires_grad = id(p) in keep
        return self


def build_model(model_class: str, config: dict[str, Any], adj: np.ndarray | None = None) -> nn.Module:
    """Rebuild a model from the ``model_class`` / ``model_config`` stored in a checkpoint."""
    if model_class == "STTransformer":
        return STTransformer(**config, adj=adj)
    if model_class == "LegacyTransformerAdapter":
        from .legacy.adapter import LegacyTransformerAdapter

        return LegacyTransformerAdapter(**config)
    raise ValueError(f"unknown model class {model_class!r}")
