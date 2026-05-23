"""TESTAM+-style mixture-of-experts spatial mixing for STAE.

Two experts share the same input ``[B, T, N, d]`` and produce the same-shaped
output:

- :class:`IdentityExpert`: spatial mixing via a fixed (e.g. distance-based)
  adjacency matrix. ``x' = A_static @ x`` per timestep — a known good prior.
- :class:`AdaptiveExpert`: spatial mixing via an :class:`AdaptiveAdjacency`
  whose embeddings are learned with the rest of the model — captures
  data-driven structure.

A small gating MLP scores each ``(b, t, n)`` cell with two logits, softmax-
normalised. The block's output is the gate-weighted sum of the two experts'
outputs. A standard MoE load-balancing loss encourages the gate distribution
not to collapse onto a single expert.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from visu_predict.models.adaptive_graph import AdaptiveAdjacency


def _mix_with_adjacency(x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """Apply spatial mixing ``A @ x`` to ``[B, T, N, d]``.

    Args:
        x: ``[B, T, N, d]``.
        adj: ``[N, N]`` adjacency matrix (row-stochastic recommended).

    Returns:
        ``[B, T, N, d]`` with each node replaced by an adjacency-weighted
        mixture of its neighbours' features.
    """
    bsz, seq_len, num_sensors, d = x.shape
    flat = x.reshape(bsz * seq_len, num_sensors, d)
    mixed = torch.matmul(adj, flat)
    return mixed.view(bsz, seq_len, num_sensors, d)


class IdentityExpert(nn.Module):
    """Spatial mixing via a fixed adjacency buffer."""

    def __init__(self, num_sensors: int) -> None:
        super().__init__()
        self.register_buffer("static_adj", torch.eye(num_sensors))

    def set_adjacency(self, adj: torch.Tensor) -> None:
        if adj.shape != self.static_adj.shape:
            raise ValueError(
                f"Static adjacency shape {adj.shape} does not match buffer "
                f"shape {self.static_adj.shape}"
            )
        self.static_adj = adj.to(self.static_adj.device, dtype=self.static_adj.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _mix_with_adjacency(x, self.static_adj)


class AdaptiveExpert(nn.Module):
    """Spatial mixing via a shared :class:`AdaptiveAdjacency` (recomputed)."""

    def __init__(self, adaptive_adj: AdaptiveAdjacency) -> None:
        super().__init__()
        # Reference, not a sub-module — the AdaptiveAdjacency is owned by the
        # parent TrafficTransformer so it can be shared across MoE blocks.
        self._adaptive_adj = [adaptive_adj]  # avoid re-registering as submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adj = self._adaptive_adj[0].adjacency()
        return _mix_with_adjacency(x, adj)


class STMoEGate(nn.Module):
    """Per-cell gate producing routing logits of shape ``[B, T, N, num_experts]``."""

    def __init__(self, d_model: int, num_experts: int = 2, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.proj(x), dim=-1)


class STMoE(nn.Module):
    """Mixture of identity + adaptive spatial experts with load-balancing loss.

    Replaces :class:`SpatialBlock` in :class:`STAttnStack` when MoE is enabled.
    Drop-in interface: ``forward(x, attn_bias=None)`` accepts the same kwargs
    so :class:`STAttnStack` doesn't care which block flavour is plugged in.
    """

    def __init__(
        self,
        d_model: int,
        num_sensors: int,
        adaptive_adj: AdaptiveAdjacency,
    ) -> None:
        super().__init__()
        self.gate = STMoEGate(d_model, num_experts=2)
        self.identity_expert = IdentityExpert(num_sensors)
        self.adaptive_expert = AdaptiveExpert(adaptive_adj)
        self.last_aux_loss: torch.Tensor | None = None
        # For visualisation parity with SpatialBlock.
        self.attn_weights: torch.Tensor | None = None

    def set_static_adjacency(self, adj: torch.Tensor) -> None:
        """Inject the fixed adjacency used by the identity expert."""
        self.identity_expert.set_adjacency(adj)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        # ``attn_bias`` is ignored — kept in the signature for SpatialBlock parity.
        del attn_bias
        gates = self.gate(x)  # [B, T, N, 2]
        identity_out = self.identity_expert(x)
        adaptive_out = self.adaptive_expert(x)
        out = gates[..., 0:1] * identity_out + gates[..., 1:2] * adaptive_out

        # Load-balancing loss: encourage mean gate distribution to be uniform.
        mean_gates = gates.mean(dim=(0, 1, 2))  # [num_experts]
        target = torch.full_like(mean_gates, 1.0 / mean_gates.numel())
        self.last_aux_loss = ((mean_gates - target) ** 2).sum()
        self.attn_weights = gates.detach()
        return out
