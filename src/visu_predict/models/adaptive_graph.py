"""Adaptive learnable adjacency.

Replaces hand-built or static graph adjacency matrices (loaded from ``.pkl``
files) with two learnable node-embedding tables ``E1`` and ``E2``. The
runtime adjacency is ``softmax(ReLU(E1 @ E2.T))``, computed each forward
pass; gradients flow back through the embeddings so the graph topology is
learned jointly with the task. This is the trick popularised by Graph
WaveNet, MTGNN, and AGCRN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAdjacency(nn.Module):
    """Learnable ``[N, N]`` adjacency from two ``[N, d_emb]`` embedding tables.

    Args:
        num_sensors: number of nodes ``N``.
        d_emb: dimensionality of the source/target embeddings (small, ~10).
        init_scale: standard deviation of the normal initialiser. The default
            ``0.01`` keeps logits small so the initial adjacency is close to
            uniform — the model learns sharper structure during training.
    """

    def __init__(self, num_sensors: int, d_emb: int = 10, init_scale: float = 0.01) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.d_emb = d_emb
        self.E1 = nn.Parameter(torch.randn(num_sensors, d_emb) * init_scale)
        self.E2 = nn.Parameter(torch.randn(num_sensors, d_emb) * init_scale)

    def adjacency(self) -> torch.Tensor:
        """Return the row-stochastic ``[N, N]`` adjacency matrix."""
        # softmax in fp32 to avoid NaNs under autocast on large N (e.g. PEMS-07).
        with torch.amp.autocast(device_type="cuda", enabled=False):
            logits = F.relu(self.E1.float() @ self.E2.float().T)
            return F.softmax(logits, dim=-1)

    def forward(self) -> torch.Tensor:
        return self.adjacency()

    def as_attention_bias(self, num_heads: int, eps: float = 1e-9) -> torch.Tensor:
        """Return an additive attention bias broadcasting to all heads.

        Shape: ``[1, num_heads, N, N]``. The bias is ``log(adj + eps)`` so it
        composes additively with attention logits before softmax — high-weight
        edges in the adaptive adjacency yield high attention scores.
        """
        adj = self.adjacency()
        bias = torch.log(adj + eps)
        return bias.unsqueeze(0).unsqueeze(0).expand(1, num_heads, -1, -1)
