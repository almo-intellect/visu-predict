"""Spatial / temporal transformer blocks operating on ``[B, T, N, d]`` tensors.

These blocks keep the sensor axis ``N`` explicit so attention can be applied
along either the time axis (``TemporalBlock`` — attends across ``T`` per
sensor) or the sensor axis (``SpatialBlock`` — attends across ``N`` per
timestep). The two are alternated by :class:`STAttnStack` to give each layer
a turn at both kinds of mixing, in the STAEformer / STPFormer style.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """Transformer encoder layer attending over the time axis.

    Input/output shape: ``[B, T, N, d]``. Internally reshapes to
    ``[B*N, T, d]`` so each sensor attends over its own temporal sequence.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, ff_multiplier: int = 4) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, num_sensors, d = x.shape
        # [B, T, N, d] -> [B, N, T, d] -> [B*N, T, d]
        x = x.permute(0, 2, 1, 3).reshape(bsz * num_sensors, seq_len, d)
        x = self.layer(x)
        # back to [B, T, N, d]
        return x.view(bsz, num_sensors, seq_len, d).permute(0, 2, 1, 3).contiguous()


class SpatialBlock(nn.Module):
    """Hand-rolled transformer encoder layer attending over the sensor axis.

    Input/output shape: ``[B, T, N, d]``. Internally reshapes to
    ``[B*T, N, d]`` so each timestep attends across all sensors. Implemented
    explicitly (not via ``nn.TransformerEncoderLayer``) so a future PR can
    inject an attention bias from a learnable adaptive adjacency matrix.

    Args:
        attn_bias: optional ``[1, num_heads, N, N]`` or ``[B*T, num_heads, N, N]``
            tensor added to the attention logits. ``None`` means standard
            self-attention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, ff_multiplier: int = 4) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, num_sensors, d = x.shape
        # [B, T, N, d] -> [B*T, N, d]
        x_flat = x.reshape(bsz * seq_len, num_sensors, d)

        # Pre-norm self-attention block
        normed = self.norm1(x_flat)
        qkv = self.qkv_proj(normed).reshape(
            bsz * seq_len, num_sensors, 3, self.num_heads, self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)  # each [B*T, N, h, head_dim]
        q = q.transpose(1, 2)  # [B*T, h, N, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B*T, h, N, N]
        if attn_bias is not None:
            logits = logits + attn_bias
        attn = F.softmax(logits, dim=-1)
        self.attn_weights = attn.detach()
        attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, v)  # [B*T, h, N, head_dim]
        ctx = ctx.transpose(1, 2).reshape(bsz * seq_len, num_sensors, d)
        x_flat = x_flat + self.dropout1(self.out_proj(ctx))

        # Feed-forward block
        x_flat = x_flat + self.dropout2(self.ff(self.norm2(x_flat)))

        return x_flat.view(bsz, seq_len, num_sensors, d)


class STAttnStack(nn.Module):
    """Alternating spatial ↔ temporal transformer stack on ``[B, T, N, d]``.

    For each of ``num_layers`` outer layers, the stack applies the two
    sub-blocks in the order given by ``interleave_order`` (``"TS"`` for
    temporal-then-spatial, ``"ST"`` for the reverse). Total sub-blocks:
    ``2 * num_layers``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        interleave_order: str = "TS",
    ) -> None:
        super().__init__()
        if interleave_order not in ("TS", "ST"):
            raise ValueError(f"interleave_order must be 'TS' or 'ST', got {interleave_order!r}")
        self.interleave_order = interleave_order
        self.temporal_blocks = nn.ModuleList(
            TemporalBlock(d_model, num_heads, dropout, ff_multiplier) for _ in range(num_layers)
        )
        self.spatial_blocks = nn.ModuleList(
            SpatialBlock(d_model, num_heads, dropout, ff_multiplier) for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor, spatial_attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        for temporal, spatial in zip(self.temporal_blocks, self.spatial_blocks, strict=True):
            if self.interleave_order == "TS":
                x = temporal(x)
                x = spatial(x, attn_bias=spatial_attn_bias)
            else:
                x = spatial(x, attn_bias=spatial_attn_bias)
                x = temporal(x)
        return x
