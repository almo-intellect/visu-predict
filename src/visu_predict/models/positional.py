"""Sinusoidal positional encoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (sequence-length-agnostic at forward time)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50_000) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # [max_len, 1, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[seq_len, batch, d_model]`` tensor.
        """
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Input feature dim {x.size(-1)} doesn't match d_model={self.d_model}"
            )
        return self.dropout(x + self.pe[: x.size(0)])
