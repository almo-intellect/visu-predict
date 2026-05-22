"""STAEformer-style embedding components.

These modules compose per-cell embeddings of shape ``[B, T, N, d_model]`` from
traffic values, discrete time-of-day / day-of-week indices, a learnable
spatio-temporal adaptive table, and a per-sensor identity vector. The
adaptive embedding is the central trick of STAEformer (CIKM'23) — a learnable
``[steps_per_day, num_sensors, d_adp]`` parameter table indexed by the
``(time-of-day, sensor)`` pair, which captures intrinsic periodic structure
that vanilla positional encoding cannot represent.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveEmbedding(nn.Module):
    """Learnable ``[steps_per_day, num_sensors, d_adp]`` parameter table.

    Indexing by ``tod_idx`` returns ``[B, T, num_sensors, d_adp]``.
    """

    def __init__(self, steps_per_day: int, num_sensors: int, d_adp: int) -> None:
        super().__init__()
        self.steps_per_day = steps_per_day
        self.num_sensors = num_sensors
        self.d_adp = d_adp
        self.weight = nn.Parameter(torch.empty(steps_per_day, num_sensors, d_adp))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, tod_idx: torch.Tensor) -> torch.Tensor:
        """Look up ``self.weight[tod_idx]``.

        Args:
            tod_idx: integer tensor of shape ``[B, T]`` with values in
                ``[0, steps_per_day)``.

        Returns:
            tensor of shape ``[B, T, num_sensors, d_adp]``.
        """
        return self.weight[tod_idx]


class TimeOfDayEmbedding(nn.Embedding):
    """Per-timestep embedding indexed by ``time_of_day_idx`` ∈ ``[0, steps_per_day)``."""

    def __init__(self, steps_per_day: int, d_tod: int) -> None:
        super().__init__(num_embeddings=steps_per_day, embedding_dim=d_tod)


class DayOfWeekEmbedding(nn.Embedding):
    """Per-timestep embedding indexed by ``day_of_week_idx`` ∈ ``[0, 7)``."""

    def __init__(self, d_dow: int) -> None:
        super().__init__(num_embeddings=7, embedding_dim=d_dow)


class NodeEmbedding(nn.Module):
    """Per-sensor identity vector ``[num_sensors, d_node]`` broadcast over time."""

    def __init__(self, num_sensors: int, d_node: int) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.d_node = d_node
        self.weight = nn.Parameter(torch.empty(num_sensors, d_node))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, batch_size: int, seq_length: int) -> torch.Tensor:
        """Return ``[batch_size, seq_length, num_sensors, d_node]``."""
        return self.weight.expand(batch_size, seq_length, -1, -1)


class STAEInputComposer(nn.Module):
    """Compose per-cell embeddings of shape ``[B, T, num_sensors, d_model]``.

    Inputs:
        traffic: ``[B, T, num_sensors]`` float
        tod_idx: ``[B, T]`` int64 (time-of-day bucket index)
        dow_idx: ``[B, T]`` int64 (day-of-week index, 0..6)

    The channel composition is:

        ``[input_proj(traffic) | tod_emb | dow_emb | adaptive_emb | node_emb]``

    where ``input_proj`` lifts each scalar traffic value to ``d_input``
    channels, the time embeddings are broadcast across sensors, and the
    adaptive embedding contributes the ``(time-of-day, sensor)``-specific
    table lookup.

    ``d_input + d_tod + d_dow + d_adp + d_node`` must equal ``d_model``.
    """

    def __init__(
        self,
        steps_per_day: int,
        num_sensors: int,
        d_input: int,
        d_tod: int,
        d_dow: int,
        d_adp: int,
        d_node: int,
    ) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_tod = d_tod
        self.d_dow = d_dow
        self.d_adp = d_adp
        self.d_node = d_node
        self.d_model = d_input + d_tod + d_dow + d_adp + d_node

        if d_input <= 0:
            raise ValueError(f"d_input must be positive, got {d_input}")

        self.input_proj = nn.Linear(1, d_input)
        self.tod_embed = TimeOfDayEmbedding(steps_per_day, d_tod) if d_tod > 0 else None
        self.dow_embed = DayOfWeekEmbedding(d_dow) if d_dow > 0 else None
        self.adaptive_embed = (
            AdaptiveEmbedding(steps_per_day, num_sensors, d_adp) if d_adp > 0 else None
        )
        self.node_embed = NodeEmbedding(num_sensors, d_node) if d_node > 0 else None

    def forward(
        self,
        traffic: torch.Tensor,
        tod_idx: torch.Tensor,
        dow_idx: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, num_sensors = traffic.shape
        parts: list[torch.Tensor] = []

        # Per-cell traffic projection: [B, T, N] -> [B, T, N, d_input]
        parts.append(self.input_proj(traffic.unsqueeze(-1)))

        if self.tod_embed is not None:
            # [B, T, d_tod] -> [B, T, N, d_tod]
            tod = self.tod_embed(tod_idx).unsqueeze(2).expand(-1, -1, num_sensors, -1)
            parts.append(tod)

        if self.dow_embed is not None:
            dow = self.dow_embed(dow_idx).unsqueeze(2).expand(-1, -1, num_sensors, -1)
            parts.append(dow)

        if self.adaptive_embed is not None:
            parts.append(self.adaptive_embed(tod_idx))

        if self.node_embed is not None:
            parts.append(self.node_embed(bsz, seq_len))

        return torch.cat(parts, dim=-1)
