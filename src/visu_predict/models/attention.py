"""Feature-wise attention block.

Each feature group (traffic, time, holiday, weather, lagged, spatial) is
embedded and processed by a dedicated transformer layer, then combined via
context-adaptive weighting and a cross-attention fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from visu_predict.models.positional import PositionalEncoding

_RESERVED_KEYS = {"concatenated", "time_of_day_idx", "day_of_week_idx"}


class FeatureAttention(nn.Module):
    """Processes a dictionary of per-feature-group tensors into a single representation."""

    def __init__(
        self,
        feature_dims: dict[str, int],
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_length: int = 50_000,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.feature_keys = [k for k in feature_dims if k not in _RESERVED_KEYS]

        self.embeddings = nn.ModuleDict({k: nn.Linear(feature_dims[k], hidden_dim) for k in self.feature_keys})
        self.pos_encoders = nn.ModuleDict({k: PositionalEncoding(hidden_dim, dropout, max_seq_length) for k in self.feature_keys})
        self.feature_transformers = nn.ModuleDict({
            k: nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            for k in self.feature_keys
        })

        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(1, len(self.feature_keys))),
        )

        self.feature_gates = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            for k in self.feature_keys
        })

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.first_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attention_weights: torch.Tensor | None = None
        self.feature_importances: dict[str, float] = {}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        available = [k for k in self.feature_keys if k in features]
        if not available:
            first = next(iter(features.values()))
            return torch.zeros(first.size(0), first.size(1), self.hidden_dim, device=first.device)

        outputs: dict[str, torch.Tensor] = {}
        for name in available:
            embedded = self.embeddings[name](features[name])  # [B, T, H]
            embedded = embedded.transpose(0, 1)
            embedded = self.pos_encoders[name](embedded)
            embedded = embedded.transpose(0, 1)
            outputs[name] = self.feature_transformers[name](embedded)

        context = outputs.get("traffic")
        if context is None:
            context = torch.stack(list(outputs.values())).mean(dim=0)
        global_context = context.mean(dim=1, keepdim=True)  # [B, 1, H]

        raw_weights = self.context_encoder(global_context).squeeze(1)  # [B, K]
        weights = torch.softmax(raw_weights[:, : len(available)], dim=1)
        self.feature_importances = {name: weights[:, i].mean().item() for i, name in enumerate(available)}

        combined = None
        for i, name in enumerate(available):
            gate_in = torch.cat([outputs[name], global_context.expand(-1, outputs[name].size(1), -1)], dim=2)
            gated = outputs[name] * self.feature_gates[name](gate_in)
            weighted = weights[:, i].view(-1, 1, 1) * gated
            combined = weighted if combined is None else combined + weighted

        first_level = self.first_fusion(combined)

        if len(available) >= 2:
            query = outputs.get("traffic", first_level)
            attn_out, attn_w = self.cross_attention(query, first_level, first_level)
            self.attention_weights = attn_w.detach()
            return self.fusion_layer(torch.cat([first_level, attn_out], dim=2))
        return first_level
