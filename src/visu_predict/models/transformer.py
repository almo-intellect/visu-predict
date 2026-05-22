"""Traffic Transformer model: encoder + (linear/MLP/transformer) decoder."""

from __future__ import annotations

import copy
import math
import random

import torch
import torch.nn as nn

from visu_predict.models.adaptive_graph import AdaptiveAdjacency
from visu_predict.models.attention import FeatureAttention
from visu_predict.models.embeddings import STAEInputComposer
from visu_predict.models.gnn import TORCH_GEOMETRIC_AVAILABLE, GCNEncoder
from visu_predict.models.patching import PatchEmbed
from visu_predict.models.positional import PositionalEncoding
from visu_predict.models.st_blocks import STAttnStack

FeatureInput = torch.Tensor | dict[str, torch.Tensor]

VALID_MODEL_PIPELINES = ("legacy", "stae")


class _AttnEncoderLayer(nn.TransformerEncoderLayer):
    """Encoder layer that exposes self-attention weights (and accepts a spatial bias)."""

    def __init__(self, *args, use_spatial_bias: bool = False, spatial_bias_type: str = "additive", **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights: torch.Tensor | None = None
        self.use_spatial_bias = use_spatial_bias
        self.spatial_bias_type = spatial_bias_type
        self._spatial_bias: torch.Tensor | None = None

    def set_spatial_bias(self, bias: torch.Tensor | None) -> None:
        self._spatial_bias = bias

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):  # type: ignore[override]
        bias = self._spatial_bias if self.use_spatial_bias else None
        if bias is not None:
            if attn_mask is None:
                attn_mask = bias
            elif self.spatial_bias_type == "additive":
                attn_mask = attn_mask + bias
            else:
                attn_mask = attn_mask * bias

        x_out, weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
        )
        self.attn_weights = weights.detach() if weights is not None else None
        return self.dropout1(x_out)


class _DecoderLayer(nn.Module):
    """Transformer decoder layer that stores self-attn and cross-attn weights."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, need_weights=False)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class _Decoder(nn.Module):
    def __init__(self, layer: _DecoderLayer, num_layers: int, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(out) if self.norm is not None else out


def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool().masked_fill(
        torch.eye(sz, device=device).bool(), False,
    ).float().masked_fill(
        torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool(), float("-inf"),
    )


class TrafficTransformer(nn.Module):
    """Transformer-based traffic forecaster with optional GNN encoder and feature attention.

    Args:
        input_dim: Input feature dim for the legacy tensor path.
        num_features: Number of sensors (output dim).
        feature_dims: Per-group feature dims for the dict-input path.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        hidden_dim: int = 336,
        num_layers: int = 3,
        num_heads: int = 16,
        dropout: float = 0.1,
        ff_dim_multiplier: int = 4,
        activation: str = "gelu",
        decoder_type: str = "linear",
        pred_len: int = 12,
        feature_dims: dict[str, int] | None = None,
        use_gnn_pre_transformer: bool = False,
        spatial_feature_dim: int = 0,
        gnn_type: str = "gcn",
        gnn_layers: int = 3,
        gnn_residual: bool = False,
        gat_heads: int = 8,
        gat_concat: bool = True,
        use_spatial_bias: bool = False,
        spatial_bias_type: str = "additive",
        max_seq_length: int = 50_000,
        num_decoder_layers: int = 3,
        teacher_forcing_ratio: float = 0.2,
        model_pipeline: str = "legacy",
        steps_per_day: int = 288,
        d_input: int = 24,
        d_tod: int = 24,
        d_dow: int = 24,
        d_adaptive: int = 80,
        d_node: int = 0,
        seq_length: int = 12,
        interleave_order: str = "TS",
        num_st_layers: int | None = None,
        use_adaptive_adjacency: bool = False,
        adaptive_adj_dim: int = 10,
        adaptive_adj_inject_into: str = "spatial_attn",
        use_temporal_patching: bool = False,
        patch_length: int = 4,
        patch_stride: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be even for positional encoding")
        if decoder_type not in ("linear", "mlp", "transformer"):
            raise ValueError(f"decoder_type must be 'linear', 'mlp', or 'transformer', got {decoder_type}")
        if use_gnn_pre_transformer and not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNN pre-transformer")
        if model_pipeline not in VALID_MODEL_PIPELINES:
            raise ValueError(f"model_pipeline must be one of {VALID_MODEL_PIPELINES}, got {model_pipeline}")

        self.input_dim = input_dim
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pred_len = pred_len
        self.decoder_type = decoder_type
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_gnn_pre_transformer = use_gnn_pre_transformer
        self.use_spatial_bias = use_spatial_bias
        self.spatial_bias_type = spatial_bias_type
        self.model_pipeline = model_pipeline

        self.feature_dims: dict[str, int] = feature_dims or {"traffic": input_dim}
        self.embedding = nn.Linear(input_dim, hidden_dim)

        if model_pipeline == "stae":
            stae_d_model = d_input + d_tod + d_dow + d_adaptive + d_node
            if stae_d_model != hidden_dim:
                raise ValueError(
                    f"STAE pipeline requires d_input+d_tod+d_dow+d_adaptive+d_node "
                    f"({stae_d_model}) to equal hidden_dim ({hidden_dim})"
                )
            self.stae_composer: STAEInputComposer | None = STAEInputComposer(
                steps_per_day=steps_per_day,
                num_sensors=num_features,
                d_input=d_input,
                d_tod=d_tod,
                d_dow=d_dow,
                d_adp=d_adaptive,
                d_node=d_node,
            )
            self.feature_attention = None
            self.stae_attn_stack: STAttnStack | None = STAttnStack(
                d_model=hidden_dim,
                num_heads=num_heads,
                num_layers=num_st_layers or num_layers,
                dropout=dropout,
                ff_multiplier=ff_dim_multiplier,
                interleave_order=interleave_order,
            )
            self.stae_patch: PatchEmbed | None = None
            head_in_steps = seq_length
            if use_temporal_patching:
                self.stae_patch = PatchEmbed(
                    d_model=hidden_dim,
                    patch_length=patch_length,
                    patch_stride=patch_stride,
                    d_out=hidden_dim,
                )
                head_in_steps = self.stae_patch.num_patches(seq_length)
                if head_in_steps <= 0:
                    raise ValueError(
                        f"seq_length {seq_length} too short for patch_length "
                        f"{patch_length} with stride {patch_stride}"
                    )
            self.use_temporal_patching = use_temporal_patching

            # STAE output head: per-sensor projection from
            # ``[N, head_in_steps * d_model]`` to ``[N, pred_len]``.
            self.stae_head: nn.Linear | None = nn.Linear(head_in_steps * hidden_dim, pred_len)
            self.stae_seq_length = seq_length
            # Learnable token that replaces masked-cell embeddings during
            # masked-reconstruction pretraining. Adds ``d_model`` channels
            # to every cell where ``mask == True``.
            self.mask_token: nn.Parameter | None = nn.Parameter(torch.zeros(hidden_dim))
            # Reconstruction head: per-cell projection from d_model back to a
            # single value, used by ``forward_reconstruct`` during pretraining.
            self.reconstruction_head: nn.Linear | None = nn.Linear(hidden_dim, 1)
        else:
            self.stae_composer = None
            self.stae_attn_stack = None
            self.stae_head = None
            self.stae_patch = None
            self.mask_token = None
            self.reconstruction_head = None
            self.use_temporal_patching = False
            self.feature_attention = FeatureAttention(
                feature_dims=self.feature_dims,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_length=max_seq_length,
            )

        # Adaptive adjacency lives outside both pipelines so it can feed
        # the spatial-attention bias (STAE pipeline) and/or the GNN encoder.
        self.use_adaptive_adjacency = use_adaptive_adjacency
        self.adaptive_adj_inject_into = adaptive_adj_inject_into
        self.adaptive_adj: AdaptiveAdjacency | None = (
            AdaptiveAdjacency(num_sensors=num_features, d_emb=adaptive_adj_dim)
            if use_adaptive_adjacency
            else None
        )

        if use_gnn_pre_transformer:
            self.gnn_encoder = GCNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=gnn_layers,
                dropout=dropout,
                gnn_type=gnn_type,
                gat_heads=gat_heads,
                gat_concat=gat_concat,
                residual=gnn_residual,
            )

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        if use_spatial_bias and spatial_feature_dim > 0:
            head_dim = hidden_dim // num_heads
            self.spatial_query_proj = nn.Linear(spatial_feature_dim, head_dim)
            self.spatial_key_proj = nn.Linear(spatial_feature_dim, head_dim)
            self.spatial_bias_layer = nn.Sequential(nn.Linear(head_dim, 1), nn.Sigmoid())

        self.encoder = nn.ModuleList(
            _AttnEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ff_dim_multiplier,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                use_spatial_bias=use_spatial_bias,
                spatial_bias_type=spatial_bias_type,
            )
            for _ in range(num_layers)
        )

        if decoder_type == "transformer":
            self.transformer_decoder = _Decoder(
                _DecoderLayer(hidden_dim, num_heads, hidden_dim * ff_dim_multiplier, dropout, activation),
                num_layers=num_decoder_layers,
                norm=nn.LayerNorm(hidden_dim),
            )
            self.decoder_proj = nn.Linear(hidden_dim, num_features)
            self.target_embedding = nn.Linear(num_features, hidden_dim)
            self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        elif decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, num_features * pred_len),
            )
        else:
            self.decoder = nn.Linear(hidden_dim, num_features * pred_len)

        self.attention_weights: torch.Tensor | None = None
        self.feature_importances: dict[str, float] = {}

    def _compute_spatial_bias(self, spatial_features: torch.Tensor) -> torch.Tensor:
        if spatial_features.dim() == 4:
            spatial_features = spatial_features[:, 0]
        head_dim = self.hidden_dim // self.num_heads
        q = self.spatial_query_proj(spatial_features)  # [B, N, head_dim]
        k = self.spatial_key_proj(spatial_features)
        inter = q.unsqueeze(2) * k.unsqueeze(1)        # [B, N, N, head_dim]
        bias = self.spatial_bias_layer(inter).squeeze(-1)  # [B, N, N]
        bias = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        if self.spatial_bias_type == "additive":
            scale = 1.0 / math.sqrt(head_dim)
            bias = (bias * 2 - 1) * scale
        return bias

    def encode(
        self,
        src: FeatureInput,
        adjacency_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.model_pipeline == "stae":
            return self._stae_encode(src)

        if isinstance(src, torch.Tensor):
            memory = self.embedding(src)
            memory = self.pos_encoder(memory.transpose(0, 1)).transpose(0, 1)
            spatial_bias = None
        else:
            src_dict = dict(src)
            spatial = src_dict.pop("spatial", None) if self.use_spatial_bias else src_dict.get("spatial")

            if self.use_gnn_pre_transformer and "traffic" in src_dict:
                # Prefer the learnable adjacency when configured for the GNN path.
                effective_adj = adjacency_matrix
                if (
                    self.adaptive_adj is not None
                    and self.adaptive_adj_inject_into in ("gnn", "both")
                ):
                    effective_adj = self.adaptive_adj.adjacency()

                if effective_adj is not None:
                    traffic = src_dict["traffic"]
                    _, seq_len, _ = traffic.shape
                    enhanced = []
                    for t in range(seq_len):
                        nodes = traffic[:, t, :].transpose(0, 1)  # [N, B]
                        enhanced_t = self.gnn_encoder(nodes, effective_adj).transpose(0, 1)
                        enhanced.append(enhanced_t)
                    src_dict["traffic"] = torch.stack(enhanced, dim=1)

            memory = self.feature_attention(src_dict)
            self.attention_weights = self.feature_attention.attention_weights
            self.feature_importances = self.feature_attention.feature_importances

            spatial_bias = None
            if self.use_spatial_bias and spatial is not None and hasattr(self, "spatial_query_proj"):
                try:
                    spatial_bias = self._compute_spatial_bias(spatial)
                except RuntimeError:
                    spatial_bias = None

        for layer in self.encoder:
            if spatial_bias is not None:
                layer.set_spatial_bias(spatial_bias)
            memory = layer(memory)
        return memory

    def _stae_encode(self, src: FeatureInput) -> torch.Tensor:
        """STAE pipeline encoder.

        Composes ``[B, T, N, d_model]`` via :class:`STAEInputComposer`, then
        runs an alternating spatial ↔ temporal transformer stack
        (:class:`STAttnStack`). Returns the same shape ``[B, T, N, d_model]``;
        the dedicated STAE head in :meth:`forward` consumes it directly.
        """
        if not isinstance(src, dict):
            raise TypeError("STAE pipeline requires a dict of features, not a raw tensor")
        for required in ("traffic", "time_of_day_idx", "day_of_week_idx"):
            if required not in src:
                raise KeyError(
                    f"STAE pipeline requires {required!r} in the feature dict; "
                    f"got keys {sorted(src.keys())}"
                )
        assert self.stae_composer is not None
        assert self.stae_attn_stack is not None

        composed = self.stae_composer(
            traffic=src["traffic"],
            tod_idx=src["time_of_day_idx"].long(),
            dow_idx=src["day_of_week_idx"].long(),
        )  # [B, T, N, d_model]

        if self.stae_patch is not None:
            composed = self.stae_patch(composed)  # [B, num_patches, N, d_model]

        spatial_bias = None
        if (
            self.adaptive_adj is not None
            and self.adaptive_adj_inject_into in ("spatial_attn", "both")
        ):
            spatial_bias = self.adaptive_adj.as_attention_bias(num_heads=self.num_heads)

        return self.stae_attn_stack(composed, spatial_attn_bias=spatial_bias)

    def forward_reconstruct(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked-reconstruction forward pass (STAE pipeline only).

        Args:
            features: same dict as the supervised STAE forward — must contain
                ``traffic`` (with masked positions zeroed out by the caller),
                ``time_of_day_idx``, ``day_of_week_idx``.
            mask: ``[B, T, N]`` boolean tensor; ``True`` cells are replaced
                with ``self.mask_token`` at the embedding stage.

        Returns:
            ``[B, T, N]`` reconstructed traffic values.
        """
        if self.model_pipeline != "stae":
            raise RuntimeError("forward_reconstruct requires model_pipeline='stae'")
        assert self.stae_composer is not None
        assert self.stae_attn_stack is not None
        assert self.mask_token is not None
        assert self.reconstruction_head is not None

        composed = self.stae_composer(
            traffic=features["traffic"],
            tod_idx=features["time_of_day_idx"].long(),
            dow_idx=features["day_of_week_idx"].long(),
        )  # [B, T, N, d_model]

        # Replace masked-cell embeddings with the learnable mask token.
        mask_expanded = mask.unsqueeze(-1)  # [B, T, N, 1]
        composed = torch.where(mask_expanded, self.mask_token.view(1, 1, 1, -1), composed)

        if self.stae_patch is not None:
            composed = self.stae_patch(composed)

        spatial_bias = None
        if (
            self.adaptive_adj is not None
            and self.adaptive_adj_inject_into in ("spatial_attn", "both")
        ):
            spatial_bias = self.adaptive_adj.as_attention_bias(num_heads=self.num_heads)

        encoded = self.stae_attn_stack(composed, spatial_attn_bias=spatial_bias)
        # [B, T (or num_patches), N, d_model] -> per-cell scalar
        per_cell = self.reconstruction_head(encoded).squeeze(-1)  # [B, T, N]

        # If patching shortens the time dim, lift the per-patch reconstruction
        # back to the full T by repeat_interleave so the caller can compute
        # cell-level losses against the original target.
        if per_cell.size(1) != mask.size(1):
            repeat = mask.size(1) // per_cell.size(1)
            per_cell = per_cell.repeat_interleave(repeat, dim=1)
            # Trim/pad in case of stride or off-by-one.
            per_cell = per_cell[:, : mask.size(1)]
        return per_cell

    def _generate(self, memory: torch.Tensor, steps: int) -> torch.Tensor:
        bsz = memory.size(0)
        device = memory.device
        decoder_input = self.start_token.expand(bsz, 1, -1)
        predictions = []
        for _ in range(steps):
            tgt_len = decoder_input.size(1)
            mask = _causal_mask(tgt_len, device) if tgt_len > 1 else None
            out = self.transformer_decoder(decoder_input, memory, tgt_mask=mask)
            next_features = self.decoder_proj(out[:, -1:])
            predictions.append(next_features)
            decoder_input = torch.cat([decoder_input, self.target_embedding(next_features)], dim=1)
        return torch.cat(predictions, dim=1)

    def forward(
        self,
        src: FeatureInput,
        target: torch.Tensor | None = None,
        adjacency_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(src, adjacency_matrix)

        if self.model_pipeline == "stae":
            assert self.stae_head is not None
            # memory: [B, T, N, d_model] -> per-sensor [N, T*d_model] -> [N, pred_len]
            bsz, seq_len, num_sensors, d_model = memory.shape
            # rearrange to [B, N, T, d_model] -> [B, N, T*d_model]
            flat = memory.permute(0, 2, 1, 3).reshape(bsz, num_sensors, seq_len * d_model)
            out = self.stae_head(flat)  # [B, N, pred_len]
            return out.permute(0, 2, 1).contiguous()  # [B, pred_len, N]

        if self.decoder_type == "transformer":
            if self.training and target is not None and random.random() < self.teacher_forcing_ratio:
                bsz = memory.size(0)
                device = memory.device
                shifted = target[:, :-1, :]
                start = self.start_token.expand(bsz, 1, -1)
                decoder_input = torch.cat([start, self.target_embedding(shifted)], dim=1) if shifted.size(1) > 0 else start
                mask = _causal_mask(decoder_input.size(1), device)
                out = self.transformer_decoder(decoder_input, memory, tgt_mask=mask)
                return self.decoder_proj(out)
            return self._generate(memory, self.pred_len)

        output = self.decoder(memory[:, -1, :])
        return output.view(-1, self.pred_len, self.num_features)

    def freeze_layers(self, freeze_encoder: bool = True, num_layers: int = 1) -> None:
        """Freeze embedding + first N encoder layers (for transfer learning)."""
        if not freeze_encoder:
            return
        for p in self.embedding.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.encoder):
            if i < num_layers:
                for p in layer.parameters():
                    p.requires_grad = False
