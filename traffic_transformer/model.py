"""
Model module for the traffic prediction transformer.

Contains all neural-network architectures: custom Transformer encoder/decoder
layers with spatial bias support, sinusoidal positional encoding, a GNN encoder,
feature-wise attention, the main TrafficTransformer model, an LSTM baseline
forecaster, and a cosine-warmup learning-rate scheduler.
"""

import copy
import math
import os
import random
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as _optim

from .config import TrainingConfig
from .spatial import SpatialIntegration, GCNEncoder as _SpatialGCNEncoder
from .spatial import load_adjacency_matrix, normalize_adj
from .utils import (
    AMP_AVAILABLE,
    DEVICE_TYPE_SUPPORTED,
    TORCH_GEOMETRIC_AVAILABLE,
    get_maputo_timestamp,
)

if TORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric.nn import GCNConv, GATConv


# =============================================================================
# 1. CustomTransformerEncoderLayer
# =============================================================================

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Transformer encoder layer that captures self-attention weights.

    Overrides ``_sa_block`` to store the raw attention-weight tensor
    produced by ``self.self_attn`` so that it can be inspected after the
    forward pass.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn_weights: Optional[torch.Tensor] = None
        # PATCH: attention-weight capture is now gated. need_weights=True
        # disables PyTorch's fused SDPA kernels and materialises the full
        # attention matrix on every batch (train AND eval), which is a large
        # slowdown on a T4. Enable via model.set_attention_capture(True)
        # only for visualisation passes.
        self.store_attention: bool = False

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Self-attention block that optionally stores attention weights.

        When ``self.store_attention`` is True, calls ``self.self_attn`` with
        ``need_weights=True`` and stores the (detached) attention matrix in
        ``self.attn_weights``; otherwise uses the fast fused path.
        """
        if self.store_attention:
            x_out, attn = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            # PATCH: detach — storing the live tensor kept graph references
            # alive between steps.
            self.attn_weights = attn.detach() if attn is not None else None
        else:
            x_out, _ = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            self.attn_weights = None
        return self.dropout1(x_out)


# =============================================================================
# 2. SpatialBiasTransformerEncoderLayer
# =============================================================================

class SpatialBiasTransformerEncoderLayer(CustomTransformerEncoderLayer):
    """Encoder layer that adds a learned spatial bias to the attention mask.

    The spatial bias can be applied additively (default) or multiplicatively.
    Call :meth:`set_spatial_bias` before the forward pass to supply the bias
    tensor for the current batch.

    Attributes
    ----------
    use_spatial_bias : bool
        Whether spatial bias is currently active (determined by whether a
        bias tensor has been set).
    spatial_bias_type : str
        ``'additive'`` or ``'multiplicative'``.
    spatial_bias : torch.Tensor or None
        The current spatial bias tensor.

    Parameters
    ----------
    *args, **kwargs :
        Forwarded to :class:`CustomTransformerEncoderLayer` /
        ``nn.TransformerEncoderLayer``.
    spatial_bias_mode : str
        ``'additive'`` (default) or ``'multiplicative'``.
    """

    def __init__(
        self,
        *args,
        spatial_bias_mode: str = "additive",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_spatial_bias: bool = False
        self.spatial_bias_type: str = spatial_bias_mode
        self.spatial_bias: Optional[torch.Tensor] = None

    def set_spatial_bias(self, bias: Optional[torch.Tensor]) -> None:
        """Set the spatial bias tensor for the next forward pass.

        Parameters
        ----------
        bias : torch.Tensor or None
            Shape ``(num_heads * batch, seq_len, seq_len)`` or
            broadcastable equivalent.  Pass ``None`` to disable spatial bias.
        """
        self.spatial_bias = bias
        self.use_spatial_bias = bias is not None

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Self-attention with spatial bias injected into the attention mask.

        When a spatial bias has been set via :meth:`set_spatial_bias`, it is
        combined with the existing ``attn_mask`` according to the configured
        ``spatial_bias_type``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        attn_mask : torch.Tensor, optional
            Attention mask.
        key_padding_mask : torch.Tensor, optional
            Key padding mask.
        is_causal : bool
            Whether the attention is causal.

        Returns
        -------
        torch.Tensor
            Attention output after dropout.
        """
        if self.spatial_bias is not None:
            if attn_mask is None:
                attn_mask = self.spatial_bias
            else:
                if self.spatial_bias_type == "multiplicative":
                    attn_mask = attn_mask * self.spatial_bias
                else:
                    attn_mask = attn_mask + self.spatial_bias

        if self.store_attention:
            x_out, attn = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            self.attn_weights = attn.detach() if attn is not None else None
        else:
            x_out, _ = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            self.attn_weights = None
        return self.dropout1(x_out)


# =============================================================================
# 3. TransformerDecoderLayer
# =============================================================================

class TransformerDecoderLayer(nn.Module):
    """Custom Transformer decoder layer with self-attention, cross-attention,
    and a feed-forward network.

    Stores both ``self_attn_weights`` and ``cross_attn_weights`` for later
    inspection / visualisation.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    nhead : int
        Number of attention heads.
    dim_feedforward : int
        Hidden dimension of the feed-forward sub-layer.
    dropout : float
        Dropout probability.
    activation : str
        Activation function name (``'relu'`` or ``'gelu'``).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False,
        )
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False,
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation
        if activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu

        # Stored attention weights
        self.self_attn_weights: Optional[torch.Tensor] = None
        self.cross_attn_weights: Optional[torch.Tensor] = None
        # PATCH: capture gated for speed; see CustomTransformerEncoderLayer.
        self.store_attention: bool = False

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one decoder layer.

        Parameters
        ----------
        tgt : torch.Tensor
            Target sequence ``(tgt_len, batch, d_model)``.
        memory : torch.Tensor
            Encoder output ``(src_len, batch, d_model)``.
        tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask :
            Optional masks.

        Returns
        -------
        torch.Tensor
            Output tensor ``(tgt_len, batch, d_model)``.
        """
        need_w = self.store_attention

        # --- Self-attention ---
        tgt2, sa_w = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=need_w,
            average_attn_weights=True,
        )
        self.self_attn_weights = sa_w.detach() if (need_w and sa_w is not None) else None
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # --- Cross-attention ---
        tgt2, ca_w = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_w,
            average_attn_weights=True,
        )
        self.cross_attn_weights = ca_w.detach() if (need_w and ca_w is not None) else None
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # --- Feed-forward ---
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# =============================================================================
# 4. TransformerDecoder
# =============================================================================

class TransformerDecoder(nn.Module):
    """Stack of :class:`TransformerDecoderLayer` with optional intermediate
    outputs.

    Each layer is a deep copy of the provided *decoder_layer*.  Attention
    weights from every layer are accumulated in ``self_attn_weights`` and
    ``cross_attn_weights`` lists after each forward pass.

    Parameters
    ----------
    decoder_layer : TransformerDecoderLayer
        A single decoder layer instance (will be deep-copied).
    num_layers : int
        Number of stacked layers.
    norm : nn.Module or None
        Optional final layer norm applied after the last decoder layer.
    return_intermediate : bool
        If ``True``, :meth:`forward` returns all intermediate decoder outputs
        stacked along a new leading dimension.
    """

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        # Stored attention weights per layer
        self.self_attn_weights: List[Optional[torch.Tensor]] = []
        self.cross_attn_weights: List[Optional[torch.Tensor]] = []

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full decoder stack.

        Parameters
        ----------
        tgt : torch.Tensor
            Target sequence ``(tgt_len, batch, d_model)``.
        memory : torch.Tensor
            Encoder output ``(src_len, batch, d_model)``.

        Returns
        -------
        torch.Tensor
            If *return_intermediate* is ``False``: ``(tgt_len, batch, d_model)``.
            Otherwise: ``(num_layers, tgt_len, batch, d_model)``.
        """
        output = tgt
        intermediate: List[torch.Tensor] = []

        self.self_attn_weights = []
        self.cross_attn_weights = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            self.self_attn_weights.append(layer.self_attn_weights)
            self.cross_attn_weights.append(layer.cross_attn_weights)

            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


# =============================================================================
# 5. PositionalEncoding
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    The encoding is registered as a non-persistent buffer with shape
    ``(max_len, 1, d_model)`` and added to the input tensor in the forward
    pass.

    Parameters
    ----------
    d_model : int
        Model dimensionality (**must be even**).
    dropout : float
        Dropout applied after the addition of the positional encoding.
    max_len : int
        Maximum supported sequence length.

    Raises
    ------
    ValueError
        If *d_model* is odd.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(
                f"PositionalEncoding requires an even d_model, got {d_model}"
            )

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: [max_len, 1, d_model]
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x*.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(seq_len, batch, d_model)``.

        Returns
        -------
        torch.Tensor
            Same shape as *x*, with positional encoding added and dropout
            applied.

        Raises
        ------
        ValueError
            If the last dimension of *x* does not match ``d_model``.
        """
        if x.size(-1) != self.pe.size(-1):
            raise ValueError(
                f"Input dimension {x.size(-1)} does not match "
                f"positional encoding dimension {self.pe.size(-1)}"
            )
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# =============================================================================
# 6. GCNEncoder
# =============================================================================

class GCNEncoder(nn.Module):
    """Graph Neural Network Encoder supporting GCN and GAT convolutions.

    Provides batched GNN processing (FIX #5) -- the model's ``encode``
    method processes all timesteps in a single batched call rather than
    looping over individual timesteps.

    The helper :meth:`_dense_to_sparse` converts a dense adjacency matrix
    to the COO ``edge_index`` format expected by ``torch_geometric``.

    Each layer is followed by :class:`~torch.nn.LayerNorm` and an optional
    residual / skip connection when the input and output dimensions match.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each node's input features.
    hidden_dim : int
        Width of hidden GNN layers.
    num_layers : int
        Number of stacked GNN layers.
    dropout : float
        Dropout probability applied after each layer.
    gnn_type : str
        ``'gcn'`` or ``'gat'``.
    gat_heads : int
        Number of attention heads when ``gnn_type='gat'``.
    gat_concat : bool
        Whether to concatenate (``True``) or average (``False``) GAT heads.
    residual : bool
        Enable residual / skip connections.

    Raises
    ------
    ImportError
        If ``torch_geometric`` is not installed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        gnn_type: str = "gcn",
        gat_heads: int = 8,
        gat_concat: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GCNEncoder but is not "
                "installed.  Install it with: pip install torch-geometric"
            )

        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()
        self.residual = residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            if self.gnn_type == "gat":
                # When concatenating heads the actual output width is
                # heads * out_dim, so we adjust per-head output accordingly.
                head_out = out_dim // gat_heads if gat_concat else out_dim
                conv = GATConv(
                    in_dim,
                    head_out,
                    heads=gat_heads,
                    concat=gat_concat,
                    dropout=dropout,
                )
                effective_out = head_out * gat_heads if gat_concat else out_dim
            else:
                conv = GCNConv(in_dim, out_dim)
                effective_out = out_dim

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(effective_out))

            # Update hidden_dim for subsequent layers if GAT concat changes it
            if i == 0 and self.gnn_type == "gat" and gat_concat:
                hidden_dim = effective_out

        self.dropout_layer = nn.Dropout(dropout)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _dense_to_sparse(adj_matrix: torch.Tensor) -> torch.Tensor:
        """Convert a dense adjacency matrix to COO ``edge_index``.

        Parameters
        ----------
        adj_matrix : torch.Tensor
            Dense ``(N, N)`` adjacency matrix.

        Returns
        -------
        torch.Tensor
            ``(2, E)`` edge-index tensor.
        """
        edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
        return edge_index

    # -- forward -------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Run the GNN encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(N, input_dim)``.
        adjacency_matrix : torch.Tensor
            Dense ``(N, N)`` adjacency matrix.

        Returns
        -------
        torch.Tensor
            Encoded node features of shape ``(N, hidden_dim)``.
        """
        edge_index = self._dense_to_sparse(adjacency_matrix)
        # PATCH: edge weights were previously discarded when densifying.
        edge_weight = adjacency_matrix[edge_index[0], edge_index[1]]

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            identity = x
            if self.gnn_type == "gcn":
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout_layer(x)

            # Residual connection (requires matching dimensions)
            if self.residual and identity.shape == x.shape:
                x = x + identity

            x = norm(x)

        return x

    # -- utilities -----------------------------------------------------------

    def reset_parameters(self) -> None:
        """Re-initialise all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()


# =============================================================================
# 6b. DenseGCNPreEncoder  (PATCH)
# =============================================================================

class DenseGCNPreEncoder(nn.Module):
    """Dense graph-convolutional pre-encoder over the sensor dimension.

    PATCH: replaces the previous torch_geometric wiring, which constructed
    ``GCNConv(input_dim=num_sensors)`` yet fed it node features of width
    ``batch * seq`` — a shape mismatch by construction (and it silently never
    ran because the adjacency matrix failed to load upstream).

    Here every timestep's N sensor readings are treated as N graph nodes with
    a scalar feature. Propagation is ``H <- act(A_hat @ H @ W)`` computed
    densely with an einsum, which is fast for N up to a few hundred, honours
    edge weights, and needs no torch_geometric. The output is projected back
    to a scalar per sensor and added residually, so the traffic block keeps
    its ``(batch, seq, N)`` shape and all downstream dimensions are unchanged.

    Parameters
    ----------
    num_layers : int
        Number of propagation layers.
    hidden_dim : int
        Width of the per-node hidden representation.
    dropout : float
        Dropout probability applied after each propagation.
    """

    def __init__(
        self,
        num_layers: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Spatially propagate traffic readings.

        Parameters
        ----------
        x : torch.Tensor
            Traffic block of shape ``(batch, seq, N)``.
        adj : torch.Tensor
            Normalised adjacency of shape ``(N, N)``.

        Returns
        -------
        torch.Tensor
            Same shape as *x* (residual output).
        """
        h = self.in_proj(x.unsqueeze(-1))                    # (B, T, N, H)
        for lin, norm in zip(self.layers, self.norms):
            msg = torch.einsum("ij,btjh->btih", adj, h)      # neighbour agg.
            msg = self.dropout(torch.relu(lin(msg)))
            h = norm(h + msg)                                # residual + LN
        out = self.out_proj(h).squeeze(-1)                   # (B, T, N)
        return x + out


# =============================================================================
# 7. FeatureAttention
# =============================================================================

class FeatureAttention(nn.Module):
    """Feature-wise attention module that processes heterogeneous input
    features (traffic, temporal, weather, spatial) with per-feature
    transformer blocks, pairwise cross-attention, context-adaptive gating,
    and two-level fusion.

    The module stores diagnostic tensors after each forward pass:

    * ``attention_weights`` -- per-feature transformer attention (currently
      captured via the underlying ``nn.TransformerEncoder`` layers).
    * ``feature_importances`` -- context-adaptive weight vector.
    * ``pairwise_weights`` -- cross-attention weight matrices between all
      feature pairs.
    * ``gate_values`` -- per-feature sigmoid gate activations.

    Parameters
    ----------
    feature_dims : dict
        Mapping of feature-group name to its raw dimensionality.
    d_model : int
        Internal model dimension for each feature stream.
    nhead : int
        Number of attention heads.
    num_layers : int
        Depth of per-feature transformer blocks.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_dims = feature_dims
        self.d_model = d_model
        self.feature_names: List[str] = sorted(feature_dims.keys())
        self.num_features = len(self.feature_names)

        # ---- Per-feature embeddings ----
        self.feature_embeddings = nn.ModuleDict({
            name: nn.Linear(dim, d_model)
            for name, dim in feature_dims.items()
        })

        # ---- Per-feature positional encoders ----
        self.feature_pos_encoders = nn.ModuleDict({
            name: PositionalEncoding(d_model, dropout=dropout)
            for name in self.feature_names
        })

        # ---- Per-feature transformer blocks ----
        self.feature_transformers = nn.ModuleDict()
        for name in self.feature_names:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=False,
            )
            self.feature_transformers[name] = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

        # ---- Pairwise cross-attention between feature groups ----
        self.cross_attention_pairs = nn.ModuleDict()
        for i, name_i in enumerate(self.feature_names):
            for j, name_j in enumerate(self.feature_names):
                if i != j:
                    pair_key = f"{name_i}_to_{name_j}"
                    self.cross_attention_pairs[pair_key] = nn.MultiheadAttention(
                        embed_dim=d_model,
                        num_heads=nhead,
                        dropout=dropout,
                        batch_first=False,
                    )

        # ---- Context-adaptive feature weights ----
        context_input_dim = d_model * self.num_features
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.num_features),
            nn.Softmax(dim=-1),
        )

        # ---- Feature gating mechanism ----
        self.feature_gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            for name in self.feature_names
        })

        # ---- Two-level fusion ----
        # First fusion: concatenation + projection
        self.first_fusion = nn.Sequential(
            nn.Linear(d_model * self.num_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cross-attention for second-level fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False,
        )
        self.cross_attention_norm = nn.LayerNorm(d_model)

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        # ---- Diagnostic storage ----
        self.attention_weights: Dict[str, Optional[torch.Tensor]] = {}
        self.feature_importances: Optional[torch.Tensor] = None
        self.pairwise_weights: Dict[str, Optional[torch.Tensor]] = {}
        self.gate_values: Dict[str, Optional[torch.Tensor]] = {}

    def forward(
        self,
        feature_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Process features through per-feature transformers, cross-attention,
        context-adaptive gating, and two-level fusion.

        Parameters
        ----------
        feature_dict : dict
            Keys are feature-group names, values are tensors of shape
            ``(seq_len, batch, feature_dim)``.

        Returns
        -------
        torch.Tensor
            Fused representation ``(seq_len, batch, d_model)``.

        Raises
        ------
        ValueError
            If *feature_dict* contains none of the expected feature names.
        """
        # ---- Stage 1: per-feature embedding + positional encoding + transformer ----
        feature_outputs: Dict[str, torch.Tensor] = {}
        for name in self.feature_names:
            if name not in feature_dict:
                continue
            x = feature_dict[name]  # (seq_len, batch, feat_dim)
            x = self.feature_embeddings[name](x)  # -> (seq_len, batch, d_model)
            x = self.feature_pos_encoders[name](x)
            x = self.feature_transformers[name](x)
            feature_outputs[name] = x

        available_names = [n for n in self.feature_names if n in feature_outputs]

        if len(available_names) == 0:
            raise ValueError("No features available in feature_dict")

        # If only one feature, skip cross-attention/gating/fusion
        if len(available_names) == 1:
            return feature_outputs[available_names[0]]

        # ---- Stage 2: pairwise cross-attention ----
        cross_attended: Dict[str, torch.Tensor] = {
            name: feature_outputs[name] for name in available_names
        }

        for i, name_i in enumerate(available_names):
            for j, name_j in enumerate(available_names):
                if i == j:
                    continue
                pair_key = f"{name_i}_to_{name_j}"
                if pair_key in self.cross_attention_pairs:
                    attn_out, attn_w = self.cross_attention_pairs[pair_key](
                        feature_outputs[name_i],
                        feature_outputs[name_j],
                        feature_outputs[name_j],
                        need_weights=True,
                        average_attn_weights=True,
                    )
                    cross_attended[name_i] = cross_attended[name_i] + attn_out
                    self.pairwise_weights[pair_key] = attn_w.detach()

        # ---- Stage 3: context-adaptive feature weights ----
        # Pool each feature over the time dimension
        seq_len = list(cross_attended.values())[0].size(0)
        batch_size = list(cross_attended.values())[0].size(1)

        pooled_features: List[torch.Tensor] = []
        for name in available_names:
            pooled = cross_attended[name].mean(dim=0)  # (batch, d_model)
            pooled_features.append(pooled)

        # Pad if fewer features than expected
        while len(pooled_features) < self.num_features:
            pooled_features.append(
                torch.zeros(
                    batch_size, self.d_model,
                    device=pooled_features[0].device,
                )
            )

        context_input = torch.cat(pooled_features, dim=-1)  # (batch, num_features * d_model)
        feature_weights = self.context_encoder(context_input)  # (batch, num_features)
        self.feature_importances = feature_weights.detach()

        # ---- Stage 4: feature gating ----
        gated_features: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(available_names):
            weight_idx = min(idx, feature_weights.size(-1) - 1)
            w = feature_weights[:, weight_idx].unsqueeze(0).unsqueeze(-1)  # (1, batch, 1)

            # Gate input is concatenation of original and cross-attended features
            gate_input = torch.cat(
                [feature_outputs[name], cross_attended[name]], dim=-1
            )  # (seq_len, batch, d_model*2)
            gate_value = self.feature_gates[name](gate_input)  # (seq_len, batch, d_model)
            self.gate_values[name] = gate_value.detach()

            gated = gate_value * cross_attended[name] * w
            gated_features[name] = gated

        # ---- Stage 5: two-level fusion ----
        # First level: concatenate gated features and project
        gated_list = [gated_features[name] for name in available_names]
        # Pad if fewer features than expected
        while len(gated_list) < self.num_features:
            gated_list.append(
                torch.zeros(
                    seq_len, batch_size, self.d_model,
                    device=gated_list[0].device,
                )
            )

        concat_features = torch.cat(gated_list, dim=-1)  # (seq_len, batch, num_features*d_model)
        fused = self.first_fusion(concat_features)  # (seq_len, batch, d_model)

        # Second level: cross-attention between fused and primary feature
        primary_name = available_names[0]  # typically 'traffic'
        cross_out, _ = self.cross_attention(
            fused,
            feature_outputs[primary_name],
            feature_outputs[primary_name],
        )
        fused = self.cross_attention_norm(fused + cross_out)

        # Final fusion
        fused = self.fusion_norm(fused + self.fusion_layer(fused))

        return fused


# =============================================================================
# 8. TrafficTransformer
# =============================================================================

class TrafficTransformer(nn.Module):
    """Main traffic-prediction transformer model.

    Supports multi-feature input (traffic, temporal, weather, spatial),
    optional GNN pre-encoding with batched processing (FIX #5), spatial
    bias in attention, and multiple decoder types (``'transformer'``,
    ``'mlp'``, ``'linear'``).

    Parameters
    ----------
    input_dim : int
        Raw input feature dimension (total across all features).
    d_model : int
        Internal model dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of encoder layers.
    dim_feedforward : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout probability.
    pred_length : int
        Prediction horizon length.
    output_dim : int
        Output feature dimension per time step.
    num_sensors : int
        Number of sensor nodes.
    use_feature_attention : bool
        Enable :class:`FeatureAttention`.
    feature_dims : dict or None
        Mapping of feature names to their raw dimensionality.
    use_gnn_pre_transformer : bool
        Enable GNN encoding before the transformer.
    gnn_hidden_dim : int
        GNN hidden dimension.
    gnn_num_layers : int
        Number of GNN layers.
    gnn_type : str
        ``'gcn'`` or ``'gat'``.
    use_spatial_bias : bool
        Enable spatial bias in encoder attention.
    spatial_bias_mode : str
        ``'additive'`` or ``'multiplicative'``.
    decoder_type : str
        ``'transformer'``, ``'mlp'``, or ``'linear'``.
    decoder_layers : int
        Number of decoder layers (for ``decoder_type='transformer'``).
    activation : str
        Activation function (``'relu'`` or ``'gelu'``).
    teacher_forcing_ratio : float
        Initial teacher-forcing ratio for transformer decoder training.
    """

    def __init__(
        self,
        input_dim: int = 207,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pred_length: int = 12,
        output_dim: int = 207,
        num_sensors: int = 207,
        use_feature_attention: bool = False,
        feature_dims: Optional[Dict[str, int]] = None,
        use_gnn_pre_transformer: bool = False,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        gnn_type: str = "gcn",
        use_spatial_bias: bool = False,
        spatial_bias_mode: str = "additive",
        decoder_type: str = "mlp",
        decoder_layers: int = 2,
        activation: str = "gelu",
        teacher_forcing_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.pred_length = pred_length
        self.output_dim = output_dim
        self.num_sensors = num_sensors
        self.use_feature_attention = use_feature_attention
        self.feature_dims = feature_dims
        self.use_gnn_pre_transformer = use_gnn_pre_transformer
        self.use_spatial_bias = use_spatial_bias
        self.spatial_bias_mode = spatial_bias_mode
        self.decoder_type = decoder_type
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # ---- Input embedding ----
        self.embedding = nn.Linear(input_dim, d_model)

        # ---- Feature attention ----
        if use_feature_attention and feature_dims is not None:
            self.feature_attention = FeatureAttention(
                feature_dims=feature_dims,
                d_model=d_model,
                nhead=nhead,
                num_layers=max(1, num_layers // 2),
                dropout=dropout,
            )
        else:
            self.feature_attention = None

        # ---- Positional encoding ----
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # ---- Optional GNN encoder ----
        # PATCH: the previous wiring built GCNConv(input_dim=num_sensors) and
        # fed it node features of width batch*seq — broken by construction.
        # The dense pre-encoder below is shape-correct, honours edge weights,
        # and requires no torch_geometric.
        self.gnn_encoder: Optional[DenseGCNPreEncoder] = None
        if use_gnn_pre_transformer:
            self.gnn_encoder = DenseGCNPreEncoder(
                num_layers=gnn_num_layers,
                hidden_dim=gnn_hidden_dim,
                dropout=dropout,
            )

        # ---- Spatial bias components ----
        if use_spatial_bias:
            self.spatial_query_proj = nn.Linear(d_model, d_model)
            self.spatial_key_proj = nn.Linear(d_model, d_model)
            self.spatial_bias_layer = nn.Sequential(
                nn.Linear(num_sensors, d_model),
                nn.ReLU(),
                nn.Linear(d_model, nhead),
            )
        else:
            self.spatial_query_proj = None
            self.spatial_key_proj = None
            self.spatial_bias_layer = None

        # ---- Encoder (ModuleList of SpatialBiasTransformerEncoderLayer) ----
        self.encoder = nn.ModuleList([
            SpatialBiasTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=False,
                spatial_bias_mode=spatial_bias_mode,
            )
            for _ in range(num_layers)
        ])
        # PATCH: the old ``self.transformer = self.encoder`` alias registered
        # the same ModuleList twice, duplicating every encoder key in the
        # state_dict (≈2x checkpoint size). A read-only property (below)
        # keeps backward compatibility for readers.

        # ---- Decoder ----
        if decoder_type == "transformer":
            decoder_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            self.decoder = TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=decoder_layers,
                return_intermediate=False,
            )
            self.decoder_proj = nn.Linear(d_model, output_dim)

            # Target embedding for decoder input
            self.target_embedding = nn.Linear(output_dim, d_model)

            # Learnable start token
            self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

            # Causal mask buffer
            causal = self._generate_square_subsequent_mask(pred_length)
            self.register_buffer("causal_mask", causal)

        elif decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, dim_feedforward // 2),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, pred_length * output_dim),
            )
        elif decoder_type == "linear":
            self.decoder = nn.Linear(d_model, pred_length * output_dim)
        else:
            raise ValueError(
                f"Unknown decoder_type '{decoder_type}'. "
                f"Choose from 'transformer', 'mlp', 'linear'."
            )

    # -- Backward-compatibility alias -----------------------------------------

    @property
    def transformer(self) -> nn.ModuleList:
        """Read-only alias for the encoder stack (see PATCH note in __init__)."""
        return self.encoder

    # -- Attention capture ----------------------------------------------------

    def set_attention_capture(self, enabled: bool = True) -> None:
        """Enable/disable attention-weight storage on all layers.

        PATCH: capture is off by default for speed (need_weights=True disables
        fused attention). Turn it on, run one forward pass, then call the
        visualisation functions.
        """
        for module in self.modules():
            if hasattr(module, "store_attention"):
                module.store_attention = bool(enabled)

    # -- FIX #3: Scheduled teacher forcing ------------------------------------

    def set_teacher_forcing_ratio(self, ratio: float) -> None:
        """Update the teacher-forcing ratio (clamped to [0, 1]).

        Parameters
        ----------
        ratio : float
            New teacher-forcing ratio.
        """
        self.teacher_forcing_ratio = max(0.0, min(1.0, ratio))

    # -- Static helpers -------------------------------------------------------

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate a causal (upper-triangular) mask of size ``sz x sz``.

        Returns
        -------
        torch.Tensor
            Float mask with ``-inf`` above the diagonal and ``0`` on and below.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    # -- Spatial bias ---------------------------------------------------------

    def compute_spatial_bias(
        self,
        adjacency_matrix: torch.Tensor,
        seq_len: int,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """Compute spatial bias from the adjacency matrix for injection into
        encoder attention.

        Parameters
        ----------
        adjacency_matrix : torch.Tensor
            Dense adjacency matrix ``(num_sensors, num_sensors)``.
        seq_len : int
            Current sequence length.
        batch_size : int
            Current batch size.

        Returns
        -------
        torch.Tensor or None
            Spatial bias tensor broadcastable to attention shape, or ``None``
            if spatial bias is not enabled.
        """
        # PATCH NOTE: this implementation reduces the adjacency to a single
        # scalar per head and broadcasts it uniformly over the (seq, seq)
        # attention logits. An additive constant shifts every logit equally,
        # which softmax cancels exactly — i.e. the 'additive' mode has NO
        # effect on the attention distribution. It is kept for API
        # compatibility but should be considered inactive; a real spatial
        # bias needs per-(i, j) terms. use_spatial_bias stays False by
        # default.
        if not self.use_spatial_bias or self.spatial_bias_layer is None:
            return None

        # adjacency_matrix: (num_sensors, num_sensors)
        # Project through bias layer to get per-head bias
        bias = self.spatial_bias_layer(adjacency_matrix)  # (num_sensors, nhead)
        bias = bias.permute(1, 0)  # (nhead, num_sensors)

        # Expand to (nhead, seq_len, seq_len) via outer-product-style
        # We use a simplified approach: replicate the sensor-level bias
        # across the sequence dimension
        bias = bias.unsqueeze(1).unsqueeze(3)  # (nhead, 1, num_sensors, 1)

        # For sequence-level attention, create (nhead, seq_len, seq_len) bias
        # by averaging the sensor-dimension bias
        bias_mean = bias.squeeze(-1).mean(dim=-1)  # (nhead, 1)
        spatial_bias = bias_mean.unsqueeze(-1).expand(
            self.nhead, seq_len, seq_len
        )  # (nhead, seq_len, seq_len)

        # Expand for batch: (nhead * batch, seq_len, seq_len)
        spatial_bias = spatial_bias.unsqueeze(1).expand(
            self.nhead, batch_size, seq_len, seq_len
        ).reshape(self.nhead * batch_size, seq_len, seq_len)

        return spatial_bias

    # -- Encode ---------------------------------------------------------------

    def encode(
        self,
        src: Union[torch.Tensor, Dict[str, torch.Tensor]],
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode the source sequence.

        When a GNN encoder is available the traffic features are processed
        in a **batched** manner (FIX #5): all timesteps are flattened into a
        single node-feature matrix, passed through the GNN in one call, and
        then reshaped back.  This replaces the previous per-timestep loop.

        Parameters
        ----------
        src : torch.Tensor or dict
            Either a plain tensor ``(batch, seq_len, input_dim)`` or a dict
            mapping feature-group names to tensors.
        adjacency_matrix : torch.Tensor, optional
            Dense adjacency matrix for GNN pre-encoding and spatial bias.

        Returns
        -------
        torch.Tensor
            Encoder output ``(seq_len, batch, d_model)``.
        """
        # ---- Handle dict vs tensor input ----
        # PATCH: copy the dict so we never mutate the caller's batch.
        if isinstance(src, dict):
            src_dict = dict(src)
        else:
            src_dict = {"traffic": src}

        # ---- GNN pre-encoding (PATCH: dense, shape-correct) ----
        # The dataset concatenates [traffic_sensors | time | weather | ...]
        # along the last axis, with the N sensor columns first. The GNN is
        # applied to those first N columns only; auxiliary features pass
        # through untouched.
        if (
            self.gnn_encoder is not None
            and adjacency_matrix is not None
            and "traffic" in src_dict
        ):
            traffic = src_dict["traffic"]
            n = adjacency_matrix.shape[0]
            if traffic.dim() == 3 and traffic.shape[-1] >= n:
                adj = adjacency_matrix.to(device=traffic.device, dtype=traffic.dtype)
                head = traffic[..., :n]
                tail = traffic[..., n:]
                head = self.gnn_encoder(head, adj)
                src_dict["traffic"] = (
                    torch.cat([head, tail], dim=-1) if tail.shape[-1] else head
                )
            else:
                warnings.warn(
                    f"GNN pre-encoder skipped: input width {traffic.shape[-1]} "
                    f"is smaller than adjacency size {n}."
                )

        # ---- Feature attention or plain embedding ----
        if self.feature_attention is not None and isinstance(src, dict):
            # Prepare feature dict in (seq_len, batch, feat_dim) format
            feature_dict_transposed: Dict[str, torch.Tensor] = {}
            for name, tensor in src_dict.items():
                if name in self.feature_attention.feature_names:
                    # Input is (batch, seq_len, feat_dim) -> (seq_len, batch, feat_dim)
                    feature_dict_transposed[name] = tensor.permute(1, 0, 2)
            x = self.feature_attention(feature_dict_transposed)
            # x: (seq_len, batch, d_model)
        else:
            # Concatenate all features if dict
            if isinstance(src, dict):
                tensors = []
                for name in sorted(src_dict.keys()):
                    tensors.append(src_dict[name])
                combined = torch.cat(tensors, dim=-1)  # (batch, seq_len, total_dim)
            else:
                combined = src  # (batch, seq_len, input_dim)

            # Embed and transpose to (seq_len, batch, d_model)
            x = self.embedding(combined)
            x = x.permute(1, 0, 2)

        # ---- Positional encoding ----
        x = self.pos_encoder(x)

        seq_len = x.size(0)
        batch_size = x.size(1)

        # ---- Compute spatial bias ----
        spatial_bias: Optional[torch.Tensor] = None
        if self.use_spatial_bias and adjacency_matrix is not None:
            spatial_bias = self.compute_spatial_bias(
                adjacency_matrix, seq_len, batch_size
            )

        # ---- Run through encoder layers ----
        for layer in self.encoder:
            # PATCH: guard — adapter-wrapped layers (transfer learning) do not
            # necessarily expose set_spatial_bias.
            if hasattr(layer, "set_spatial_bias"):
                layer.set_spatial_bias(spatial_bias)
            x = layer(x)

        return x

    # -- Decode single step (for autoregressive generation) -------------------

    def decode_single_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a single decoder step.

        Parameters
        ----------
        tgt : torch.Tensor
            Target input ``(tgt_len, batch, d_model)``.
        memory : torch.Tensor
            Encoder output ``(src_len, batch, d_model)``.
        tgt_mask : torch.Tensor, optional
            Causal mask.

        Returns
        -------
        torch.Tensor
            Decoded output ``(tgt_len, batch, output_dim)``.

        Raises
        ------
        RuntimeError
            If the decoder type is not ``'transformer'``.
        """
        if self.decoder_type != "transformer":
            raise RuntimeError(
                "decode_single_step is only available with decoder_type='transformer'"
            )

        decoder_output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.decoder_proj(decoder_output)
        return output

    # -- Autoregressive generation --------------------------------------------

    def generate(
        self,
        memory: torch.Tensor,
        pred_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Auto-regressively generate predictions using the transformer
        decoder.

        At each step the last decoded token is projected to ``output_dim``,
        re-embedded via ``target_embedding``, and appended to the decoder
        input sequence.

        Parameters
        ----------
        memory : torch.Tensor
            Encoder output ``(src_len, batch, d_model)``.
        pred_length : int, optional
            Number of steps to generate (defaults to ``self.pred_length``).

        Returns
        -------
        torch.Tensor
            Generated predictions ``(batch, pred_length, output_dim)``.
        """
        if pred_length is None:
            pred_length = self.pred_length

        batch_size = memory.size(1)
        device = memory.device

        # Start with learnable start token
        decoder_input = self.start_token.expand(1, batch_size, -1)  # (1, batch, d_model)
        outputs: List[torch.Tensor] = []

        for t in range(pred_length):
            tgt_len = decoder_input.size(0)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)

            decoder_output = self.decoder(
                decoder_input, memory, tgt_mask=tgt_mask
            )
            # Take the last time step
            last_output = decoder_output[-1:]  # (1, batch, d_model)
            pred = self.decoder_proj(last_output)  # (1, batch, output_dim)
            outputs.append(pred)

            # Prepare next input
            next_input = self.target_embedding(pred)  # (1, batch, d_model)
            decoder_input = torch.cat([decoder_input, next_input], dim=0)

        # Stack outputs: (pred_length, batch, output_dim) -> (batch, pred_length, output_dim)
        output = torch.cat(outputs, dim=0).permute(1, 0, 2)
        return output

    # -- Forward --------------------------------------------------------------

    def forward(
        self,
        src: Union[torch.Tensor, Dict[str, torch.Tensor]],
        tgt: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward pass: encode, then decode.

        Parameters
        ----------
        src : torch.Tensor or dict
            Source input. Plain tensor of shape ``(batch, seq_len, input_dim)``
            or a dict mapping feature names to tensors.
        tgt : torch.Tensor, optional
            Target sequence ``(batch, pred_length, output_dim)`` for teacher
            forcing (transformer decoder only).
        adjacency_matrix : torch.Tensor, optional
            Dense adjacency matrix.

        Returns
        -------
        torch.Tensor
            Predictions ``(batch, pred_length, output_dim)``.
        """
        # ---- Encode ----
        memory = self.encode(src, adjacency_matrix=adjacency_matrix)
        # memory: (seq_len, batch, d_model)

        batch_size = memory.size(1)
        device = memory.device

        # ---- Decode ----
        if self.decoder_type == "transformer":
            if tgt is not None and self.training:
                # Scheduled teacher forcing
                use_teacher_forcing = (
                    random.random() < self.teacher_forcing_ratio
                )

                if use_teacher_forcing:
                    # Teacher forcing: feed ground truth as decoder input
                    # tgt: (batch, pred_length, output_dim)
                    tgt_embedded = self.target_embedding(tgt)  # (batch, pred_len, d_model)
                    tgt_embedded = tgt_embedded.permute(1, 0, 2)  # (pred_len, batch, d_model)

                    # Prepend start token
                    start = self.start_token.expand(1, batch_size, -1)
                    decoder_input = torch.cat(
                        [start, tgt_embedded[:-1]], dim=0
                    )  # (pred_len, batch, d_model)

                    tgt_mask = self._generate_square_subsequent_mask(
                        self.pred_length
                    ).to(device)

                    decoder_output = self.decoder(
                        decoder_input, memory, tgt_mask=tgt_mask
                    )
                    output = self.decoder_proj(decoder_output)
                    # (pred_len, batch, output_dim) -> (batch, pred_len, output_dim)
                    output = output.permute(1, 0, 2)
                else:
                    # Autoregressive with scheduled teacher forcing off
                    output = self.generate(memory, pred_length=self.pred_length)
            else:
                # Inference: always autoregressive
                output = self.generate(memory, pred_length=self.pred_length)

        elif self.decoder_type == "mlp":
            # Use mean-pooled encoder output
            pooled = memory.mean(dim=0)  # (batch, d_model)
            flat_output = self.decoder(pooled)  # (batch, pred_length * output_dim)
            output = flat_output.view(batch_size, self.pred_length, self.output_dim)

        elif self.decoder_type == "linear":
            pooled = memory.mean(dim=0)  # (batch, d_model)
            flat_output = self.decoder(pooled)  # (batch, pred_length * output_dim)
            output = flat_output.view(batch_size, self.pred_length, self.output_dim)

        else:
            raise ValueError(f"Unknown decoder_type '{self.decoder_type}'")

        return output

    # -- Layer freezing -------------------------------------------------------

    def freeze_layers(
        self,
        freeze_embedding: bool = False,
        freeze_encoder_layers: Optional[int] = None,
        freeze_feature_attention: bool = False,
    ) -> None:
        """Selectively freeze model layers for transfer learning.

        Parameters
        ----------
        freeze_embedding : bool
            Freeze the input embedding layer.
        freeze_encoder_layers : int or None
            Number of encoder layers to freeze (from the bottom).
            ``None`` means do not freeze any.
        freeze_feature_attention : bool
            Freeze the feature attention module.
        """
        if freeze_embedding:
            for param in self.embedding.parameters():
                param.requires_grad = False

        if freeze_encoder_layers is not None:
            for idx, layer in enumerate(self.encoder):
                if idx < freeze_encoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        if freeze_feature_attention and self.feature_attention is not None:
            for param in self.feature_attention.parameters():
                param.requires_grad = False

        # Log frozen parameter count
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        trainable_params = total_params - frozen_params
        print(
            f"Froze {frozen_params:,} / {total_params:,} parameters. "
            f"Trainable: {trainable_params:,}"
        )


# =============================================================================
# 9. PyTorchLSTMForecaster
# =============================================================================

class PyTorchLSTMForecaster(nn.Module):
    """LSTM-based baseline model for traffic forecasting.

    Includes :meth:`fit` and :meth:`predict` convenience methods for
    training and inference with standard PyTorch DataLoaders.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int
        LSTM hidden-state dimension.
    num_layers : int
        Number of stacked LSTM layers.
    output_dim : int
        Number of output features per time step.
    pred_length : int
        Prediction horizon length.
    dropout : float
        Dropout between LSTM layers.
    bidirectional : bool
        Use bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int = 207,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 207,
        pred_length: int = 12,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.pred_length = pred_length
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_dim, pred_length * output_dim)
        self.dropout = nn.Dropout(dropout)

        # Training state
        self._optimizer: Optional[_optim.Optimizer] = None
        self._criterion: Optional[nn.Module] = None
        self._device: str = "cpu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(batch, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Predictions ``(batch, pred_length, output_dim)``.
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last time step output
        last_output = lstm_out[:, -1, :]  # (batch, lstm_output_dim)
        last_output = self.dropout(last_output)

        # Project to prediction
        flat_output = self.fc(last_output)  # (batch, pred_length * output_dim)
        output = flat_output.view(-1, self.pred_length, self.output_dim)
        return output

    def fit(
        self,
        train_loader,
        val_loader=None,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 10,
        device: str = "cpu",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the LSTM model.

        Parameters
        ----------
        train_loader :
            PyTorch DataLoader yielding ``(input, target)`` batches.
        val_loader :
            Optional validation DataLoader.
        num_epochs : int
            Maximum number of training epochs.
        learning_rate : float
            Optimiser learning rate.
        patience : int
            Early-stopping patience (epochs without improvement).
        device : str
            Device string.
        verbose : bool
            Print progress every 10 epochs.

        Returns
        -------
        dict
            Keys ``'train_loss'`` and optionally ``'val_loss'``, each
            mapping to a list of per-epoch loss values.
        """
        self._device = device
        self.to(device)

        self._optimizer = _optim.Adam(self.parameters(), lr=learning_rate)
        self._criterion = nn.MSELoss()

        history: Dict[str, List[float]] = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Optional[Dict] = None

        for epoch in range(num_epochs):
            # ---- Training ----
            self.train()
            train_losses: List[float] = []

            for batch_input, batch_target in train_loader:
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)

                self._optimizer.zero_grad()
                predictions = self(batch_input)
                loss = self._criterion(predictions, batch_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self._optimizer.step()
                train_losses.append(loss.item())

            epoch_train_loss = np.mean(train_losses)
            history["train_loss"].append(float(epoch_train_loss))

            # ---- Validation ----
            if val_loader is not None:
                self.eval()
                val_losses: List[float] = []
                with torch.no_grad():
                    for batch_input, batch_target in val_loader:
                        batch_input = batch_input.to(device)
                        batch_target = batch_target.to(device)
                        predictions = self(batch_input)
                        loss = self._criterion(predictions, batch_target)
                        val_losses.append(loss.item())

                epoch_val_loss = np.mean(val_losses)
                history["val_loss"].append(float(epoch_val_loss))

                # Early stopping
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    best_state = copy.deepcopy(self.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(
                                f"Early stopping at epoch {epoch + 1} "
                                f"(best val loss: {best_val_loss:.6f})"
                            )
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} - "
                        f"Train Loss: {epoch_train_loss:.6f} - "
                        f"Val Loss: {epoch_val_loss:.6f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} - "
                        f"Train Loss: {epoch_train_loss:.6f}"
                    )

        # Restore best model
        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    def predict(
        self,
        data_loader,
        device: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for all batches in *data_loader*.

        Parameters
        ----------
        data_loader :
            PyTorch DataLoader yielding ``(input, target)`` batches.
        device : str, optional
            Override device (defaults to the device used during training).

        Returns
        -------
        predictions : np.ndarray
            Concatenated predictions.
        actuals : np.ndarray
            Concatenated ground-truth targets.
        """
        if device is None:
            device = self._device
        self.to(device)
        self.eval()

        all_predictions: List[np.ndarray] = []
        all_actuals: List[np.ndarray] = []

        with torch.no_grad():
            for batch_input, batch_target in data_loader:
                batch_input = batch_input.to(device)
                predictions = self(batch_input)
                all_predictions.append(predictions.cpu().numpy())
                all_actuals.append(batch_target.numpy())

        return np.concatenate(all_predictions, axis=0), np.concatenate(
            all_actuals, axis=0
        )


# =============================================================================
# 10. CosineWarmupLR
# =============================================================================

class CosineWarmupLR(_optim.lr_scheduler.LRScheduler):
    """Cosine annealing learning-rate schedule with linear warmup.

    During the first ``warmup_epochs`` the LR linearly increases from
    ``warmup_lr`` to ``base_lr``.  Afterwards a cosine decay is applied
    from ``base_lr`` down to zero over the remaining epochs.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimiser.
    warmup_epochs : int
        Number of linear-warmup epochs.
    total_epochs : int
        Total number of training epochs.
    base_lr : float
        Peak learning rate reached at the end of warmup.
    warmup_lr : float
        Starting learning rate for the warmup phase.
    last_epoch : int
        Index of the last epoch (for resume).  ``-1`` starts fresh.
    """

    def __init__(
        self,
        optimizer: _optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        warmup_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """Compute the learning rate for the current epoch.

        Returns linear warmup LR during the warmup phase, then cosine
        decay for the remainder.

        Returns
        -------
        list[float]
            One learning rate per optimizer parameter group.
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup.
            # PATCH: was last_epoch / warmup_epochs, which made the entire
            # first epoch train at exactly LR = 0 (a wasted epoch).
            alpha = min(1.0, (self.last_epoch + 1) / max(1, self.warmup_epochs))
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * alpha
            return [lr] * len(self.optimizer.param_groups)
        else:
            # Cosine annealing
            progress = float(self.last_epoch - self.warmup_epochs) / float(
                max(1, self.total_epochs - self.warmup_epochs)
            )
            lr = max(
                0.0,
                self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )
            return [lr] * len(self.optimizer.param_groups)
