"""Optional GNN encoder for pre-transformer node-feature mixing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    GCNConv = GATConv = None  # type: ignore[assignment]
    TORCH_GEOMETRIC_AVAILABLE = False


class GCNEncoder(nn.Module):
    """Stacked GCN or GAT encoder with optional residual + LayerNorm."""

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
            raise ImportError("PyTorch Geometric is required for GNN encoder")
        if gnn_type not in ("gcn", "gat"):
            raise ValueError(f"gnn_type must be 'gcn' or 'gat', got {gnn_type}")

        self.gnn_type = gnn_type
        self.use_residual = residual
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))

        if gnn_type == "gcn":
            in_dim = input_dim
            for _ in range(num_layers):
                self.layers.append(GCNConv(in_dim, hidden_dim))
                in_dim = hidden_dim
        else:
            head_out = max(1, hidden_dim // gat_heads) if gat_concat else hidden_dim
            self.layers.append(
                GATConv(input_dim, head_out, heads=gat_heads, concat=gat_concat,
                        dropout=dropout, add_self_loops=True)
            )
            middle_in = hidden_dim if gat_concat else head_out
            for _ in range(1, num_layers - 1):
                self.layers.append(
                    GATConv(middle_in, head_out, heads=gat_heads, concat=gat_concat,
                            dropout=dropout, add_self_loops=True)
                )
                middle_in = hidden_dim if gat_concat else head_out
            if num_layers > 1:
                self.layers.append(
                    GATConv(middle_in, hidden_dim, heads=1, concat=False,
                            dropout=dropout, add_self_loops=True)
                )

    @staticmethod
    def _to_edge_index(adj: torch.Tensor) -> torch.Tensor:
        return torch.stack(torch.nonzero(adj, as_tuple=True))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        edge_index = self._to_edge_index(adjacency)
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms, strict=True)):
            residual = x if self.use_residual and x.size(-1) == layer.out_channels else None
            x_new = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)
            if residual is not None:
                x_new = x_new + residual
            x = norm(x_new)
        return x
