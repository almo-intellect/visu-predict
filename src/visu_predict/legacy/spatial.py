"""
Spatial features module for the traffic prediction transformer.

Provides adjacency-matrix loading, spatial normalization, coordinate-based
node embeddings, and optional GNN encoding (GCN / GAT) when torch_geometric
is available.
"""

import math
import os
import pickle
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .utils import TORCH_GEOMETRIC_AVAILABLE

if TORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric.nn import GATConv, GCNConv


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def load_adjacency_matrix(
    adjacency_matrix_path: str,
    fallback_size: int = 10,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Load an adjacency matrix from a pickle file.

    Supported pickle formats:

    * **List format** -- ``[sensor_ids, node_ids, adj_matrix]``
    * **Direct numpy array** -- a bare ``np.ndarray``

    On any error an identity-matrix fallback of shape
    ``(fallback_size, fallback_size)`` is returned with synthetic IDs.

    Returns
    -------
    adj_matrix : np.ndarray
        The (possibly fallback) adjacency matrix.
    sensor_ids : list[str]
        Sensor identifier strings.
    node_ids : list[int]
        Integer node identifiers.
    """
    try:
        with open(adjacency_matrix_path, "rb") as f:
            raw = f.read()
        # PATCH: the original DCRNN adjacency pickles (METR-LA / PEMS-BAY)
        # are Python-2 pickles and need encoding='latin1'; previously the
        # UnicodeDecodeError silently produced an identity-matrix fallback.
        try:
            data = pickle.loads(raw)
        except UnicodeDecodeError:
            data = pickle.loads(raw, encoding="latin1")

        if isinstance(data, (list, tuple)) and len(data) == 3:
            sensor_ids, id_map, adj_matrix = data
            sensor_ids = [str(s) for s in sensor_ids]
            # PATCH: DCRNN stores element 1 as a {sensor_id: index} dict.
            if isinstance(id_map, dict):
                node_ids = [int(v) for v in id_map.values()]
            else:
                node_ids = [int(n) for n in id_map]
            adj_matrix = np.array(adj_matrix, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            adj_matrix = data.astype(np.float32)
            n = adj_matrix.shape[0]
            sensor_ids = [str(i) for i in range(n)]
            node_ids = list(range(n))
        else:
            raise ValueError(
                f"Unrecognised adjacency-matrix format: {type(data)}"
            )

        print(f"Loaded adjacency matrix of shape {adj_matrix.shape} "
              f"from {adjacency_matrix_path}")
        return adj_matrix, sensor_ids, node_ids

    except Exception as e:
        warnings.warn(
            f"Failed to load adjacency matrix from {adjacency_matrix_path}: "
            f"{e}.  Creating identity fallback of size {fallback_size}."
        )
        adj_matrix = np.eye(fallback_size, dtype=np.float32)
        sensor_ids = [str(i) for i in range(fallback_size)]
        node_ids = list(range(fallback_size))
        return adj_matrix, sensor_ids, node_ids


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalisation: D^{-1/2} (A + I) D^{-1/2}.

    Parameters
    ----------
    adj : np.ndarray
        Square adjacency matrix.

    Returns
    -------
    np.ndarray
        The symmetrically normalised adjacency matrix.
    """
    adj = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    d = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, np.power(d, -0.5), 0.0)
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ adj @ d_mat


def create_distance_adj_matrix(
    coordinates: np.ndarray,
    threshold: float = 0.1,
    sigma: float = 0.1,
) -> np.ndarray:
    """Build a Gaussian-kernel adjacency matrix from geographic coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Shape ``(N, 2)`` array of (latitude, longitude) pairs.
    threshold : float
        Entries below this value are zeroed out.
    sigma : float
        Bandwidth of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Shape ``(N, N)`` adjacency matrix.
    """
    n = coordinates.shape[0]
    # Pairwise squared Euclidean distances via broadcasting
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)

    adj = np.exp(-dist_sq / (2.0 * sigma ** 2))
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


# ---------------------------------------------------------------------------
# SpatialIntegration (singleton)
# ---------------------------------------------------------------------------

class SpatialIntegration:
    """Manages adjacency matrices, coordinate data, and node embeddings.

    Uses a singleton pattern -- obtain the shared instance via
    :meth:`get_instance` and tear it down (e.g. between tests) with
    :meth:`reset_instance`.
    """

    _instance: Optional["SpatialIntegration"] = None

    @classmethod
    def get_instance(cls) -> Optional["SpatialIntegration"]:
        """Return the singleton instance, or ``None`` if not yet created."""
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy the singleton instance so a new one can be created."""
        cls._instance = None

    def __init__(
        self,
        adjacency_matrix_path: str,
        coordinates_path: str,
        num_sensors: int = 207,
        spatial_dim: int = 207,
        embedding_dim: int = 207,
        device: str = "cpu",
    ) -> None:
        self.num_sensors = num_sensors
        self.spatial_dim = spatial_dim
        self.embedding_dim = embedding_dim
        self.device = device

        # --- Adjacency matrix ---
        try:
            self.adj_matrix, self.sensor_ids, self.node_ids = (
                load_adjacency_matrix(adjacency_matrix_path, fallback_size=num_sensors)
            )
        except Exception as e:
            warnings.warn(
                f"Could not load adjacency matrix: {e}. "
                f"Using identity fallback ({num_sensors}x{num_sensors})."
            )
            self.adj_matrix = np.eye(num_sensors, dtype=np.float32)
            self.sensor_ids = [str(i) for i in range(num_sensors)]
            self.node_ids = list(range(num_sensors))

        # --- Coordinates ---
        self.coordinates = self._load_coordinates(coordinates_path)

        # --- Normalised adjacency ---
        self.normalized_adj = normalize_adj(self.adj_matrix)

        # --- Torch tensors ---
        self.adj_tensor = (
            torch.tensor(self.adj_matrix, dtype=torch.float32).to(self.device)
        )
        self.normalized_adj_tensor = (
            torch.tensor(self.normalized_adj, dtype=torch.float32).to(self.device)
        )

        # --- Node embeddings ---
        self.node_embeddings = self._create_node_embeddings()

        # Register as singleton
        SpatialIntegration._instance = self

    # -- private helpers -----------------------------------------------------

    def _load_coordinates(
        self, coordinates_path: str
    ) -> Optional[np.ndarray]:
        """Load sensor coordinates from a CSV with columns
        ``sensor_id``, ``latitude``, ``longitude``.

        Returns
        -------
        np.ndarray or None
            Shape ``(N, 2)`` array of ``[latitude, longitude]`` rows, or
            ``None`` on failure.
        """
        try:
            df = pd.read_csv(coordinates_path)
            coords = df[["latitude", "longitude"]].values.astype(np.float64)
            print(f"Loaded coordinates for {len(coords)} sensors "
                  f"from {coordinates_path}")
            return coords
        except Exception as e:
            warnings.warn(f"Failed to load coordinates: {e}")
            return None

    def _create_node_embeddings(self) -> torch.Tensor:
        """Create positional-style node embeddings.

        When coordinates are available a sinusoidal encoding is built using
        vectorised broadcasting (Fix #4).  Otherwise Xavier-initialised
        learnable parameters are returned.
        """
        if self.coordinates is not None:
            coordinate_embedding = np.zeros(
                (self.num_sensors, self.embedding_dim), dtype=np.float64
            )
            # Ensure we only fill complete groups of 4
            max_j = (self.embedding_dim // 4) * 4
            freqs = 1.0 / np.power(
                10000, np.arange(0, max_j, 4) / self.embedding_dim
            )
            # Vectorized: [num_sensors] x [num_freqs] -> [num_sensors, num_freqs]
            lat_sin = np.sin(np.outer(self.coordinates[:, 0], freqs))
            lat_cos = np.cos(np.outer(self.coordinates[:, 0], freqs))
            lon_sin = np.sin(np.outer(self.coordinates[:, 1], freqs))
            lon_cos = np.cos(np.outer(self.coordinates[:, 1], freqs))

            coordinate_embedding[:, 0:max_j:4] = lat_sin
            coordinate_embedding[:, 1:max_j:4] = lat_cos
            coordinate_embedding[:, 2:max_j:4] = lon_sin
            coordinate_embedding[:, 3:max_j:4] = lon_cos

            return torch.tensor(
                coordinate_embedding, dtype=torch.float32
            ).to(self.device)
        else:
            # PATCH: a bare nn.Parameter outside any module never trains;
            # return a plain initialised tensor instead.
            embeddings = torch.empty(self.num_sensors, self.embedding_dim)
            nn.init.xavier_normal_(embeddings)
            return embeddings.to(self.device)

    # -- public API ----------------------------------------------------------

    def get_adjacency_matrix(self) -> torch.Tensor:
        """Return the raw adjacency matrix as a tensor."""
        return self.adj_tensor

    def get_normalized_adjacency_matrix(self) -> torch.Tensor:
        """Return the symmetrically normalised adjacency tensor."""
        return self.normalized_adj_tensor

    def get_node_embeddings(self) -> torch.Tensor:
        """Return the node-embedding tensor."""
        return self.node_embeddings

    def get_projected_embeddings(self) -> np.ndarray:
        """Return node embeddings as a ``(num_sensors, embedding_dim)`` array.

        PATCH: ``TrafficDataset.create_spatial_features`` called this method,
        which did not exist; it is now the documented numpy accessor.
        """
        emb = self.node_embeddings
        if isinstance(emb, torch.Tensor):
            return emb.detach().cpu().numpy()
        return np.asarray(emb)

    def get_spatial_features(self, batch_size: int) -> torch.Tensor:
        """Expand node embeddings to match a batch dimension.

        Parameters
        ----------
        batch_size : int
            Number of samples in the current batch.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, num_sensors, embedding_dim)``.
        """
        return self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

    # -- factory -------------------------------------------------------------

    @staticmethod
    def create_spatial_integration_from_config(
        config, device: str = "cpu"
    ) -> "SpatialIntegration":
        """Build a :class:`SpatialIntegration` from a config dataclass.

        Resolves the adjacency-matrix path via
        ``config.get_adjacency_matrix_path`` (imported locally to avoid
        circular imports).

        Parameters
        ----------
        config :
            A configuration object that exposes at least
            ``num_sensors``, ``spatial_dim``, ``embedding_dim``, and
            ``coordinates_path`` attributes.
        device : str
            PyTorch device string.
        """
        from .config import get_adjacency_matrix_path

        adjacency_path = get_adjacency_matrix_path(config)
        # PATCH: the config fields are ``coordinates_file`` and
        # ``spatial_feature_dim``; the old names here never existed, so this
        # factory raised on every call and spatial features silently failed.
        return SpatialIntegration(
            adjacency_matrix_path=adjacency_path or "",
            coordinates_path=getattr(config, "coordinates_file", None) or "",
            num_sensors=getattr(config, "num_sensors", 207),
            spatial_dim=getattr(config, "spatial_feature_dim", 207),
            embedding_dim=getattr(config, "embedding_dim", 207),
            device=device,
        )


# ---------------------------------------------------------------------------
# GCNEncoder (requires torch_geometric)
# ---------------------------------------------------------------------------

class GCNEncoder(nn.Module):
    """GNN encoder supporting both GCN and GAT convolutions.

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

    @staticmethod
    def _dense_to_sparse_weighted(adj_matrix: torch.Tensor):
        """PATCH: also return edge weights (previously discarded)."""
        edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
        edge_weight = adj_matrix[edge_index[0], edge_index[1]]
        return edge_index, edge_weight

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
        edge_index, edge_weight = self._dense_to_sparse_weighted(adjacency_matrix)

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
