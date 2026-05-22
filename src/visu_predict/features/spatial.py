"""Spatial features: adjacency matrix loading, node embeddings."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_adjacency_matrix(
    path: str | Path, fallback_size: int = 10,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Load an adjacency matrix pickle, with a safe identity-matrix fallback.

    Supports two pickle layouts: ``[sensor_ids, node_ids, adj_matrix]`` (DCRNN format)
    or a bare ``np.ndarray``.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Adjacency file missing: %s — falling back to identity(%d)", path, fallback_size)
        return _identity_fallback(fallback_size)

    with path.open("rb") as f:
        try:
            graph_data = pickle.load(f, encoding="latin1")
        except (UnicodeDecodeError, TypeError):
            f.seek(0)
            graph_data = pickle.load(f)

    if isinstance(graph_data, list) and len(graph_data) >= 3:
        sensor_ids, node_ids, adj = graph_data[:3]
        if adj.shape[0] != adj.shape[1]:
            raise ValueError(f"Adjacency matrix is not square: {adj.shape}")
        return adj, list(sensor_ids), list(node_ids)

    if isinstance(graph_data, np.ndarray):
        if graph_data.shape[0] != graph_data.shape[1]:
            raise ValueError(f"Adjacency matrix is not square: {graph_data.shape}")
        n = graph_data.shape[0]
        return graph_data, [f"sensor_{i}" for i in range(n)], list(range(n))

    raise ValueError(f"Unexpected adjacency pickle structure: {type(graph_data)}")


def _identity_fallback(n: int) -> tuple[np.ndarray, list[str], list[int]]:
    return np.eye(n), [f"sensor_{i}" for i in range(n)], list(range(n))


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetric GCN normalisation: D^(-1/2) (A + I) D^(-1/2)."""
    adj = adj + np.eye(adj.shape[0])
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def gaussian_distance_adjacency(
    coordinates: np.ndarray, threshold: float = 0.1, sigma: float = 0.1,
) -> np.ndarray:
    """Build an adjacency matrix from coordinates via Gaussian kernel."""
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    if distances.max() > 0:
        distances = distances / distances.max()
    adj = np.exp(-(distances ** 2) / (sigma ** 2))
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj


def _coordinate_positional_embedding(
    coordinates: np.ndarray, embedding_dim: int,
) -> np.ndarray:
    """Sinusoidal encoding of 2D coordinates, à la Transformer positional encoding."""
    n = coordinates.shape[0]
    emb = np.zeros((n, embedding_dim))
    for j in range(0, embedding_dim - 3, 4):
        div = 1.0 / np.power(10000.0, j / embedding_dim)
        emb[:, j] = np.sin(coordinates[:, 0] * div)
        emb[:, j + 1] = np.cos(coordinates[:, 0] * div)
        emb[:, j + 2] = np.sin(coordinates[:, 1] * div)
        emb[:, j + 3] = np.cos(coordinates[:, 1] * div)
    return emb


class SpatialIntegration:
    """Owns adjacency matrix and node embeddings for the spatial pathway."""

    def __init__(
        self,
        adjacency_path: str | Path | None = None,
        coordinates_path: str | Path | None = None,
        num_sensors: int = 207,
        spatial_dim: int = 207,
        embedding_dim: int = 207,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_sensors = num_sensors
        self.spatial_dim = spatial_dim
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)

        if adjacency_path and Path(adjacency_path).exists():
            self.adjacency_matrix, self.sensor_ids, self.node_ids = load_adjacency_matrix(
                adjacency_path, fallback_size=num_sensors,
            )
            self.num_sensors = self.adjacency_matrix.shape[0]
        else:
            logger.warning("No adjacency provided; using identity(%d)", num_sensors)
            self.adjacency_matrix, self.sensor_ids, self.node_ids = _identity_fallback(num_sensors)

        self.coordinates = self._load_coordinates(coordinates_path)
        self.normalized_adjacency = normalize_adjacency(self.adjacency_matrix)

        self.adjacency_tensor = torch.tensor(self.adjacency_matrix, dtype=torch.float32, device=self.device)
        self.normalized_adjacency_tensor = torch.tensor(
            self.normalized_adjacency, dtype=torch.float32, device=self.device,
        )
        self.node_embeddings = self._build_node_embeddings()
        self._projection: nn.Linear | None = None

    def _load_coordinates(self, coordinates_path: str | Path | None) -> np.ndarray | None:
        if not coordinates_path:
            return None
        path = Path(coordinates_path)
        if not path.exists():
            logger.warning("Coordinates file missing: %s", path)
            return None

        df = pd.read_csv(path)
        rename = {"id": "sensor_id", "lat": "latitude", "lon": "longitude"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        required = {"sensor_id", "latitude", "longitude"}
        if not required.issubset(df.columns):
            raise ValueError(f"Coordinates file must contain {required}; got {set(df.columns)}")

        coords = np.zeros((self.num_sensors, 2))
        df_ids = df["sensor_id"].astype(str).tolist()
        for i, sid in enumerate(self.sensor_ids):
            sid_str = str(sid)
            if sid_str in df_ids:
                row = df[df["sensor_id"].astype(str) == sid_str].iloc[0]
                coords[i] = [row["latitude"], row["longitude"]]
            else:
                coords[i] = [i / self.num_sensors, i / self.num_sensors]

        return StandardScaler().fit_transform(coords)

    def _build_node_embeddings(self) -> torch.Tensor:
        if self.coordinates is not None:
            arr = _coordinate_positional_embedding(self.coordinates, self.embedding_dim)
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        emb = torch.empty(self.num_sensors, self.embedding_dim, device=self.device)
        nn.init.xavier_normal_(emb)
        return emb

    def spatial_features(self, batch_size: int) -> torch.Tensor:
        """Return ``[batch, num_sensors, spatial_dim]`` features."""
        emb = self.node_embeddings
        if self.embedding_dim != self.spatial_dim:
            if self._projection is None:
                self._projection = nn.Linear(self.embedding_dim, self.spatial_dim).to(self.device)
            emb = self._projection(emb)
        return emb.unsqueeze(0).expand(batch_size, -1, -1)
