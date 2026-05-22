"""Resolve dataset-related auxiliary files (adjacency matrix, coordinates, weather)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ADJACENCY_FILENAMES: dict[str, str] = {
    "METR-LA": "adj_METR-LA.pkl",
    "PEMS-BAY": "adj_PEMS-BAY.pkl",
    "PEMS-03": "adj_PEMS-03.pkl",
    "PEMS-04": "adj_PEMS-04.pkl",
    "PEMS-07": "adj_PEMS-07.pkl",
    "PEMS-08": "adj_PEMS-08.pkl",
}

COORDINATES_FILENAMES: dict[str, str] = {
    "METR-LA": "graph_sensor_locations_metr_la.csv",
    "PEMS-BAY": "graph_sensor_locations_pems_bay.csv",
    "PEMS-03": "graph_sensor_locations_pems_03.csv",
    "PEMS-04": "graph_sensor_locations_pems_04.csv",
    "PEMS-07": "graph_sensor_locations_pems_07.csv",
    "PEMS-08": "graph_sensor_locations_pems_08.csv",
}


def _first_existing(candidates: list[Path]) -> Optional[Path]:
    for c in candidates:
        if c.exists():
            return c
    return None


def find_adjacency_matrix(dataset_name: str, input_dir: str | Path) -> Optional[Path]:
    """Locate adjacency matrix pickle for a dataset within input_dir."""
    filename = ADJACENCY_FILENAMES.get(dataset_name, f"adj_{dataset_name}.pkl")
    input_dir = Path(input_dir)
    candidates = [
        input_dir / filename,
        input_dir / "adjacency" / filename,
        Path.cwd() / filename,
    ]
    result = _first_existing(candidates)
    if result is None:
        logger.warning("Adjacency matrix not found for %s (looked in %s)", dataset_name, input_dir)
    return result


def find_coordinates(dataset_name: str, input_dir: str | Path) -> Optional[Path]:
    """Locate sensor coordinates CSV for a dataset within input_dir."""
    input_dir = Path(input_dir)
    filename = COORDINATES_FILENAMES.get(dataset_name, f"graph_sensor_locations_{dataset_name}.csv")
    candidates = [
        input_dir / filename,
        input_dir / "graph_sensor_locations.csv",
    ]
    return _first_existing(candidates)
