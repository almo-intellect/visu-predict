"""Shared fixtures: a small synthetic sensor network written in the input-file format.

Tests that need the real METR-LA / PEMS-BAY files run only when the
environment variable ``VISU_DATA_DIR`` points at a folder containing them
(``visu-predict download --dest <folder>``).
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

SMALL_MODEL = ["--dim-input", "8", "--dim-tod", "8", "--dim-dow", "8", "--dim-adaptive", "8",
               "--ff-dim", "32", "--heads", "2", "--t-layers", "1", "--s-layers", "1"]


def write_dataset(root: str, name: str = "SYN", timesteps: int = 2000, sensors: int = 6, seed: int = 0):
    """Daily-cycle traffic with noise and a few zero (missing) readings, plus a chain adjacency."""
    idx = pd.date_range("2024-01-01", periods=timesteps, freq="5min")
    rng = np.random.default_rng(seed)
    base = 50 + 10 * np.sin(2 * np.pi * np.arange(timesteps) / 288)[:, None]
    vals = base + rng.normal(0, 1, (timesteps, sensors))
    vals[rng.integers(0, timesteps, 50), rng.integers(0, sensors, 50)] = 0.0
    pd.DataFrame(vals, index=idx, columns=[f"s{i}" for i in range(sensors)]).to_csv(
        os.path.join(root, f"{name}.csv"))
    adj = np.eye(sensors, dtype=np.float32)
    for i in range(sensors - 1):
        adj[i, i + 1] = 0.8
    with open(os.path.join(root, f"adj_{name}.pkl"), "wb") as f:
        pickle.dump([[f"s{i}" for i in range(sensors)], {f"s{i}": i for i in range(sensors)}, adj], f)
    return vals


@pytest.fixture(scope="session")
def syn(tmp_path_factory):
    """(data folder, raw values) of the synthetic dataset ``SYN``."""
    root = tmp_path_factory.mktemp("data")
    vals = write_dataset(str(root))
    return str(root), vals


@pytest.fixture
def make_dataset(tmp_path):
    """Factory writing a fresh synthetic dataset into this test's own folder."""
    def make(name: str = "SYN", **kwargs):
        return str(tmp_path), write_dataset(str(tmp_path), name, **kwargs)
    return make


@pytest.fixture(scope="session")
def real_data_dir():
    path = os.environ.get("VISU_DATA_DIR")
    if not path or not os.path.exists(os.path.join(path, "METR-LA.csv")):
        pytest.skip("set VISU_DATA_DIR to a folder with the METR-LA / PEMS-BAY files")
    return path
