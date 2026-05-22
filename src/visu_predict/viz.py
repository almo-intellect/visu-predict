"""Lightweight visualisations: training history, predictions vs actuals, attention."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_training_history(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    output_path: str | Path,
    title: str = "Training History",
) -> Path:
    """Save a PNG with train/val loss curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved training history plot to %s", output_path)
    return output_path


def plot_predictions_vs_actual(
    predictions: np.ndarray,
    actuals: np.ndarray,
    output_path: str | Path,
    sensor_ids: Optional[Sequence[int]] = None,
    max_sensors: int = 6,
    horizon: Optional[int] = None,
) -> Path:
    """Plot prediction vs actual time series for a handful of sensors."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if predictions.ndim == 3:
        if horizon is None:
            horizon = predictions.shape[1] - 1
        predictions = predictions[:, horizon, :]
        actuals = actuals[:, horizon, :]

    num_sensors = predictions.shape[-1]
    chosen = sensor_ids[:max_sensors] if sensor_ids else range(min(max_sensors, num_sensors))

    fig, axes = plt.subplots(len(list(chosen)), 1, figsize=(10, 2.2 * len(list(chosen))), sharex=True)
    if len(list(chosen)) == 1:
        axes = [axes]

    for ax, sid in zip(axes, chosen):
        ax.plot(actuals[:, sid], label="actual", linewidth=1.2)
        ax.plot(predictions[:, sid], label="prediction", linewidth=1.2, alpha=0.85)
        ax.set_title(f"Sensor {sid}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved predictions plot to %s", output_path)
    return output_path


def plot_attention_weights(
    weights: np.ndarray,
    output_path: str | Path,
    title: str = "Attention Weights",
) -> Path:
    """Heatmap of an attention map. ``weights`` shape: ``[seq, seq]`` or ``[heads, seq, seq]``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if weights.ndim == 3:
        weights = weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(weights, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
