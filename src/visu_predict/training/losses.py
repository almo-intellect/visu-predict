"""Loss functions and regression metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def quantile_loss(output: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """Pinball loss over a list of quantiles.

    Output is expected to have shape ``[batch, pred_len, num_quantiles]`` and
    target ``[batch, pred_len]``.
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - output[..., i]
        losses.append(torch.maximum((q - 1) * errors, q * errors).mean())
    return torch.stack(losses).sum()


def hybrid_loss(output: torch.Tensor, target: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Weighted combination of MSE and MAE."""
    return alpha * nn.functional.mse_loss(output, target) + (1 - alpha) * nn.functional.l1_loss(output, target)


def robust_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """MAPE that ignores zero targets and adds a small epsilon to the denominator."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100)


def make_criterion(loss_function: str) -> nn.Module:
    """Map a config string ('mae', 'mse', 'huber') to an nn.Module loss."""
    if loss_function == "mae":
        return nn.L1Loss()
    if loss_function == "mse":
        return nn.MSELoss()
    if loss_function == "huber":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss_function: {loss_function!r}")
