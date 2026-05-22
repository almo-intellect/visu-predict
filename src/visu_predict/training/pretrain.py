"""Masked-reconstruction pretraining for the STAE pipeline.

Random ``(time, sensor)`` cells of the traffic input are masked out and
replaced with a learnable mask token at the embedding stage. The model is
trained to reconstruct the original values at those masked positions only.
After pretraining, the encoder weights can be transferred to a downstream
forecasting head via :func:`save_pretrained_encoder` /
:func:`load_pretrained_encoder`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from visu_predict.config import TrainingConfig
from visu_predict.data import TrafficDataset

logger = logging.getLogger(__name__)


class MaskedSTDataset(TrafficDataset):
    """Variant of :class:`TrafficDataset` that emits a cell-level mask.

    Returns ``(features_dict_with_mask, full_traffic_window)``. The
    ``"mask"`` entry of the features dict is a boolean tensor of shape
    ``[seq_length, num_sensors]`` where ``True`` indicates a masked cell.
    The target is the **full** (unmasked) traffic window — the loss should
    be applied only at masked positions.
    """

    def __init__(
        self,
        data: np.ndarray,
        timestamps: pd.DatetimeIndex | None,
        config: TrainingConfig,
        mask_ratio: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(data, timestamps, config)
        self.mask_ratio = mask_ratio if mask_ratio is not None else config.mask_ratio
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {self.mask_ratio}")
        self._rng = rng or np.random.default_rng(config.seed)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        features, _target = super().__getitem__(idx)
        # Reconstruction target is the *input* traffic window itself (not the
        # future prediction target), so the model fills in masked cells.
        traffic_window = features["traffic"]
        mask = torch.from_numpy(
            self._rng.random(traffic_window.shape) < self.mask_ratio
        )
        features["mask"] = mask
        # Zero out masked positions in the traffic feature so the model can't
        # cheat by reading the value; the model's mask_token replaces them at
        # the embedding stage.
        features["traffic"] = traffic_window.masked_fill(mask, 0.0)
        return features, traffic_window


def masked_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error computed only over masked cells.

    Args:
        pred: ``[B, T, N]`` model reconstruction.
        target: ``[B, T, N]`` original unmasked traffic.
        mask: ``[B, T, N]`` boolean mask (True = compute loss here).
    """
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype, requires_grad=True)
    diff = (pred - target) ** 2
    return diff[mask].mean()


def pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    val_loader: DataLoader | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, list[float], list[float]]:
    """Run masked-reconstruction pretraining.

    The model must expose a ``forward_reconstruct(features, mask) -> [B, T, N]``
    method (added on ``TrafficTransformer`` in this PR).
    """
    if not hasattr(model, "forward_reconstruct"):
        raise AttributeError(
            "model has no forward_reconstruct method; pretraining requires the "
            "STAE pipeline (set model_pipeline='stae')."
        )

    device = torch.device(device)
    model.to(device)
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(config.pretrain_num_epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for features, target in train_loader:
            features = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()
            }
            target = target.to(device, non_blocking=True)
            mask = features["mask"]
            pred = model.forward_reconstruct(features, mask)
            loss = masked_reconstruction_loss(pred, target, mask)
            optimizer.zero_grad()
            loss.backward()
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            running += loss.item()
            n_batches += 1
        avg = running / max(1, n_batches)
        train_losses.append(avg)

        if val_loader is not None:
            val_loss = _evaluate_pretrain(model, val_loader, device)
            val_losses.append(val_loss)
            logger.info("pretrain epoch %d | train=%.4f val=%.4f", epoch + 1, avg, val_loss)
        else:
            logger.info("pretrain epoch %d | train=%.4f", epoch + 1, avg)

        if scheduler is not None:
            if config.scheduler_type == "plateau" and val_loader is not None:
                scheduler.step(val_losses[-1])
            else:
                scheduler.step()

    return model, train_losses, val_losses


def _evaluate_pretrain(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    running = 0.0
    n_batches = 0
    with torch.no_grad():
        for features, target in loader:
            features = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()
            }
            target = target.to(device, non_blocking=True)
            mask = features["mask"]
            pred = model.forward_reconstruct(features, mask)
            running += masked_reconstruction_loss(pred, target, mask).item()
            n_batches += 1
    return running / max(1, n_batches)


# ---- Checkpoint helpers ----------------------------------------------------

_ENCODER_PREFIXES = (
    "stae_composer.",
    "stae_attn_stack.",
    "stae_patch.",
    "adaptive_adj.",
    "mask_token",
)


def _encoder_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value for key, value in model.state_dict().items()
        if any(key.startswith(prefix) or key == prefix for prefix in _ENCODER_PREFIXES)
    }


def save_pretrained_encoder(model: nn.Module, path: str | Path) -> Path:
    """Save only the encoder + embeddings (not the forecasting head)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state_dict": _encoder_state(model)}, path)
    logger.info("Pretrained encoder saved to %s", path)
    return path


def load_pretrained_encoder(model: nn.Module, path: str | Path) -> nn.Module:
    """Load encoder weights from a pretraining checkpoint (non-strict)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained encoder not found: {path}")
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("encoder_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.info("Missing keys when loading pretrained encoder: %d", len(missing))
    if unexpected:
        logger.info("Unexpected keys when loading pretrained encoder: %d", len(unexpected))
    return model
