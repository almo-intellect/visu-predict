"""Transfer learning utilities: load a pre-trained checkpoint and fine-tune on a new dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from visu_predict.config import TrainingConfig

logger = logging.getLogger(__name__)


def load_pretrained(model: nn.Module, checkpoint_path: str | Path, strict: bool = False) -> nn.Module:
    """Load weights from a checkpoint dict produced by ``training.train``."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = state.get("model_state_dict", state)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.info("Missing keys when loading pretrained weights: %d", len(missing))
    if unexpected:
        logger.info("Unexpected keys when loading pretrained weights: %d", len(unexpected))
    return model


def freeze_for_fine_tuning(
    model: nn.Module,
    freeze_encoder: bool = True,
    num_layers: int = 1,
) -> dict[str, int]:
    """Freeze parameters per the transfer-learning config; return param counts."""
    if hasattr(model, "freeze_layers"):
        model.freeze_layers(freeze_encoder=freeze_encoder, num_layers=num_layers)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %d / %d (%.1f%%)",
        trainable, total, 100.0 * trainable / max(1, total),
    )
    return {"trainable": trainable, "total": total}


def transfer_config_overrides(config: TrainingConfig) -> TrainingConfig:
    """Apply transfer-learning specific tweaks to a copy of ``config``."""
    config.learning_rate = config.transfer_learning_rate
    return config
