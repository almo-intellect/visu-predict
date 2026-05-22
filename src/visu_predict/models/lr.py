"""Learning-rate schedules."""

from __future__ import annotations

import math

import torch


class CosineWarmupLR(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
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
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        n_groups = len(self.optimizer.param_groups)
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * alpha
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = max(0.0, self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return [lr] * n_groups
