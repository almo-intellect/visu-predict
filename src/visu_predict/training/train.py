"""Train / evaluate / predict loops for the traffic transformer."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from visu_predict.config import TrainingConfig
from visu_predict.training.losses import hybrid_loss, quantile_loss, robust_mape

logger = logging.getLogger(__name__)

try:
    from torch.amp import GradScaler, autocast
    _AMP_KWARGS = {"device_type": "cuda"}
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast  # type: ignore[no-redef]
        _AMP_KWARGS: dict[str, str] = {}
        AMP_AVAILABLE = True
    except ImportError:
        GradScaler = autocast = None  # type: ignore[assignment]
        _AMP_KWARGS = {}
        AMP_AVAILABLE = False


Metrics = tuple[float, float, float, float]


def _to_device(data: Any, target: torch.Tensor, device: torch.device) -> tuple[Any, torch.Tensor]:
    if isinstance(data, dict):
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
    else:
        data = data.to(device, non_blocking=True)
    return data, target.to(device, non_blocking=True)


def _compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    criterion: Callable,
    config: TrainingConfig,
) -> torch.Tensor:
    if config.use_quantile_regression:
        return quantile_loss(output, target, config.quantiles)
    if config.loss_function == "hybrid":
        return hybrid_loss(output, target)
    return criterion(output, target)


def _forward(
    model: nn.Module,
    data: Any,
    target: Optional[torch.Tensor],
    adjacency_matrix: Optional[torch.Tensor],
    config: TrainingConfig,
    *,
    teacher_forcing: bool,
) -> torch.Tensor:
    uses_transformer_decoder = config.decoder_type == "transformer"
    needs_adj = config.use_spatial_features and config.use_gnn_pre_transformer

    kwargs: dict[str, Any] = {}
    if needs_adj and adjacency_matrix is not None:
        kwargs["adjacency_matrix"] = adjacency_matrix

    if uses_transformer_decoder and teacher_forcing and target is not None:
        return model(data, target=target, **kwargs)
    return model(data, **kwargs)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    config: TrainingConfig,
    data_scaler: Any,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: str | torch.device = "cpu",
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> tuple[nn.Module, list[float], list[float]]:
    """Train the model with early stopping. Returns (model, train_losses, val_losses)."""
    device = torch.device(device)
    model.to(device)

    use_amp = config.use_mixed_precision and AMP_AVAILABLE and device.type == "cuda"
    scaler = GradScaler(**_AMP_KWARGS) if use_amp else None

    best_loss = float("inf")
    no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    adj_on_device = adjacency_matrix.to(device) if adjacency_matrix is not None else None

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = _to_device(data, target, device)

            if use_amp:
                with autocast(**_AMP_KWARGS):
                    output = _forward(model, data, target, adj_on_device, config, teacher_forcing=True)
                    loss = _compute_loss(output, target, criterion, config) / config.accumulation_steps
                scaler.scale(loss).backward()
            else:
                output = _forward(model, data, target, adj_on_device, config, teacher_forcing=True)
                loss = _compute_loss(output, target, criterion, config) / config.accumulation_steps
                loss.backward()

            is_step = (batch_idx + 1) % config.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader)
            if is_step:
                if config.gradient_clip is not None:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * config.accumulation_steps

        avg_train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        val_loss, _ = evaluate(model, val_loader, criterion, data_scaler, config, device, adjacency_matrix=adj_on_device)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            if config.model_dir is not None:
                _save_checkpoint(model, optimizer, epoch, best_loss, config)
        else:
            no_improve += 1
            if no_improve >= config.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if scheduler is not None:
            if config.scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f",
            epoch + 1, config.num_epochs, avg_train_loss, val_loss,
        )

    return model, train_losses, val_losses


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: TrainingConfig,
) -> None:
    assert config.model_dir is not None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(config.model_dir) / f"best_model_{timestamp}.pth"
    try:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "config": config.to_dict(),
            },
            path,
        )
    except OSError as exc:
        logger.warning("Could not save checkpoint to %s: %s", path, exc)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    data_scaler: Any,
    config: TrainingConfig,
    device: str | torch.device = "cpu",
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> tuple[float, Metrics]:
    """Compute validation loss and (MAE, RMSE, R², MAPE) in original scale."""
    device = torch.device(device)
    model.eval()

    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    use_amp = config.use_mixed_precision and AMP_AVAILABLE and device.type == "cuda"
    adj_on_device = adjacency_matrix.to(device) if adjacency_matrix is not None and adjacency_matrix.device != device else adjacency_matrix

    with torch.no_grad():
        for data, target in dataloader:
            data, target = _to_device(data, target, device)
            if use_amp:
                with autocast(**_AMP_KWARGS):
                    output = _forward(model, data, None, adj_on_device, config, teacher_forcing=False)
                    loss = _compute_loss(output, target, criterion, config)
            else:
                output = _forward(model, data, None, adj_on_device, config, teacher_forcing=False)
                loss = _compute_loss(output, target, criterion, config)

            total_loss += loss.item()
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_targets)
    num_features = predictions.shape[-1]
    preds_2d = predictions.reshape(-1, num_features)
    acts_2d = actuals.reshape(-1, num_features)

    preds_inv = data_scaler.inverse_transform(preds_2d)
    acts_inv = data_scaler.inverse_transform(acts_2d)

    mae = float(mean_absolute_error(acts_inv.ravel(), preds_inv.ravel()))
    rmse = float(np.sqrt(mean_squared_error(acts_inv.ravel(), preds_inv.ravel())))
    r2 = float(r2_score(acts_inv.ravel(), preds_inv.ravel()))
    mape = robust_mape(acts_inv.ravel(), preds_inv.ravel())

    logger.info("eval | MAE=%.3f RMSE=%.3f R²=%.3f MAPE=%.2f%%", mae, rmse, r2, mape)
    return total_loss / max(1, len(dataloader)), (mae, rmse, r2, mape)


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    data_scaler: Any,
    config: TrainingConfig,
    device: str | torch.device = "cpu",
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions; returns (predictions, actuals) in original scale."""
    device = torch.device(device)
    model.eval()
    adj_on_device = adjacency_matrix.to(device) if adjacency_matrix is not None else None

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = _to_device(data, target, device)
            output = _forward(model, data, None, adj_on_device, config, teacher_forcing=False)
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_targets)
    num_features = predictions.shape[-1]
    preds_inv = data_scaler.inverse_transform(predictions.reshape(-1, num_features))
    acts_inv = data_scaler.inverse_transform(actuals.reshape(-1, num_features))
    return preds_inv, acts_inv
