"""
Training module for the traffic prediction transformer.

Provides the core training loop, evaluation, prediction, loss functions,
and factory helpers for optimizers, schedulers, and criteria.
"""

import gc
import os
import shutil
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .config import TrainingConfig
from .model import CosineWarmupLR
from .utils import (
    AMP_AVAILABLE,
    DEVICE_TYPE_SUPPORTED,
    get_maputo_timestamp,
    make_autocast,
    make_grad_scaler,
)


# =============================================================================
# Forward-pass helpers  (KEY FIX #6)
# =============================================================================

def _forward_pass(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    config: TrainingConfig,
    adjacency_matrix: Optional[torch.Tensor],
    device: str,
    uses_transformer_decoder: bool,
) -> torch.Tensor:
    """Unified forward pass handling spatial/GNN/decoder variants.

    PATCH: the adjacency matrix used to be forwarded only when
    ``use_spatial_features`` AND ``use_gnn_pre_transformer`` were both set —
    with spatial features off, the GNN silently never received its graph.
    It is now forwarded whenever it exists (a plain ``model(data)`` call is
    kept for adjacency-free models such as the LSTM baseline).
    """
    if uses_transformer_decoder:
        if adjacency_matrix is not None:
            return model(data, target, adjacency_matrix)
        return model(data, target)
    else:
        if adjacency_matrix is not None:
            return model(data, adjacency_matrix=adjacency_matrix)
        return model(data)


def _inference_forward(
    model: nn.Module,
    data: torch.Tensor,
    config: Optional[TrainingConfig],
    adjacency_matrix: Optional[torch.Tensor],
    device: str,
) -> torch.Tensor:
    """Forward pass for evaluation/prediction (no teacher forcing target).

    PATCH: see _forward_pass — adjacency is passed whenever it exists.
    """
    if adjacency_matrix is not None:
        return model(data, adjacency_matrix=adjacency_matrix)
    return model(data)


def _compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    criterion: Any,
    config: TrainingConfig,
) -> torch.Tensor:
    """Unified loss computation."""
    if config.use_quantile_regression:
        return quantile_loss(output, target, config.quantiles)
    elif config.loss_function == 'hybrid':
        return hybrid_loss(output, target)
    return criterion(output, target)


# =============================================================================
# AMP context helper
# =============================================================================

def _get_autocast_context(use_amp: bool, device: str) -> Any:
    """Return the appropriate autocast context manager or a no-op.

    PATCH: delegates to utils.make_autocast, which uses the modern
    ``torch.amp`` API (the old DEVICE_TYPE_SUPPORTED probe was broken and
    always selected the deprecated ``torch.cuda.amp`` path).
    """
    return make_autocast(device, enabled=use_amp)


# =============================================================================
# Batch processing helper
# =============================================================================

def _process_batch(
    data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    target: torch.Tensor,
    device: str,
) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
    """Move batch data and target to *device*, handling both dict and tensor inputs."""
    if isinstance(data, dict):
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    else:
        data = data.to(device)
    target = target.to(device)
    return data, target


# =============================================================================
# Loss functions
# =============================================================================

def quantile_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    quantiles: Optional[List[float]] = None,
) -> torch.Tensor:
    """Compute the pinball (quantile) loss across the given quantiles.

    Parameters
    ----------
    output : torch.Tensor
        Model predictions.  If the last dimension equals ``len(quantiles)``,
        each slice along that dimension is treated as a separate quantile
        prediction.
    target : torch.Tensor
        Ground-truth values.
    quantiles : list of float, optional
        Quantile levels.  Defaults to ``[0.1, 0.5, 0.9]``.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    losses: List[torch.Tensor] = []
    for i, q in enumerate(quantiles):
        if output.shape[-1] == len(quantiles):
            pred = output[..., i]
        else:
            pred = output
        errors = target.squeeze(-1) - pred if pred.dim() < target.dim() else target - pred
        losses.append(torch.max((q - 1) * errors, q * errors).mean())
    return torch.stack(losses).mean()


def hybrid_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Convex combination of MSE and MAE losses.

    Parameters
    ----------
    output, target : torch.Tensor
        Prediction and ground-truth tensors.
    alpha : float
        Weight for MSE; ``(1 - alpha)`` is applied to MAE.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    mse = nn.MSELoss()(output, target)
    mae = nn.L1Loss()(output, target)
    return alpha * mse + (1 - alpha) * mae


def masked_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    null_val: float = 0.0,
    threshold: float = 1e-4,
) -> float:
    """MAPE that excludes entries where the target equals *null_val*.

    PATCH: in METR-LA / PEMS-style data, zero readings are sensor faults, and
    dividing by (0 + eps) made MAPE explode to astronomical values. Masking
    near-null targets is the convention used by the DCRNN lineage.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values (real, inverse-transformed scale).
    null_val : float
        The value treated as missing/faulty (default 0.0).
    threshold : float
        Tolerance around *null_val*.

    Returns
    -------
    float
        Masked MAPE as a percentage; 0.0 if nothing survives the mask.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.abs(y_true - null_val) > threshold
    if not np.any(mask):
        return 0.0
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def robust_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error with a small epsilon to avoid division by zero.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values.
    epsilon : float
        Added to the denominator for numerical stability.

    Returns
    -------
    float
        MAPE as a percentage (0--100+ scale).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


# =============================================================================
# train_model
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any,
    criterion: Any,
    config: TrainingConfig,
    data_scaler: Optional[Union[MinMaxScaler, StandardScaler, RobustScaler]] = None,
    device: str = 'cpu',
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Run the full training loop with validation, early stopping, and checkpointing.

    Parameters
    ----------
    model : nn.Module
        The transformer model to train.
    train_loader, val_loader : DataLoader
        Training and validation data loaders.
    optimizer : torch.optim.Optimizer
        Optimiser instance.
    scheduler :
        Learning-rate scheduler (or ``None``).
    criterion :
        Loss function (or callable).
    config : TrainingConfig
        Experiment configuration.
    data_scaler : sklearn scaler, optional
        Fitted scaler used for inverse-transforming predictions.
    device : str
        Target device (``'cpu'`` or ``'cuda'``).
    adjacency_matrix : torch.Tensor, optional
        Pre-computed adjacency matrix for GNN variants.

    Returns
    -------
    tuple of (model, train_losses, val_losses)
    """
    num_epochs = config.num_epochs
    patience = config.patience
    accumulation_steps = max(1, config.accumulation_steps)

    # AMP setup (PATCH: modern torch.amp API via utils helper)
    use_amp = config.use_mixed_precision and AMP_AVAILABLE and device != 'cpu'
    scaler = make_grad_scaler(device, enabled=True) if use_amp else None

    # Detect decoder type
    uses_transformer_decoder = config.decoder_type == 'transformer'

    # Early stopping state
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses: List[float] = []
    val_losses: List[float] = []

    # Model directory for checkpoints
    model_dir = config.model_dir or '.'

    model.to(device)

    # PATCH: move the adjacency matrix to the device once instead of on
    # every batch.
    if adjacency_matrix is not None:
        adjacency_matrix = adjacency_matrix.to(device)

    # PATCH: disable attention-weight capture during training for speed;
    # restore the previous state afterwards. Enable it explicitly (see
    # model.set_attention_capture) before visualisation passes.
    if hasattr(model, 'set_attention_capture'):
        model.set_attention_capture(False)

    # ------------------------------------------------------------------
    # PATCH: optional resume from a checkpoint (Colab sessions die).
    # ------------------------------------------------------------------
    start_epoch = 0
    resume_path = getattr(config, 'resume_from', None)
    if resume_path and os.path.exists(resume_path):
        # weights_only=False: this is our own trusted checkpoint, and it
        # contains non-tensor state (config dict, scheduler state).
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None and ckpt.get('scheduler_state_dict'):
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception as e:
                warnings.warn(f"Could not restore scheduler state: {e}")
        if scaler is not None and ckpt.get('scaler_state_dict'):
            try:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception as e:
                warnings.warn(f"Could not restore AMP scaler state: {e}")
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        best_val_loss = float(ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf'))))
        print(
            f"Resumed from {resume_path} at epoch {start_epoch} "
            f"(best val loss so far: {best_val_loss:.6f})"
        )

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        # ------------------------------------------------------------------
        # FIX #3 -- Scheduled teacher forcing
        # ------------------------------------------------------------------
        if hasattr(model, 'set_teacher_forcing_ratio'):
            decay_epochs = config.teacher_forcing_decay_epochs or config.num_epochs
            tf_ratio = (
                config.teacher_forcing_start
                - (config.teacher_forcing_start - config.teacher_forcing_end)
                * min(1.0, epoch / max(1, decay_epochs))
            )
            model.set_teacher_forcing_ratio(tf_ratio)

        # ------------------------------------------------------------------
        # Training batches
        # ------------------------------------------------------------------
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = _process_batch(data, target, device)

            with _get_autocast_context(use_amp, device):
                output = _forward_pass(
                    model, data, target, config,
                    adjacency_matrix, device, uses_transformer_decoder,
                )
                loss = _compute_loss(output, target, criterion, config) / accumulation_steps

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if config.gradient_clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if config.gradient_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        val_loss, _ = evaluate_model(
            model, val_loader, criterion, config,
            data_scaler=data_scaler, device=device,
            adjacency_matrix=adjacency_matrix,
        )
        val_losses.append(val_loss)

        # ------------------------------------------------------------------
        # Scheduler step
        # ------------------------------------------------------------------
        if scheduler is not None:
            if config.scheduler_type == 'cosine_warmup':
                scheduler.step()
            elif config.scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # ------------------------------------------------------------------
        # Early stopping & checkpoint  (FIX #7)
        # ------------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(model_dir, config.checkpoint_filename)
            # PATCH: store the config as a plain dict (the pickled dataclass
            # made checkpoints unloadable under torch>=2.6 weights_only
            # defaults and tied them to the package import path), and include
            # scheduler/scaler state so runs can be resumed exactly.
            try:
                config_payload = asdict(config)
            except Exception:
                config_payload = dict(getattr(config, '__dict__', {}))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'val_loss': val_loss,
                'train_loss': avg_train_loss,
                'best_val_loss': best_val_loss,
                'config': config_payload,
            }, checkpoint_path)
            print(f"  Saved best model checkpoint to {checkpoint_path}")

            # PATCH: mirror the checkpoint to Drive so a Colab disconnect
            # loses nothing (config.drive_backup_dir).
            backup_dir = getattr(config, 'drive_backup_dir', None)
            if backup_dir:
                try:
                    os.makedirs(backup_dir, exist_ok=True)
                    shutil.copy2(
                        checkpoint_path,
                        os.path.join(backup_dir, config.checkpoint_filename),
                    )
                except Exception as e:
                    warnings.warn(f"Drive checkpoint backup failed: {e}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"({patience} epochs without improvement)"
                )
                break

    # Load best model
    checkpoint_path = os.path.join(model_dir, config.checkpoint_filename)
    if os.path.exists(checkpoint_path):
        # PATCH: weights_only=False — torch>=2.6 defaults to True and refuses
        # checkpoints with non-tensor payloads; this is our own trusted file.
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {int(checkpoint.get('epoch', 0)) + 1}")

    return model, train_losses, val_losses


# =============================================================================
# evaluate_model
# =============================================================================

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Any,
    config: TrainingConfig,
    data_scaler: Optional[Union[MinMaxScaler, StandardScaler, RobustScaler]] = None,
    device: str = 'cpu',
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> Tuple[float, Tuple[float, float, float, float]]:
    """Evaluate the model on a data loader.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    data_loader : DataLoader
        Evaluation data.
    criterion :
        Loss function.
    config : TrainingConfig
        Experiment configuration.
    data_scaler : sklearn scaler, optional
        Fitted scaler for inverse transformation.
    device : str
        Device string.
    adjacency_matrix : torch.Tensor, optional
        Adjacency matrix for GNN variants.

    Returns
    -------
    tuple of (avg_loss, (mae, rmse, r2, mape))
    """
    model.eval()
    total_loss = 0.0
    all_predictions: List[np.ndarray] = []
    all_actuals: List[np.ndarray] = []

    # PATCH: move adjacency once, not per batch.
    if adjacency_matrix is not None:
        adjacency_matrix = adjacency_matrix.to(device)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = _process_batch(data, target, device)

            with _get_autocast_context(
                config.use_mixed_precision and AMP_AVAILABLE and device != 'cpu',
                device,
            ):
                output = _inference_forward(model, data, config, adjacency_matrix, device)
                loss = _compute_loss(output, target, criterion, config)

            total_loss += loss.item()

            preds = output.detach().cpu().numpy()
            acts = target.detach().cpu().numpy()

            all_predictions.append(preds)
            all_actuals.append(acts)

    avg_loss = total_loss / max(1, len(data_loader))

    # Flatten and compute metrics
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    predictions_flat = predictions.reshape(-1, predictions.shape[-1])
    actuals_flat = actuals.reshape(-1, actuals.shape[-1])

    # Inverse transform if scaler available
    if data_scaler is not None:
        try:
            predictions_flat = data_scaler.inverse_transform(predictions_flat)
            actuals_flat = data_scaler.inverse_transform(actuals_flat)
        except Exception:
            pass

    mae = float(mean_absolute_error(actuals_flat, predictions_flat))
    rmse = float(np.sqrt(mean_squared_error(actuals_flat, predictions_flat)))
    r2 = float(r2_score(actuals_flat.flatten(), predictions_flat.flatten()))
    # PATCH: masked MAPE (excludes faulty zero readings).
    mape = masked_mape(actuals_flat, predictions_flat)

    return avg_loss, (mae, rmse, r2, mape)


# =============================================================================
# predict
# =============================================================================

def predict(
    model: nn.Module,
    data_loader: DataLoader,
    config: TrainingConfig,
    data_scaler: Optional[Union[MinMaxScaler, StandardScaler, RobustScaler]] = None,
    device: str = 'cpu',
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions for a complete data loader.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    data_loader : DataLoader
        Data to predict on.
    config : TrainingConfig
        Experiment configuration.
    data_scaler : sklearn scaler, optional
        Fitted scaler for inverse transformation.
    device : str
        Device string.
    adjacency_matrix : torch.Tensor, optional
        Adjacency matrix for GNN variants.

    Returns
    -------
    tuple of (predictions_inv, actuals_inv)
        Both arrays are inverse-transformed (if a scaler is provided).
    """
    model.eval()
    all_predictions: List[np.ndarray] = []
    all_actuals: List[np.ndarray] = []

    # PATCH: move adjacency once, not per batch.
    if adjacency_matrix is not None:
        adjacency_matrix = adjacency_matrix.to(device)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = _process_batch(data, target, device)

            with _get_autocast_context(
                config.use_mixed_precision and AMP_AVAILABLE and device != 'cpu',
                device,
            ):
                output = _inference_forward(model, data, config, adjacency_matrix, device)

            preds = output.detach().cpu().numpy()
            acts = target.detach().cpu().numpy()

            all_predictions.append(preds)
            all_actuals.append(acts)

    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    predictions_flat = predictions.reshape(-1, predictions.shape[-1])
    actuals_flat = actuals.reshape(-1, actuals.shape[-1])

    # Inverse transform
    if data_scaler is not None:
        try:
            predictions_inv = data_scaler.inverse_transform(predictions_flat)
            actuals_inv = data_scaler.inverse_transform(actuals_flat)
        except Exception:
            predictions_inv = predictions_flat
            actuals_inv = actuals_flat
    else:
        predictions_inv = predictions_flat
        actuals_inv = actuals_flat

    return predictions_inv, actuals_inv


# =============================================================================
# Baseline helpers
# =============================================================================

def evaluate_baseline_model(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute standard regression metrics for baseline comparison.

    Parameters
    ----------
    predictions, actuals : np.ndarray
        Predicted and ground-truth values.

    Returns
    -------
    tuple of (mae, rmse, r2, mape)
    """
    predictions = np.asarray(predictions).flatten()
    actuals = np.asarray(actuals).flatten()

    mae = float(mean_absolute_error(actuals, predictions))
    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    r2 = float(r2_score(actuals, predictions))
    # PATCH: masked MAPE (excludes faulty zero readings).
    mape = masked_mape(actuals, predictions)

    return mae, rmse, r2, mape


def train_baseline_models(
    train_data: Any = None,
    test_data: Any = None,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """Train baseline models for comparison (stub).

    Returns
    -------
    dict
        Empty dictionary; baseline training is not implemented here.
    """
    return {}


# =============================================================================
# Discriminative parameter groups  (FIX #16)
# =============================================================================

def create_discriminative_param_groups(
    model: nn.Module,
    base_lr: float,
    decay_factor: float = 0.1,
) -> List[Dict[str, Any]]:
    """Create parameter groups with decreasing LR for lower encoder layers.

    Lower (earlier) encoder layers receive a smaller learning rate, allowing
    the model to preserve learned representations while fine-tuning upper
    layers more aggressively.

    Parameters
    ----------
    model : nn.Module
        Model whose ``encoder`` attribute (if present) is an iterable of layers.
    base_lr : float
        Learning rate for the topmost encoder layer and all non-encoder
        parameters.
    decay_factor : float
        Multiplicative factor applied per layer from the top.

    Returns
    -------
    list of dict
        Parameter groups suitable for passing to an optimiser constructor.
    """
    param_groups: List[Dict[str, Any]] = []

    if hasattr(model, 'encoder'):
        num_layers = len(model.encoder)
        for i, layer in enumerate(model.encoder):
            lr = base_lr * (decay_factor ** (num_layers - 1 - i))
            param_groups.append({'params': list(layer.parameters()), 'lr': lr})

    # Collect IDs of encoder parameters to avoid double-adding
    encoder_params: Set[int] = set()
    if hasattr(model, 'encoder'):
        for layer in model.encoder:
            encoder_params.update(id(p) for p in layer.parameters())

    other_params = [
        p for p in model.parameters()
        if id(p) not in encoder_params and p.requires_grad
    ]
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr})

    return param_groups


# =============================================================================
# Factory helpers: optimizer, scheduler, criterion
# =============================================================================

def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> optim.Optimizer:
    """Instantiate an optimiser based on *config.optimizer_type*.

    Supported types: ``'adam'``, ``'adamw'``.
    Falls back to AdamW with a warning for unrecognised values.
    """
    if config.optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=config.learning_rate)

    warnings.warn(
        f"Invalid optimizer: {config.optimizer_type}, using AdamW"
    )
    return optim.AdamW(model.parameters(), lr=config.learning_rate)


def create_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
) -> Optional[Any]:
    """Instantiate a learning-rate scheduler based on *config.scheduler_type*.

    Supported types: ``'plateau'``, ``'cosine'``, ``'step'``, ``'cosine_warmup'``.
    Returns ``None`` for unrecognised values.
    """
    if config.scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
        )
    elif config.scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
        )
    elif config.scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_scheduler_step_size,
            gamma=config.step_scheduler_gamma,
        )
    elif config.scheduler_type == 'cosine_warmup':
        if config.warmup_epochs <= 0:
            config.warmup_epochs = max(1, config.num_epochs // 10)
        return CosineWarmupLR(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.num_epochs,
            base_lr=config.learning_rate,
        )
    return None


def create_criterion(config: TrainingConfig) -> Any:
    """Instantiate a loss criterion based on *config.loss_function*.

    Supported values: ``'mse'``, ``'mae'``, ``'huber'``, ``'hybrid'``.
    Falls back to MSELoss with a warning for unrecognised values.
    """
    if config.loss_function == 'mse':
        return nn.MSELoss()
    elif config.loss_function == 'mae':
        return nn.L1Loss()
    elif config.loss_function == 'huber':
        return nn.SmoothL1Loss()
    elif config.loss_function == 'hybrid':
        return hybrid_loss

    warnings.warn(
        f"Invalid loss: {config.loss_function}, using MSELoss"
    )
    return nn.MSELoss()
