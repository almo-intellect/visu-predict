"""End-to-end training pipeline: orchestrates data loading, model, training, evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from visu_predict.config import TrainingConfig, setup_directories
from visu_predict.data import TrafficDataset, load_traffic_dataframe, prepare_data
from visu_predict.features.spatial import SpatialIntegration
from visu_predict.models.lr import CosineWarmupLR
from visu_predict.models.transformer import TrafficTransformer
from visu_predict.paths import find_adjacency_matrix, find_coordinates
from visu_predict.training.losses import make_criterion
from visu_predict.training.train import predict as run_predict
from visu_predict.training.train import train
from visu_predict.training.transfer import freeze_for_fine_tuning, load_pretrained, transfer_config_overrides
from visu_predict.utils.gpu import best_available_device, log_gpu_memory
from visu_predict.utils.logging_setup import setup_logging
from visu_predict.utils.seed import set_seed
from visu_predict.viz import plot_predictions_vs_actual, plot_training_history

logger = logging.getLogger(__name__)


def _build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if config.optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    raise ValueError(f"Unknown optimizer_type: {config.optimizer_type!r}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    kind = config.scheduler_type
    if kind in (None, "none", ""):
        return None
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience,
        )
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_scheduler_step_size, gamma=config.step_scheduler_gamma,
        )
    if kind == "cosine_warmup":
        return CosineWarmupLR(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.num_epochs,
            base_lr=config.learning_rate,
        )
    raise ValueError(f"Unknown scheduler_type: {kind!r}")


def _build_model(
    sample_features: dict[str, torch.Tensor],
    feature_dims: dict[str, int],
    num_features: int,
    config: TrainingConfig,
) -> TrafficTransformer:
    return TrafficTransformer(
        input_dim=sample_features["concatenated"].shape[-1],
        num_features=num_features,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        ff_dim_multiplier=config.ff_dim_multiplier,
        activation=config.activation,
        decoder_type=config.decoder_type,
        pred_len=config.pred_length,
        feature_dims=feature_dims,
        use_gnn_pre_transformer=config.use_gnn_pre_transformer,
        spatial_feature_dim=config.spatial_feature_dim if config.use_spatial_bias else 0,
        gnn_type=config.gnn_type,
        gnn_layers=config.gnn_layers,
        gnn_residual=config.gnn_residual,
        gat_heads=config.gat_heads,
        gat_concat=config.gat_concat,
        use_spatial_bias=config.use_spatial_bias,
        spatial_bias_type=config.spatial_bias_type,
        max_seq_length=config.max_seq_length,
        num_decoder_layers=config.num_decoder_layers,
        teacher_forcing_ratio=config.teacher_forcing_ratio,
        model_pipeline=config.model_pipeline,
        steps_per_day=config.steps_per_day or 288,
        d_input=config.d_input,
        d_tod=config.d_tod,
        d_dow=config.d_dow,
        d_adaptive=config.d_adaptive,
        d_node=config.d_node,
        seq_length=config.seq_length,
        interleave_order=config.interleave_order,
        num_st_layers=config.num_st_layers,
        use_adaptive_adjacency=config.use_adaptive_adjacency,
        adaptive_adj_dim=config.adaptive_adj_dim,
        adaptive_adj_inject_into=config.adaptive_adj_inject_into,
        use_temporal_patching=config.use_temporal_patching,
        patch_length=config.patch_length,
        patch_stride=config.patch_stride,
    )


def run_training(config: TrainingConfig, data_path: str | Path) -> dict[str, object]:
    """Full pipeline: load → train → evaluate → plot. Returns a result dict."""
    config = setup_directories(config)
    setup_logging(log_file=Path(config.output_dir) / "training.log")
    set_seed(config.seed)

    device = best_available_device()
    logger.info("Using device: %s", device)
    log_gpu_memory()

    df = load_traffic_dataframe(data_path)
    logger.info("Loaded data with shape %s from %s", df.shape, data_path)

    data, timestamps, scaler, num_features = prepare_data(df, config)

    val_size = max(1, int(len(data) * 0.2))
    train_data, val_data = data[:-val_size], data[-val_size:]
    train_ts, val_ts = timestamps[:-val_size], timestamps[-val_size:]

    train_ds = TrafficDataset(train_data, train_ts, config)
    val_ds = TrafficDataset(val_data, val_ts, config)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )

    adjacency_tensor = None
    if config.use_spatial_features or config.use_gnn_pre_transformer:
        assert config.input_dir is not None
        adj_path = find_adjacency_matrix(config.dataset_name, config.input_dir)
        coord_path = find_coordinates(config.dataset_name, config.input_dir)
        spatial = SpatialIntegration(
            adjacency_path=adj_path,
            coordinates_path=coord_path,
            num_sensors=num_features,
            spatial_dim=config.spatial_feature_dim,
            embedding_dim=config.embedding_dim,
            device=device,
        )
        adjacency_tensor = spatial.normalized_adjacency_tensor

    sample_features, _ = train_ds[0]
    sample_features = {k: v.unsqueeze(0) for k, v in sample_features.items()}
    feature_dims = {k: v for k, v in train_ds.feature_dims.items()}

    model = _build_model(sample_features, feature_dims, num_features, config)

    if config.enable_transfer_learning and config.source_model_path:
        model = load_pretrained(model, config.source_model_path)
        freeze_for_fine_tuning(model, freeze_encoder=config.freeze_encoder, num_layers=config.freeze_layers)
        config = transfer_config_overrides(config)

    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    criterion = make_criterion(config.loss_function)

    model, train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=config,
        data_scaler=scaler,
        device=device,
        adjacency_matrix=adjacency_tensor,
    )

    preds, actuals = run_predict(model, val_loader, scaler, config, device, adjacency_matrix=adjacency_tensor)

    if config.generate_plots:
        plot_training_history(train_losses, val_losses, Path(config.results_dir) / "training_history.png")
        plot_predictions_vs_actual(preds, actuals, Path(config.results_dir) / "predictions.png")

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "predictions": preds,
        "actuals": actuals,
        "scaler": scaler,
    }


def run_pretraining(config: TrainingConfig, data_path: str | Path) -> dict[str, object]:
    """Masked-reconstruction pretraining (STAE pipeline only).

    Builds the same STAE model that ``run_training`` would build, but trains
    it to reconstruct masked traffic cells instead of forecasting. The
    encoder weights are saved to ``<output>/pretrained/encoder.pth`` and
    can be loaded later via the ``pretrained_encoder_path`` config field.
    """
    from visu_predict.training.pretrain import MaskedSTDataset, pretrain, save_pretrained_encoder

    if config.model_pipeline != "stae":
        raise ValueError(
            "Pretraining requires model_pipeline='stae'; got "
            f"{config.model_pipeline!r}"
        )

    config = setup_directories(config)
    setup_logging(log_file=Path(config.output_dir) / "pretrain.log")
    set_seed(config.seed)

    device = best_available_device()
    logger.info("Pretraining on device: %s", device)
    log_gpu_memory()

    df = load_traffic_dataframe(data_path)
    logger.info("Loaded data with shape %s from %s", df.shape, data_path)

    data, timestamps, _scaler, num_features = prepare_data(df, config)
    val_size = max(1, int(len(data) * 0.2))
    train_data, val_data = data[:-val_size], data[-val_size:]
    train_ts, val_ts = timestamps[:-val_size], timestamps[-val_size:]

    train_ds = MaskedSTDataset(train_data, train_ts, config)
    val_ds = MaskedSTDataset(val_data, val_ts, config)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )

    sample_features, _ = train_ds[0]
    sample_features = {k: v.unsqueeze(0) for k, v in sample_features.items()}
    feature_dims = {k: v for k, v in train_ds.feature_dims.items()}
    model = _build_model(sample_features, feature_dims, num_features, config)

    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    model, train_losses, val_losses = pretrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    ckpt_path = Path(config.output_dir) / "pretrained" / "encoder.pth"
    save_pretrained_encoder(model, ckpt_path)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "checkpoint": str(ckpt_path),
    }
