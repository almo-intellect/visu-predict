"""Training configuration with YAML loading and directory setup."""

from __future__ import annotations

import logging
import multiprocessing
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DATASET_SENSOR_COUNTS: dict[str, int] = {
    "METR-LA": 207,
    "PEMS-BAY": 325,
    "PEMS-03": 357,
    "PEMS-04": 306,
    "PEMS-07": 882,
    "PEMS-08": 169,
}

VALID_WEATHER_FEATURES: tuple[str, ...] = (
    "all_features", "temperature", "weather_condition_code", "visibility",
    "wind_speed", "wind_direction_code", "wind", "humidity", "dew_point",
    "cloud_cover_code",
)
VALID_SCALERS: tuple[str, ...] = ("minmax", "standard", "robust")
VALID_LOSSES: tuple[str, ...] = ("mae", "mse", "huber", "hybrid")
VALID_DECODERS: tuple[str, ...] = ("linear", "mlp", "transformer")
VALID_MISSING_STRATEGIES: tuple[str, ...] = (
    "ffill_bfill", "zero", "mean", "median", "interpolate",
)
VALID_MODEL_PIPELINES: tuple[str, ...] = ("legacy", "stae")


@dataclass
class TrainingConfig:
    """Configuration for traffic transformer training.

    Required:
        base_output_dir: Root directory where run outputs are written.
    """

    base_output_dir: str

    dataset_name: str = "METR-LA"
    data_file: str | None = None

    batch_size: int = 16
    seq_length: int = 12
    pred_length: int = 12
    num_epochs: int = 100
    patience: int = 30
    learning_rate: float = 1e-4

    hidden_dim: int = 336
    num_layers: int = 3
    num_heads: int = 16
    dropout: float = 0.05
    ff_dim_multiplier: int = 4
    activation: str = "gelu"

    data_scaler_type: str = "minmax"
    optimizer_type: str = "adamw"
    loss_function: str = "mae"

    use_time_features: bool = True
    use_holiday_feature: bool = False
    holiday_country_code: str = "US"
    use_weather_feature: bool = False
    weather_feature_type: str = "all_features"
    weather_data_file: str | None = None

    gradient_clip: float | None = 1.0
    scheduler_type: str | None = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    step_scheduler_step_size: int = 10
    step_scheduler_gamma: float = 0.1

    use_lagged_features: bool = False
    num_lags: int = 1

    decoder_type: str = "linear"
    num_decoder_layers: int = 3
    dim_feedforward: int = 336
    teacher_forcing_ratio: float = 0.2

    use_spatial_features: bool = False
    spatial_feature_dim: int = 336
    use_gnn_pre_transformer: bool = False
    gnn_type: str = "gcn"
    gat_heads: int = 16
    gat_concat: bool = True
    gnn_residual: bool = False
    gnn_layers: int = 3

    coordinates_file: str | None = None
    num_sensors: int = 207
    embedding_dim: int = 207
    use_spatial_bias: bool = False
    spatial_bias_type: str = "additive"

    max_seq_length: int = 100_000

    save_predictions: bool = True
    generate_plots: bool = True

    missing_value_strategy: str = "mean"
    time_format: str = "%Y-%m-%d %H:%M:%S"

    use_quantile_regression: bool = False
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    warmup_epochs: int = 30
    use_mixed_precision: bool = True
    accumulation_steps: int = 8
    num_workers: int = 2
    pin_memory: bool = True

    enable_transfer_learning: bool = False
    source_model_path: str | None = None
    target_dataset_name: str | None = None
    target_data_path: str | None = None
    freeze_encoder: bool = True
    freeze_layers: int = 1
    adapter_dim: int = 64
    transfer_learning_rate: float = 5e-5

    seed: int = 42

    # --- STAE / SOTA upgrade (Tier 1+) ---
    # Switching ``model_pipeline`` to "stae" enables the STAEformer-style path:
    # a learnable adaptive embedding indexed by (time-of-day, sensor), plus
    # discrete time-of-day / day-of-week lookup embeddings. Legacy is the
    # default and reproduces the pre-Tier-1 model exactly.
    model_pipeline: str = "legacy"
    use_discrete_time_embeddings: bool = False
    steps_per_day: int | None = None
    d_input: int = 24
    d_tod: int = 24
    d_dow: int = 24
    d_adaptive: int = 80
    d_node: int = 0
    interleave_order: str = "TS"
    num_st_layers: int | None = None
    use_adaptive_adjacency: bool = False
    adaptive_adj_dim: int = 10
    adaptive_adj_inject_into: str = "spatial_attn"
    use_temporal_patching: bool = False
    patch_length: int = 4
    patch_stride: int | None = None

    input_dir: str | None = None
    output_dir: str | None = None
    model_dir: str | None = None
    results_dir: str | None = None

    def __post_init__(self) -> None:
        self._adjust_hidden_dim()
        self._validate_choices()

        max_workers = multiprocessing.cpu_count()
        if self.num_workers > max_workers:
            logger.warning(
                "Reducing num_workers from %d to %d (CPU count)",
                self.num_workers, max_workers,
            )
            self.num_workers = max_workers

        if self.num_sensors == 207 and self.dataset_name in DATASET_SENSOR_COUNTS:
            self.num_sensors = DATASET_SENSOR_COUNTS[self.dataset_name]

    def _adjust_hidden_dim(self) -> None:
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_dim % self.num_heads != 0:
            original = self.hidden_dim
            self.hidden_dim = max(self.num_heads, (self.hidden_dim // self.num_heads) * self.num_heads)
            warnings.warn(
                f"Adjusted hidden_dim from {original} to {self.hidden_dim} "
                f"to be divisible by num_heads={self.num_heads}",
                stacklevel=2,
            )
        if self.hidden_dim % 2 != 0:
            self.hidden_dim += 1
            warnings.warn(
                f"Adjusted hidden_dim to {self.hidden_dim} to be even for positional encoding",
                stacklevel=2,
            )

    def _validate_choices(self) -> None:
        if self.weather_feature_type not in VALID_WEATHER_FEATURES:
            raise ValueError(
                f"weather_feature_type must be one of {VALID_WEATHER_FEATURES}, "
                f"got {self.weather_feature_type!r}"
            )
        if self.data_scaler_type not in VALID_SCALERS:
            raise ValueError(f"data_scaler_type must be one of {VALID_SCALERS}, got {self.data_scaler_type!r}")
        if self.loss_function not in VALID_LOSSES:
            raise ValueError(f"loss_function must be one of {VALID_LOSSES}, got {self.loss_function!r}")
        if self.decoder_type not in VALID_DECODERS:
            raise ValueError(f"decoder_type must be one of {VALID_DECODERS}, got {self.decoder_type!r}")
        if self.missing_value_strategy not in VALID_MISSING_STRATEGIES:
            raise ValueError(
                f"missing_value_strategy must be one of {VALID_MISSING_STRATEGIES}, "
                f"got {self.missing_value_strategy!r}"
            )
        if self.spatial_bias_type not in ("additive", "multiplicative"):
            raise ValueError(f"spatial_bias_type must be 'additive' or 'multiplicative', got {self.spatial_bias_type!r}")
        if self.adaptive_adj_inject_into not in ("spatial_attn", "gnn", "both"):
            raise ValueError(
                "adaptive_adj_inject_into must be 'spatial_attn', 'gnn', or 'both', "
                f"got {self.adaptive_adj_inject_into!r}"
            )
        if self.model_pipeline not in VALID_MODEL_PIPELINES:
            raise ValueError(
                f"model_pipeline must be one of {VALID_MODEL_PIPELINES}, got {self.model_pipeline!r}"
            )
        if self.model_pipeline == "stae":
            if not self.use_discrete_time_embeddings:
                raise ValueError(
                    "model_pipeline='stae' requires use_discrete_time_embeddings=True"
                )
            stae_dim_sum = self.d_input + self.d_tod + self.d_dow + self.d_adaptive + self.d_node
            if stae_dim_sum != self.hidden_dim:
                raise ValueError(
                    f"STAE pipeline requires d_input + d_tod + d_dow + d_adaptive + d_node "
                    f"({stae_dim_sum}) to equal hidden_dim ({self.hidden_dim})"
                )
            if self.interleave_order not in ("TS", "ST"):
                raise ValueError(
                    f"interleave_order must be 'TS' or 'ST', got {self.interleave_order!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: str | Path) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return TrainingConfig(**data)


def setup_directories(config: TrainingConfig, run_name: str | None = None) -> TrainingConfig:
    """Create timestamped output, model, and results directories.

    Updates the config in place with the resolved paths and returns it.
    """
    timestamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(config.base_output_dir).expanduser().resolve()
    run_dir = base / f"{config.dataset_name}_{timestamp}"

    input_dir = (base / "inputs") if config.input_dir is None else Path(config.input_dir)
    model_dir = run_dir / "models"
    results_dir = run_dir / "results"

    for d in (input_dir, run_dir, model_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    config.input_dir = str(input_dir)
    config.output_dir = str(run_dir)
    config.model_dir = str(model_dir)
    config.results_dir = str(results_dir)
    return config
