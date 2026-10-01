"""
Configuration module for the traffic prediction transformer.

Provides ``TrainingConfig`` (the main experiment configuration dataclass),
``TransferLearningConfig``, directory-setup helpers, and YAML config loading.
"""

import multiprocessing
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

from .utils import (
    AMP_AVAILABLE,
    DEVICE_TYPE_SUPPORTED,
    TORCH_GEOMETRIC_AVAILABLE,
    get_maputo_timestamp,
)


# =============================================================================
# TrainingConfig
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration class for training parameters with transfer learning support."""

    # -- Centralised Google Drive paths ------------------------------------
    DRIVE_BASE: str = '/content/drive/Shareddrives/Almo-2002-R&D/1 - RD-Traffic-Prediction'
    DRIVE_PROJECT: str = DRIVE_BASE + '/Transformer_Versions/COLAB_NOBASELINE_V14-6'

    # -- Reproducibility ----------------------------------------------------
    seed: int = 42

    # -- Core training parameters ------------------------------------------
    base_output_dir: str = ''
    dataset_name: str = 'PEMS-BAY'
    batch_size: int = 16
    seq_length: int = 12
    pred_length: int = 12
    num_epochs: int = 1000
    patience: int = 30
    learning_rate: float = 0.0001
    hidden_dim: int = 336
    num_layers: int = 3
    num_heads: int = 16
    dropout: float = 0.05
    ff_dim_multiplier: int = 4
    activation: str = 'gelu'
    data_scaler_type: str = 'minmax'
    optimizer_type: str = 'adamw'
    loss_function: str = 'mae'
    use_time_features: bool = True
    use_holiday_feature: bool = False
    holiday_country_code: str = 'US'
    use_weather_feature: bool = False
    weather_feature_type: str = 'all_features'
    weather_data_file: Optional[str] = None
    gradient_clip: Optional[float] = 1.0
    scheduler_type: Optional[str] = 'cosine_warmup'
    scheduler_patience: int = 30
    scheduler_factor: float = 0.5
    step_scheduler_step_size: int = 10
    step_scheduler_gamma: float = 0.1
    use_lagged_features: bool = False
    num_lags: int = 1
    # PATCH: when True, TrafficDataset yields {'traffic':..., 'time':..., ...}
    # dicts (PyTorch's default collate batches dicts natively) and the model
    # routes them through the FeatureAttention module — the paper's
    # feature-wise attention mechanism. Previously there was no way to
    # activate this path: the dataset always emitted concatenated tensors.
    use_feature_attention: bool = False

    # -- Decoder parameters ------------------------------------------------
    decoder_type: str = 'linear'
    num_decoder_layers: int = 3
    dim_feedforward: int = 336
    teacher_forcing_ratio: float = 0.20

    # Scheduled teacher forcing (NEW)
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.0
    teacher_forcing_decay_epochs: Optional[int] = None

    # -- GNN parameters ----------------------------------------------------
    use_spatial_features: bool = False
    spatial_feature_dim: int = 336
    use_gnn_pre_transformer: bool = False
    gnn_type: str = 'gcn'
    gat_heads: int = 16
    gat_concat: bool = True
    gnn_residual: bool = False
    gnn_layers: int = 3

    # -- Spatial features parameters ---------------------------------------
    coordinates_file: Optional[str] = None  # PATCH: was False, breaking the `is None` auto-resolution
    num_sensors: int = 325
    embedding_dim: int = 325
    use_spatial_bias: bool = False
    spatial_bias_type: str = 'additive'

    # -- Model architecture ------------------------------------------------
    max_seq_length: int = 100000
    feature_dims: Optional[Dict[str, int]] = None  # FIX: was incorrectly = 336

    # -- Attention visualisation -------------------------------------------
    attention_visualization: bool = True
    attention_head_analysis: bool = True
    visualize_layer_progression: bool = True

    # -- Output and visualisation ------------------------------------------
    save_predictions: bool = True
    num_sensors_to_plot: int = 25
    generate_report: bool = True

    # -- Data processing ---------------------------------------------------
    missing_value_strategy: str = 'mean'
    time_format: str = '%Y-%m-%d %H:%M:%S'

    # -- Training optimisation ---------------------------------------------
    use_quantile_regression: bool = False
    quantiles: Optional[List[float]] = None
    optuna_trials: Optional[int] = 0
    warmup_epochs: int = 30
    use_mixed_precision: bool = True
    accumulation_steps: int = 1  # PATCH: was 8; set >1 only when you deliberately want accumulation
    num_workers: int = 2
    pin_memory: bool = True
    find_optimal_batch_size: bool = True
    monitor_gpu_usage: bool = True

    # -- Transfer learning support -----------------------------------------
    enable_transfer_learning: bool = False
    run_only_transfer_learning: bool = False
    source_model_path: Optional[str] = None
    target_dataset_name: Optional[str] = 'METR-MPT'
    target_data_path: Optional[str] = None
    freeze_encoder: bool = True
    freeze_layers: int = 1
    adapter_dim: int = 64
    transfer_learning_rate: float = 5e-5

    # -- Checkpoint naming (NEW) -------------------------------------------
    checkpoint_filename: str = 'best_model.pth'
    # PATCH: Colab resilience. If set, every improved checkpoint is also
    # copied to this (Drive) directory so a session disconnect loses nothing.
    drive_backup_dir: Optional[str] = None
    # PATCH: path to a checkpoint to resume training from (model, optimizer,
    # scheduler, scaler, epoch, and best-val-loss state are restored).
    resume_from: Optional[str] = None

    # -- Discriminative fine-tuning (NEW) ----------------------------------
    discriminative_lr_factor: float = 0.1

    # -- Directories (set after initialisation) ----------------------------
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    model_dir: Optional[str] = None
    results_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Quantiles default
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]

        # Initialise feature_dims if None
        if self.feature_dims is None:
            self.feature_dims = {}

        # Dataset-specific defaults
        self._set_dataset_specific_defaults()

        # Ensure hidden_dim is divisible by num_heads
        if self.num_heads > 0:
            if self.hidden_dim % self.num_heads != 0:
                original_hidden_dim = self.hidden_dim
                self.hidden_dim = (self.hidden_dim // self.num_heads) * self.num_heads
                if self.hidden_dim == 0:
                    self.hidden_dim = self.num_heads
                warnings.warn(
                    f"Adjusted hidden_dim from {original_hidden_dim} to "
                    f"{self.hidden_dim} to ensure divisibility by "
                    f"num_heads={self.num_heads}"
                )

            # Ensure hidden_dim is even for positional encoding
            if self.hidden_dim % 2 != 0:
                original_hidden_dim = self.hidden_dim
                self.hidden_dim += 1
                warnings.warn(
                    f"Adjusted hidden_dim from {original_hidden_dim} to "
                    f"{self.hidden_dim} to ensure it's even for positional encoding"
                )

        # Mixed precision availability
        if self.use_mixed_precision and not AMP_AVAILABLE:
            self.use_mixed_precision = False
            warnings.warn("Mixed precision requested but not available, disabled")

        # GNN availability
        # PATCH: the GNN pre-encoder is now a dense implementation (see
        # model.DenseGCNPreEncoder) and does NOT require torch_geometric.
        # Only the optional GAT variant needs it; fall back to dense GCN.
        if (
            self.use_gnn_pre_transformer
            and self.gnn_type == 'gat'
            and not TORCH_GEOMETRIC_AVAILABLE
        ):
            warnings.warn(
                "gnn_type='gat' requires torch_geometric, which is not "
                "installed. Falling back to the dense GCN pre-encoder."
            )
            self.gnn_type = 'gcn'

        # Adjust workers based on system capabilities
        max_workers = multiprocessing.cpu_count()
        if self.num_workers > max_workers:
            warnings.warn(
                f"Reducing num_workers from {self.num_workers} to {max_workers} "
                f"based on system capabilities"
            )
            self.num_workers = max_workers

        # Validate weather feature type
        valid_weather_types = [
            'all_features', 'temperature', 'weather_condition_code',
            'visibility', 'wind_speed', 'wind_direction_code',
            'wind', 'humidity', 'dew_point', 'cloud_cover_code',
        ]
        if self.weather_feature_type not in valid_weather_types:
            warnings.warn(
                f"Invalid weather_feature_type: {self.weather_feature_type}, "
                f"using 'all_features'"
            )
            self.weather_feature_type = 'all_features'

        # Validate spatial bias type
        valid_spatial_bias_types = ['additive', 'multiplicative']
        if self.spatial_bias_type not in valid_spatial_bias_types:
            warnings.warn(
                f"Invalid spatial_bias_type: {self.spatial_bias_type}, "
                f"using 'additive'"
            )
            self.spatial_bias_type = 'additive'

        # Validate missing value strategy
        valid_missing_strategies = [
            'ffill_bfill', 'zero', 'mean', 'median', 'interpolate',
            'mean_replace_zeros',  # PATCH: was implemented but rejected by validation
        ]
        if self.missing_value_strategy not in valid_missing_strategies:
            warnings.warn(
                f"Invalid missing_value_strategy: {self.missing_value_strategy}, "
                f"using 'ffill_bfill'"
            )
            self.missing_value_strategy = 'ffill_bfill'

        # Validate transfer learning settings if enabled
        if self.enable_transfer_learning:
            if self.target_dataset_name is None:
                warnings.warn(
                    "Transfer learning enabled but target_dataset_name not set. "
                    "Using 'mozambique'"
                )
                self.target_dataset_name = 'mozambique'

            if self.target_data_path is None:
                self.target_data_path = f"{self.target_dataset_name}_traffic.csv"
                warnings.warn(
                    f"Transfer learning enabled but target_data_path not set. "
                    f"Using '{self.target_data_path}'"
                )

    def _set_dataset_specific_defaults(self) -> None:
        """Set default values based on the selected dataset."""
        dataset_sensor_map = {
            'PEMS-BAY': 325,
            'PEMS-03': 357,
            'PEMS-04': 306,
            'PEMS-07': 882,
            'PEMS-08': 169,
        }
        if not hasattr(self, 'num_features'):
            self.num_features = dataset_sensor_map.get(self.dataset_name, 207)


# =============================================================================
# Dimension compatibility helper
# =============================================================================

def ensure_compatible_dimensions(
    model_params: Dict[str, Any],
    config: TrainingConfig,
) -> Tuple[Dict[str, Any], TrainingConfig]:
    """Adjust *hidden_dim* so that it is divisible by *num_heads* and is even.

    Returns the updated ``(model_params, config)`` pair.
    """
    if model_params['hidden_dim'] % model_params['num_heads'] != 0:
        adjusted = (model_params['hidden_dim'] // model_params['num_heads']) * model_params['num_heads']
        if adjusted == 0:
            adjusted = model_params['num_heads']
        print(
            f"Warning: Adjusting hidden_dim from {model_params['hidden_dim']} to "
            f"{adjusted} to ensure divisibility by num_heads={model_params['num_heads']}"
        )
        model_params['hidden_dim'] = adjusted
        config.hidden_dim = adjusted

    if model_params['hidden_dim'] % 2 != 0:
        adjusted = model_params['hidden_dim'] + 1
        print(
            f"Warning: Adjusting hidden_dim from {model_params['hidden_dim']} to "
            f"{adjusted} to ensure it's even for positional encoding"
        )
        model_params['hidden_dim'] = adjusted
        config.hidden_dim = adjusted

    return model_params, config


# =============================================================================
# YAML config loading
# =============================================================================

def load_config(config_path: str = 'config.yaml') -> TrainingConfig:
    """Load configuration from a YAML file and return a ``TrainingConfig``.

    Falls back to default values if the file cannot be read.
    """
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return TrainingConfig(**config_dict)
    except (FileNotFoundError, yaml.YAMLError) as e:
        warnings.warn(
            f"Error loading config from {config_path}: {e}. "
            f"Using default configuration."
        )
        return TrainingConfig(
            base_output_dir=TrainingConfig.DRIVE_PROJECT,
        )


# =============================================================================
# Directory setup
# =============================================================================

def setup_directories(config: TrainingConfig) -> Tuple[str, str, str, str]:
    """Create timestamped output directories and update *config* in place.

    Returns ``(input_dir, output_dir, model_dir, results_dir)``.
    """
    timestamp = get_maputo_timestamp()

    output_dir = os.path.join(config.base_output_dir, f"Transformers_Output_{timestamp}")

    # Respect user-provided input_dir; fall back to DRIVE_PROJECT/Transformers_Input
    if config.input_dir:
        input_dir = config.input_dir
    else:
        input_dir = os.path.join(config.DRIVE_PROJECT, 'Transformers_Input')

    model_dir = os.path.join(output_dir, f"Models_{timestamp}")
    results_dir = os.path.join(output_dir, f"Results_{timestamp}")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    config.input_dir = input_dir
    config.output_dir = output_dir
    config.model_dir = model_dir
    config.results_dir = results_dir

    # -- Coordinates file --------------------------------------------------
    if config.coordinates_file is None:
        coords_map = {
            'PEMS-BAY': 'graph_sensor_locations_pems_bay.csv',
            'PEMS-03': 'graph_sensor_locations_pems_03.csv',
            'PEMS03_new': 'graph_sensor_locations_pems_03.csv',
            'PEMS-04': 'graph_sensor_locations_pems_04.csv',
            'PEMS-07': 'graph_sensor_locations_pems_07.csv',
            'PEMS-08': 'graph_sensor_locations_pems_08.csv',
            'METR-LA': 'graph_sensor_locations_metr_la.csv',
        }
        coords_filename = coords_map.get(config.dataset_name)
        if coords_filename is not None:
            config.coordinates_file = os.path.join(input_dir, coords_filename)
        else:
            print(f"Coordinates file not found for dataset: {config.dataset_name}")

    # -- Weather data file -------------------------------------------------
    weather_file_map = {
        'PEMS-BAY': 'clean_weather_data_pems_bay.csv',
        'PEMS-03': 'clean_weather_data_pems_03.csv',
        'PEMS03_new': 'clean_weather_data_pems_03.csv',
        'PEMS-04': 'clean_weather_data_pems_04.csv',
        'PEMS-07': 'clean_weather_data_pems_07.csv',
        'PEMS-08': 'clean_weather_data_pems_08.csv',
        'METR-LA': 'clean_weather_data_v3.csv',
    }
    weather_filename = weather_file_map.get(config.dataset_name)
    if weather_filename is not None:
        weather_src = config.weather_data_file or os.path.join(input_dir, weather_filename)
    else:
        # PATCH: previously this branch overwrote a user-provided
        # weather_data_file with None for datasets outside the map.
        weather_src = config.weather_data_file
        if weather_src is None:
            print(f"No default weather file mapping for dataset: {config.dataset_name}")
    config.weather_data_file = weather_src

    # -- Visualisation directory -------------------------------------------
    if (
        config.attention_visualization
        or config.attention_head_analysis
        or config.visualize_layer_progression
    ):
        vis_dir = os.path.join(results_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Created visualization directory at {vis_dir}")

    # -- Transfer learning target dataset ----------------------------------
    if config.enable_transfer_learning and config.target_data_path:
        target_src = config.target_data_path
        target_dest = os.path.join(input_dir, os.path.basename(config.target_data_path))

        if os.path.exists(target_src) and not os.path.exists(target_dest):
            try:
                import shutil
                shutil.copyfile(target_src, target_dest)
                print(f"Target dataset copied to {target_dest}")
                config.target_data_path = target_dest
            except Exception as e:
                warnings.warn(f"Could not copy target dataset file: {e}")
        elif os.path.exists(target_dest):
            config.target_data_path = target_dest

    # -- Reports directory -------------------------------------------------
    if config.generate_report:
        reports_dir = os.path.join(results_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        print(f"Created reports directory at {reports_dir}")

    # -- Predictions directory ---------------------------------------------
    if config.save_predictions:
        predictions_dir = os.path.join(results_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        print(f"Created predictions directory at {predictions_dir}")

    return input_dir, output_dir, model_dir, results_dir


# =============================================================================
# Adjacency matrix path helper
# =============================================================================

def get_adjacency_matrix_path(
    config: TrainingConfig,
    input_dir: Optional[str] = None,
) -> Optional[str]:
    """Locate the adjacency matrix ``.pkl`` file for the configured dataset.

    Returns the first existing path or ``None`` if nothing is found.

    PATCH: *input_dir* is now optional and defaults to ``config.input_dir``;
    several call sites (TrafficDataset, SpatialIntegration) passed only the
    config object and crashed with a TypeError.
    """
    if input_dir is None:
        input_dir = config.input_dir or '.'
    dataset_name = config.dataset_name

    adj_filenames: Dict[str, str] = {
        'PEMS-BAY': 'adj_PEMS-BAY.pkl',
        'PEMS-03': 'adj_PEMS-03.pkl',
        'PEMS-04': 'adj_PEMS-04.pkl',
        'PEMS-07': 'adj_PEMS-07.pkl',
        'PEMS-08': 'adj_PEMS-08.pkl',
        'PEMS03_new': None,
        'METR-LA': 'adj_METR-LA.pkl',
    }

    adj_filename = adj_filenames.get(dataset_name, f'adj_{dataset_name}.pkl')

    # Dataset-specific subdirectory names
    subdir_map: Dict[str, str] = {
        'PEMS-BAY': 'ADJACENCY_MATRIX_PEMS_BAY',
        'PEMS-03': 'ADJACENCY_MATRIX_PEMS_03',
        'PEMS-04': 'ADJACENCY_MATRIX_PEMS_04',
        'PEMS-07': 'ADJACENCY_MATRIX_PEMS_07',
        'PEMS-08': 'ADJACENCY_MATRIX_PEMS_08',
        'PEMS03_new': 'ADJACENCY_MATRIX_PEMS_03',
        'METR-LA': 'ADJACENCY_MATRIX_METR_LA',
    }

    potential_paths: List[str] = []

    if adj_filename is not None:
        subdir = subdir_map.get(dataset_name, '')
        potential_paths.extend([
            os.path.join(input_dir, adj_filename),
            os.path.join(input_dir, subdir, adj_filename) if subdir else '',
            os.path.join(config.DRIVE_BASE, subdir, adj_filename) if subdir else '',
            os.path.join(config.DRIVE_BASE, adj_filename),
        ])

    # Generic fallbacks
    fallback_filename = f'adj_{dataset_name}.pkl'
    potential_paths.extend([
        os.path.join(input_dir, fallback_filename),
        os.path.join('.', fallback_filename),
        os.path.join('./data', fallback_filename),
    ])

    # Remove empty strings
    potential_paths = [p for p in potential_paths if p]

    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found adjacency matrix at: {path}")
            return path

    print(f"Warning: No adjacency matrix found for dataset {dataset_name}")
    return None


# =============================================================================
# Transfer learning helpers
# =============================================================================

def apply_transfer_config(
    main_config: TrainingConfig,
    transfer_config: 'TransferLearningConfig',
) -> TrainingConfig:
    """Apply a ``TransferLearningConfig`` onto the main ``TrainingConfig``.

    Returns the updated *main_config*.
    """
    main_config.enable_transfer_learning = True

    main_config.source_model_path = transfer_config.pretrained_model_path
    main_config.target_dataset_name = transfer_config.target_dataset_name
    main_config.target_data_path = transfer_config.target_data_path
    main_config.freeze_encoder = transfer_config.freeze_encoder
    main_config.freeze_layers = transfer_config.freeze_layers

    if transfer_config.use_adapters:
        main_config.adapter_dim = transfer_config.adapter_dim
    else:
        main_config.adapter_dim = 0

    main_config.transfer_learning_rate = transfer_config.learning_rate

    main_config.num_epochs = min(main_config.num_epochs, transfer_config.num_epochs)
    main_config.patience = min(main_config.patience, transfer_config.patience)
    main_config.batch_size = min(main_config.batch_size, transfer_config.batch_size)
    main_config.gradient_clip = transfer_config.gradient_clip

    print(
        f"Applied transfer learning configuration for "
        f"{main_config.target_dataset_name} dataset"
    )
    return main_config


# =============================================================================
# TransferLearningConfig (converted to dataclass)
# =============================================================================

@dataclass
class TransferLearningConfig:
    """Configuration dataclass for transfer learning parameters."""

    # Source model (pre-trained) settings
    pretrained_model_path: Optional[str] = None
    source_dataset_name: str = 'METR-LA'

    # Target dataset settings
    target_dataset_name: str = 'mozambique'
    target_data_path: Optional[str] = None
    test_split: float = 0.2

    # Transfer learning strategy
    freeze_encoder: bool = True
    freeze_layers: int = 1
    adapter_dim: int = 64
    use_adapters: bool = True

    # Fine-tuning hyperparameters
    learning_rate: float = 5e-5
    num_epochs: int = 50
    patience: int = 5
    batch_size: int = 32
    gradient_clip: float = 1.0

    # Scheduler settings
    scheduler_type: Optional[str] = 'plateau'
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    # Evaluation settings
    evaluate_on_source: bool = True
    save_comparison_report: bool = True

    # Visualisation settings
    visualize_attention: bool = True
    visualize_training_curve: bool = True
    visualize_metrics_comparison: bool = True

    # Output settings
    save_fine_tuned_model: bool = True
    fine_tuned_model_prefix: str = 'fine_tuned'

    def update(self, **kwargs: Any) -> 'TransferLearningConfig':
        """Update attributes from keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: TransferLearningConfig has no attribute '{key}'")
        return self

    def print_summary(self) -> None:
        """Print a human-readable summary of the configuration."""
        print("\n==== Transfer Learning Configuration ====")
        print(f"Target Dataset: {self.target_dataset_name}")
        print(f"Pre-trained Model: {self.pretrained_model_path}")
        print(
            f"Strategy: "
            f"{'Partial fine-tuning' if self.freeze_encoder else 'Full fine-tuning'}"
        )
        if self.freeze_encoder:
            print(f"  - Freezing {self.freeze_layers} transformer layers")
        print(f"  - Adapters: {'Enabled' if self.use_adapters else 'Disabled'}")
        if self.use_adapters:
            print(f"    - Adapter dimension: {self.adapter_dim}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Max Epochs: {self.num_epochs}")
        print("========================================\n")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a plain dictionary."""
        return asdict(self)
