"""
traffic_transformer - Transformer-based traffic flow prediction package.

Modules are organized in a strict layered dependency hierarchy:
  Layer 0: utils, config          (no internal deps)
  Layer 1: weather, spatial       (depends on L0)
  Layer 2: data_module            (depends on L0-L1)
  Layer 3: model                  (depends on L0-L2)
  Layer 4: training               (depends on L0-L3)
  Layer 5: transfer_learning      (depends on L0-L4)
  Layer 6: visualization, report  (depends on L0-L5)
"""

# Layer 0
from .utils import (
    AMP_AVAILABLE,
    DEVICE_TYPE_SUPPORTED,
    TORCH_GEOMETRIC_AVAILABLE,
    STATSMODELS_AVAILABLE,
    OPTUNA_AVAILABLE,
    IN_COLAB,
    set_reproducibility_seed,
    get_maputo_timestamp,
    get_gpu_memory_info,
    TeeLogger,
    save_predictions_and_actuals,
    save_experiment_results,
)

from .config import (
    TrainingConfig,
    TransferLearningConfig,
    setup_directories,
    get_adjacency_matrix_path,
    load_config,
    apply_transfer_config,
    ensure_compatible_dimensions,
)

# Layer 1
from .weather import WeatherIntegration
from .spatial import (
    load_adjacency_matrix,
    normalize_adj,
    SpatialIntegration,
    GCNEncoder,
)

# Layer 2
from .data_module import TrafficDataset, prepare_data, load_and_prepare_data

# Layer 3
from .model import (
    TrafficTransformer,
    PositionalEncoding,
    FeatureAttention,
    CosineWarmupLR,
    PyTorchLSTMForecaster,
)

# Layer 4
from .training import (
    train_model,
    evaluate_model,
    predict,
    create_optimizer,
    create_scheduler,
    create_criterion,
)

# Layer 5
from .transfer_learning import (
    AdapterLayer,
    TransferLearningModule,
    train_transfer_model,
)

# Layer 6
from .visualization import (
    plot_attention_weights,
    visualize_decoder_attention,
    plot_layer_progression,
    visualize_feature_attribution,
    visualize_prediction_explanation,
    visualize_hidden_states,
    plot_attention_heads,
    visualize_feature_importance,
    create_feature_mapping,
    plot_predictions_vs_actual,
    plot_training_history,
    create_summary_comparison_plot,
)

from .report import generate_traffic_report

# V19: benchmark-protocol pipeline + node-level spatio-temporal transformer
from .metrics import horizon_metrics, masked_metrics, masked_mae_loss, format_horizon_table
from .st_data import load_st_benchmark, STDataBundle, ZScoreScaler
from .st_model import STTransformer, LegacyTransformerAdapter, GraphDistanceBias
from .st_training import STTrainConfig, fit as fit_st, predict as predict_st, naive_baselines

__all__ = [
    # V19 benchmark pipeline
    "horizon_metrics", "masked_metrics", "masked_mae_loss", "format_horizon_table",
    "load_st_benchmark", "STDataBundle", "ZScoreScaler",
    "STTransformer", "LegacyTransformerAdapter", "GraphDistanceBias",
    "STTrainConfig", "fit_st", "predict_st", "naive_baselines",
    # utils
    "AMP_AVAILABLE", "DEVICE_TYPE_SUPPORTED", "TORCH_GEOMETRIC_AVAILABLE",
    "set_reproducibility_seed", "get_maputo_timestamp", "get_gpu_memory_info",
    "TeeLogger", "save_predictions_and_actuals", "save_experiment_results",
    # config
    "TrainingConfig", "TransferLearningConfig", "setup_directories",
    "get_adjacency_matrix_path", "load_config", "apply_transfer_config",
    "ensure_compatible_dimensions",
    # weather & spatial
    "WeatherIntegration", "load_adjacency_matrix", "normalize_adj",
    "SpatialIntegration", "GCNEncoder",
    # data
    "TrafficDataset", "prepare_data", "load_and_prepare_data",
    # model
    "TrafficTransformer", "PositionalEncoding", "FeatureAttention",
    "CosineWarmupLR", "PyTorchLSTMForecaster",
    # training
    "train_model", "evaluate_model", "predict",
    "create_optimizer", "create_scheduler", "create_criterion",
    # transfer learning
    "AdapterLayer", "TransferLearningModule", "train_transfer_model",
    # visualization
    "plot_attention_weights", "visualize_decoder_attention",
    "plot_layer_progression", "visualize_feature_attribution",
    "visualize_prediction_explanation", "visualize_hidden_states",
    "plot_attention_heads", "visualize_feature_importance",
    "create_feature_mapping", "plot_predictions_vs_actual",
    "plot_training_history", "create_summary_comparison_plot",
    # report
    "generate_traffic_report",
]
