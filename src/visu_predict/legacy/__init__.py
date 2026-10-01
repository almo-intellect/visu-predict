"""
visu_predict.legacy - the V18 ``TrafficTransformer`` package (formerly ``traffic_transformer``).

Kept so that earlier experiments stay reproducible and so the V18 model can be
scored under the V19 benchmark protocol (``visu-predict train --model legacy``).
It is not recommended for new work: under the standard protocol V18 is worse
than simply repeating the last reading at 15 minutes (see docs/results.md).
Needs the optional dependencies: ``pip install "visu-predict[legacy]"``.

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

# Runs the V18 model inside the V19 benchmark harness. The V19 pipeline itself
# (STTransformer, load_st_benchmark, fit, ...) is imported from ``visu_predict``.
from .adapter import LegacyTransformerAdapter

__all__ = [
    # benchmark-harness adapter
    "LegacyTransformerAdapter",
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
