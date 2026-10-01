"""
VISU Predict: traffic forecasting for road sensor networks.

V19 of the VISU traffic model: a node-level spatio-temporal Transformer
(STAEformer family) evaluated under the standard METR-LA / PEMS-BAY protocol.

Quick use from Python::

    from visu_predict import STTrainConfig, STTransformer, fit, load_st_benchmark

    data = load_st_benchmark("data", "METR-LA", device="cuda")
    model = STTransformer(num_nodes=data.num_nodes, steps_per_day=data.steps_per_day)
    results = fit(model, data, STTrainConfig(max_epochs=200), run_dir="runs/metr-la", device="cuda")
    print(results["test"]["h12"])   # 60-minute MAE / RMSE / MAPE

The previous model (V18 ``TrafficTransformer``) lives in :mod:`visu_predict.legacy`.
"""

__version__ = "0.2.0"

from .data import STDataBundle, ZScoreScaler, load_st_benchmark
from .metrics import horizon_metrics, masked_mae_loss, masked_metrics
from .model import GraphDistanceBias, STTransformer, build_model
from .training import STTrainConfig, evaluate, fit, load_checkpoint, naive_baselines, predict

__all__ = [
    "GraphDistanceBias",
    "STDataBundle",
    "STTrainConfig",
    "STTransformer",
    "ZScoreScaler",
    "__version__",
    "build_model",
    "evaluate",
    "fit",
    "horizon_metrics",
    "load_checkpoint",
    "load_st_benchmark",
    "masked_mae_loss",
    "masked_metrics",
    "naive_baselines",
    "predict",
]
