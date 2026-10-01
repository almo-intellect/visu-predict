import numpy as np
import torch

from visu_predict.analysis import score
from visu_predict.metrics import horizon_metrics, masked_mae_loss, masked_metrics

# one masked zero label: errors 2, 2, 4 on labels 10, 20, 40
Y = np.array([[[10.0, 0.0], [20.0, 40.0]]])      # (1 sample, 2 steps, 2 sensors)
P = np.array([[[12.0, 5.0], [18.0, 44.0]]])
MAE = (2 + 2 + 4) / 3
RMSE = np.sqrt((4 + 4 + 16) / 3)
MAPE = (0.2 + 0.1 + 0.1) / 3 * 100


def test_masked_metrics_exclude_zero_labels():
    m = masked_metrics(P, Y)
    assert np.isclose(m["mae"], MAE)
    assert np.isclose(m["rmse"], RMSE)
    assert np.isclose(m["mape"], MAPE)
    assert m["count"] == 3


def test_horizons_are_one_based():
    hm = horizon_metrics(P, Y, horizons=(1, 2))
    assert np.isclose(hm["h1"]["mae"], 2.0)
    assert np.isclose(hm["h2"]["mae"], 3.0)
    assert np.isclose(hm["all"]["mae"], MAE)
    assert len(hm["per_step"]) == 2


def test_torch_loss_matches_numpy_metric():
    assert np.isclose(masked_mae_loss(torch.tensor(P), torch.tensor(Y)).item(), MAE)


def test_low_memory_score_matches_horizon_metrics():
    rng = np.random.default_rng(1)
    y = rng.uniform(20, 70, (40, 12, 5))
    y[rng.random(y.shape) < 0.05] = 0.0
    p = y + rng.normal(0, 3, y.shape)
    ref, fast = horizon_metrics(p, y), score(p, y)
    for k in ("h3", "h6", "h12", "all"):
        for m in ("mae", "rmse", "mape"):
            assert np.isclose(ref[k][m], fast[k][m])
