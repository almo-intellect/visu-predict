import json
import os

import numpy as np

from visu_predict import STTrainConfig, STTransformer, evaluate, fit, load_checkpoint, load_st_benchmark
from visu_predict.training import naive_baselines

TINY = dict(input_embedding_dim=8, tod_embedding_dim=8, dow_embedding_dim=8, adaptive_embedding_dim=8,
            feed_forward_dim=32, num_heads=2, num_temporal_layers=1, num_spatial_layers=1)


def _cfg(**kwargs):
    base = dict(max_epochs=2, patience=5, max_train_batches=4, save_predictions=True, milestones=(1,))
    return STTrainConfig(**{**base, **kwargs})


def test_fit_writes_results_and_a_self_contained_checkpoint(syn, tmp_path):
    root, _ = syn
    data = load_st_benchmark(root, "SYN", batch_size=8)
    model = STTransformer(num_nodes=data.num_nodes, **TINY)
    res = fit(model, data, _cfg(), str(tmp_path), device="cpu")

    for name in ("results.json", "history.json", "train_log.txt", "best.pt", "last.pt", "test_predictions.npz"):
        assert os.path.exists(tmp_path / name), name
    assert set(res["test"]) == {"h3", "h6", "h12", "all"}
    assert res["model_class"] == "STTransformer" and res["model_config"]["num_nodes"] == 6
    assert np.isfinite(res["test"]["all"]["mae"])
    with open(tmp_path / "results.json") as f:
        assert json.load(f)["epochs_run"] == 2

    # best.pt alone is enough to rebuild the model and reproduce the test metrics
    rebuilt, scaler, _ = load_checkpoint(str(tmp_path / "best.pt"))
    assert scaler.state_dict() == data.scaler.state_dict()
    metrics, preds, labels = evaluate(rebuilt, data)
    assert np.isclose(metrics["all"]["mae"], res["test"]["all"]["mae"], atol=1e-5)
    saved = np.load(tmp_path / "test_predictions.npz")
    assert saved["pred"].shape == tuple(preds.shape) == tuple(labels.shape)


def test_resume_continues_from_last_epoch(syn, tmp_path):
    root, _ = syn
    data = load_st_benchmark(root, "SYN", batch_size=8)
    fit(STTransformer(num_nodes=6, **TINY), data, _cfg(max_epochs=1, save_predictions=False), str(tmp_path), "cpu")
    res = fit(STTransformer(num_nodes=6, **TINY), data, _cfg(max_epochs=2, resume=True, save_predictions=False),
              str(tmp_path), "cpu")
    assert res["epochs_run"] == 2
    with open(tmp_path / "train_log.txt", encoding="utf-8") as f:
        assert "Resumed from" in f.read()


def test_training_reduces_the_error(syn, tmp_path):
    root, _ = syn
    data = load_st_benchmark(root, "SYN", batch_size=16)
    cfg = _cfg(max_epochs=4, max_train_batches=None, save_predictions=False, milestones=(20, 30))
    res = fit(STTransformer(num_nodes=6, **TINY), data, cfg, str(tmp_path), "cpu")
    with open(tmp_path / "history.json") as f:
        val = [h["val_mae"] for h in json.load(f)]
    assert val[-1] < 0.8 * val[0]
    assert len(res["test_per_step"]) == 12


def test_naive_baselines_are_finite(syn):
    root, _ = syn
    res = naive_baselines(load_st_benchmark(root, "SYN"))
    assert set(res) == {"persistence", "historical_average"}
    assert all(np.isfinite(r["all"]["mae"]) for r in res.values())
