"""
Train, evaluate and baseline one model under the standard protocol.

These functions back the ``visu-predict train / baselines / evaluate``
commands (see :mod:`visu_predict.cli`). Each training run writes
``<out>/<run-name>/{train_log.txt, history.json, results.json, best.pt, last.pt}``
and, with ``--save-predictions``, ``test_predictions.npz``.
"""

import argparse
import json
import os
import time
from typing import Any

import torch

from .data import STDataBundle, load_st_benchmark
from .metrics import format_horizon_table
from .model import STTransformer
from .training import (
    RunLogger,
    STTrainConfig,
    count_parameters,
    evaluate,
    fit,
    load_checkpoint,
    naive_baselines,
    save_predictions,
)

# STAEformer-style recipe per dataset: (weight_decay, milestones, max_epochs, patience)
RECIPES = {
    "METR-LA": (3e-4, (20, 30), 200, 30),
    "PEMS-BAY": (1e-4, (10, 30), 300, 20),
}
DEFAULT_RECIPE = (3e-4, (20, 30), 200, 30)

# Argument defaults, used to rebuild models of runs whose results.json predates an option.
MODEL_DEFAULTS: dict[str, Any] = {
    "model": "st_transformer", "in_steps": 12, "out_steps": 12, "dim_input": 24, "dim_tod": 24,
    "dim_dow": 24, "dim_node": 0, "dim_adaptive": 80, "dim_exo": 16, "ff_dim": 256, "heads": 4,
    "t_layers": 3, "s_layers": 3, "dropout": 0.1, "norm_first": False, "graph_bias": False,
    "graph_max_hops": 6, "legacy_d_model": 256, "legacy_layers": 4, "legacy_heads": 8,
    "weather": False, "weather_file": None, "holidays": False, "holiday_country": None,
    "history_lags": [],
}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_data(a: argparse.Namespace, eval_batch_size=None) -> STDataBundle:
    return load_st_benchmark(
        a.data_dir, a.dataset, batch_size=a.batch_size, in_steps=a.in_steps,
        out_steps=a.out_steps, use_weather=a.weather, weather_file=a.weather_file,
        use_holidays=a.holidays, holiday_country=a.holiday_country,
        device=a.device, seed=a.seed, history_lags=a.history_lags,
        eval_batch_size=eval_batch_size,
    )


def make_model(a: argparse.Namespace, data: STDataBundle) -> torch.nn.Module:
    """Build the model selected by ``a.model`` for the shapes of ``data``."""
    if a.model == "legacy":
        try:
            from .legacy.adapter import LegacyTransformerAdapter
        except ImportError as e:  # pragma: no cover - depends on optional extras
            raise SystemExit(f"--model legacy needs the legacy extras: pip install 'visu-predict[legacy]' ({e})") from e
        return LegacyTransformerAdapter(
            data.num_nodes, data.steps_per_day, a.out_steps, d_model=a.legacy_d_model,
            nhead=a.legacy_heads, num_layers=a.legacy_layers,
        )
    return STTransformer(
        num_nodes=data.num_nodes, in_steps=a.in_steps, out_steps=a.out_steps,
        steps_per_day=data.steps_per_day, num_day_types=data.num_day_types,
        input_dim=data.input_dim, input_embedding_dim=a.dim_input, tod_embedding_dim=a.dim_tod,
        dow_embedding_dim=a.dim_dow, node_embedding_dim=a.dim_node,
        adaptive_embedding_dim=a.dim_adaptive,
        exo_dim=data.exo_dim, exo_embedding_dim=a.dim_exo if data.exo_dim else 0,
        feed_forward_dim=a.ff_dim, num_heads=a.heads,
        num_temporal_layers=a.t_layers, num_spatial_layers=a.s_layers,
        dropout=a.dropout, norm_first=a.norm_first, adj=data.adj,
        graph_bias=a.graph_bias, graph_max_hops=a.graph_max_hops,
    )


def build_train_config(a: argparse.Namespace) -> STTrainConfig:
    wd, milestones, epochs, patience = RECIPES.get(a.dataset, DEFAULT_RECIPE)
    if a.model == "legacy":
        # the V18 recipe: AdamW 1e-4, cosine with 10 warm-up epochs, 100 epochs
        defaults = dict(lr=1e-4, weight_decay=1e-2, optimizer="adamw", scheduler="cosine",
                        warmup_epochs=10, max_epochs=100, patience=30, milestones=milestones)
    else:
        defaults = dict(lr=1e-3, weight_decay=wd, optimizer="adam", scheduler="multistep",
                        warmup_epochs=0, max_epochs=epochs, patience=patience, milestones=milestones)

    def pick(value, key):
        return defaults[key] if value is None else value

    return STTrainConfig(
        max_epochs=pick(a.epochs, "max_epochs"), patience=pick(a.patience, "patience"),
        lr=pick(a.lr, "lr"), weight_decay=pick(a.weight_decay, "weight_decay"),
        optimizer=pick(a.optimizer, "optimizer"), scheduler=pick(a.scheduler, "scheduler"),
        milestones=tuple(pick(a.milestones, "milestones")), lr_decay=a.lr_decay,
        warmup_epochs=pick(a.warmup_epochs, "warmup_epochs"), clip_grad=a.clip_grad,
        loss=a.loss, precision=a.precision, seed=a.seed, compile=a.compile, resume=a.resume,
        save_predictions=a.save_predictions, log_every=a.log_every,
        max_train_batches=a.max_train_batches, max_eval_batches=a.max_eval_batches,
    )


def default_run_name(a: argparse.Namespace) -> str:
    tag = a.model
    if a.model == "st_transformer":
        tag += ("_gb" if a.graph_bias else "") + ("_wx" if a.weather else "") + ("_hol" if a.holidays else "")
        tag += "_hist" if a.history_lags else ""
    return f"{a.dataset}_{tag}_s{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}"


# =============================================================================
# Commands
# =============================================================================

def run_baselines(a: argparse.Namespace) -> dict:
    """Persistence and historical-average scores on the test split."""
    run_dir = os.path.join(a.out_dir, a.run_name or f"{a.dataset}_baselines")
    os.makedirs(run_dir, exist_ok=True)
    log = RunLogger(run_dir)
    data = load_data(a)
    log(f"data: {json.dumps(data.info)}")
    out = {}
    for name, r in naive_baselines(data).items():
        log(format_horizon_table(r, title=f"{a.dataset} | {name}"))
        out[name] = {k: v for k, v in r.items() if k != "per_step"}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"-> {run_dir}")
    return out


def run_train(a: argparse.Namespace) -> dict:
    """Train one model with early stopping and report test metrics."""
    if a.model == "baselines":        # kept for queue files written for the old runner
        return run_baselines(a)
    run_name = a.run_name or default_run_name(a)
    run_dir = os.path.join(a.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log = RunLogger(run_dir)
    log(f"=== {run_name} ===")
    log("args: " + json.dumps(vars(a), default=str))

    t0 = time.time()
    data = load_data(a)
    log(f"data: {json.dumps(data.info)} (loaded in {time.time() - t0:.1f}s)")
    model = make_model(a, data)
    cfg = build_train_config(a)
    log(f"model: {type(model).__name__} with {count_parameters(model):,} parameters")
    log("train config: " + json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in cfg.__dict__.items()}))
    args_record = {k: v for k, v in vars(a).items() if k != "func"}
    results = fit(model, data, cfg, run_dir, device=a.device,
                  extra_info={"run_name": run_name, "args": args_record, "wall_time_min": None})
    results["wall_time_min"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    log(f"done in {results['wall_time_min']} min -> {run_dir}")
    return results


def _training_args(run_dir: str) -> dict[str, Any]:
    path = os.path.join(run_dir, "results.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("args", {}) or {}


def run_evaluate(a: argparse.Namespace) -> dict:
    """Re-score a trained run (or checkpoint) on the test split."""
    ckpt = a.run if a.run.endswith((".pt", ".pth")) else os.path.join(a.run, "best.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    run_dir = os.path.dirname(os.path.abspath(ckpt))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved = {**MODEL_DEFAULTS, **_training_args(run_dir)}
    info = ck.get("data", {}) or {}

    # data options must match training; take them from the run unless overridden
    ns = argparse.Namespace(**saved)
    ns.dataset = a.dataset or saved.get("dataset") or info.get("dataset")
    if not ns.dataset:
        raise SystemExit("cannot tell which dataset this run used; pass --dataset")
    ns.data_dir, ns.device, ns.seed = a.data_dir, a.device, saved.get("seed", 42)
    ns.batch_size = saved.get("batch_size", 16)
    ns.history_lags = saved.get("history_lags") or info.get("history_lags") or []
    data = load_data(ns, eval_batch_size=a.batch_size)
    if ns.weather and not data.exo_dim:
        raise SystemExit("this run was trained with --weather but no weather file was found in "
                         f"{a.data_dir}; build it with `visu-predict weather --data {a.data_dir}`")

    if ck.get("model_class") and ck.get("model_config") is not None:
        model, _, _ = load_checkpoint(ckpt, adj=data.adj, device=a.device)
    else:  # checkpoints written before model configs were stored
        model = make_model(ns, data)
        model.load_state_dict(ck["model"])
    if "scaler" in ck and abs(ck["scaler"]["mean"] - data.scaler.mean) > 1e-3:
        print(f"warning: training scaler mean {ck['scaler']['mean']:.4f} differs from this data "
              f"({data.scaler.mean:.4f}) - is this the same dataset file?")

    precision = a.precision or ("tf32" if a.device.startswith("cuda") else "fp32")
    metrics, preds, labels = evaluate(model, data, a.device, precision)
    print(format_horizon_table(metrics, title=f"{ns.dataset} | {os.path.basename(run_dir)} "
                                              f"(epoch {ck.get('epoch', -1) + 1})"))
    out = {k: v for k, v in metrics.items() if k != "per_step"}
    ref_path = os.path.join(run_dir, "results.json")
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as f:
            ref = json.load(f).get("test")
        if ref and "all" in ref:
            diff = out["all"]["mae"] - ref["all"]["mae"]
            print(f"recorded test MAE {ref['all']['mae']:.4f}, now {out['all']['mae']:.4f} ({diff:+.4f})")
    if a.save_predictions:
        save_predictions(a.save_predictions, preds, labels)
        print(f"predictions -> {a.save_predictions}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return out
