#!/usr/bin/env python
"""
Benchmark runner (standard METR-LA / PEMS-BAY protocol).

Examples
--------
  # sanity baselines (persistence, historical average)
  python run_benchmark.py --dataset PEMS-BAY --model baselines

  # new node-level spatio-temporal transformer, STAEformer-style recipe
  python run_benchmark.py --dataset PEMS-BAY --model st_transformer --precision tf32

  # + road-graph attention prior, + weather, + holidays
  python run_benchmark.py --dataset PEMS-BAY --model st_transformer --graph-bias --weather --holidays

  # legacy V18 TrafficTransformer evaluated under the same protocol
  python run_benchmark.py --dataset PEMS-BAY --model legacy

Each run writes <output-dir>/<run-name>/{train_log.txt, history.json,
results.json, best.pt, last.pt}. Re-running with --resume --run-name <name>
continues an interrupted run.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from traffic_transformer.metrics import format_horizon_table  # noqa: E402
from traffic_transformer.st_data import load_st_benchmark  # noqa: E402
from traffic_transformer.st_model import LegacyTransformerAdapter, STTransformer  # noqa: E402
from traffic_transformer.st_training import (  # noqa: E402
    RunLogger, STTrainConfig, count_parameters, fit, naive_baselines,
)

# STAEformer-style recipe per dataset: (weight_decay, milestones, max_epochs, patience)
RECIPES = {
    "METR-LA": (3e-4, (20, 30), 200, 30),
    "PEMS-BAY": (1e-4, (10, 30), 300, 20),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--dataset", default="PEMS-BAY")
    p.add_argument("--input-dir", default=os.path.join(here, "Transformers_Input"))
    p.add_argument("--output-dir", default=os.path.join(here, "results", "benchmark"))
    p.add_argument("--model", choices=["st_transformer", "legacy", "baselines"], default="st_transformer")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    # data
    p.add_argument("--in-steps", type=int, default=12)
    p.add_argument("--out-steps", type=int, default=12)
    p.add_argument("--weather", action="store_true")
    p.add_argument("--holidays", action="store_true")
    p.add_argument("--history-lags", type=int, nargs="*", default=[],
                   help="e.g. 288 2016: add same-time-yesterday / last-week channels (5-min data)")
    # model (STTransformer)
    p.add_argument("--dim-input", type=int, default=24)
    p.add_argument("--dim-tod", type=int, default=24)
    p.add_argument("--dim-dow", type=int, default=24)
    p.add_argument("--dim-node", type=int, default=0)
    p.add_argument("--dim-adaptive", type=int, default=80)
    p.add_argument("--dim-exo", type=int, default=16)
    p.add_argument("--ff-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--t-layers", type=int, default=3)
    p.add_argument("--s-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm-first", action="store_true")
    p.add_argument("--graph-bias", action="store_true")
    p.add_argument("--graph-max-hops", type=int, default=6)
    # model (legacy)
    p.add_argument("--legacy-d-model", type=int, default=256)
    p.add_argument("--legacy-layers", type=int, default=4)
    p.add_argument("--legacy-heads", type=int, default=8)
    # training
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--milestones", type=int, nargs="*", default=None)
    p.add_argument("--lr-decay", type=float, default=0.1)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    p.add_argument("--scheduler", choices=["multistep", "cosine", "none"], default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--clip-grad", type=float, default=0.0)
    p.add_argument("--loss", choices=["masked_mae", "masked_huber"], default="masked_mae")
    p.add_argument("--precision", choices=["fp32", "tf32", "bf16", "fp16"], default="tf32")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=None, help="smoke tests only")
    p.add_argument("--max-eval-batches", type=int, default=None, help="smoke tests only")
    return p.parse_args(argv)


def build_train_config(a) -> STTrainConfig:
    wd, milestones, epochs, patience = RECIPES.get(a.dataset, (3e-4, (20, 30), 200, 30))
    if a.model == "legacy":
        # the V18 recipe: AdamW 1e-4, cosine with 10 warm-up epochs, 100 epochs
        defaults = dict(lr=1e-4, weight_decay=1e-2, optimizer="adamw", scheduler="cosine",
                        warmup_epochs=10, max_epochs=100, patience=30, milestones=milestones)
    else:
        defaults = dict(lr=1e-3, weight_decay=wd, optimizer="adam", scheduler="multistep",
                        warmup_epochs=0, max_epochs=epochs, patience=patience, milestones=milestones)
    pick = lambda v, k: defaults[k] if v is None else v  # noqa: E731
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


def main(argv=None):
    a = parse_args(argv)
    tag = a.model
    if a.model == "st_transformer":
        tag += ("_gb" if a.graph_bias else "") + ("_wx" if a.weather else "") + ("_hol" if a.holidays else "")
        tag += "_hist" if a.history_lags else ""
    run_name = a.run_name or f"{a.dataset}_{tag}_s{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(a.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log = RunLogger(run_dir)
    log(f"=== {run_name} ===")
    log("args: " + json.dumps(vars(a)))

    t0 = time.time()
    data = load_st_benchmark(
        a.input_dir, a.dataset, batch_size=a.batch_size, in_steps=a.in_steps,
        out_steps=a.out_steps, use_weather=a.weather, use_holidays=a.holidays,
        device=a.device, seed=a.seed, history_lags=a.history_lags,
    )
    log(f"data: {json.dumps(data.info)} (loaded in {time.time() - t0:.1f}s)")

    if a.model == "baselines":
        res = naive_baselines(data)
        out = {}
        for name, r in res.items():
            log(format_horizon_table(r, title=f"{a.dataset} | {name}"))
            out[name] = {k: v for k, v in r.items() if k != "per_step"}
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(out, f, indent=1)
        return out

    if a.model == "legacy":
        model = LegacyTransformerAdapter(
            data.num_nodes, data.steps_per_day, a.out_steps, d_model=a.legacy_d_model,
            nhead=a.legacy_heads, num_layers=a.legacy_layers,
        )
    else:
        model = STTransformer(
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
    cfg = build_train_config(a)
    log(f"model: {type(model).__name__} with {count_parameters(model):,} parameters")
    log("train config: " + json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in cfg.__dict__.items()}))
    results = fit(model, data, cfg, run_dir, device=a.device,
                  extra_info={"run_name": run_name, "args": vars(a),
                              "wall_time_min": None})
    results["wall_time_min"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    log(f"done in {results['wall_time_min']} min -> {run_dir}")
    return results


if __name__ == "__main__":
    main()
