"""
Training / evaluation loop for the benchmark protocol (node-level models).

Recipe follows the strong baselines of the METR-LA / PEMS-BAY literature:
masked MAE computed on the original scale, Adam with step decay, early
stopping on validation MAE, evaluation per horizon on the untouched test set.

Resilience for Colab: ``last.pt`` (full state, for ``resume=True``) and
``best.pt`` are written to ``run_dir`` every epoch; ``train_log.txt`` is
flushed per epoch, so a run directory on Google Drive can be monitored live.
"""

import copy
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .metrics import format_horizon_table, horizon_metrics, masked_huber_loss, masked_mae_loss, masked_metrics
from .st_data import STDataBundle


@dataclass
class STTrainConfig:
    max_epochs: int = 200
    patience: int = 30
    lr: float = 1e-3
    weight_decay: float = 3e-4
    optimizer: str = "adam"                 # 'adam' | 'adamw'
    scheduler: str = "multistep"            # 'multistep' | 'cosine' | 'none'
    milestones: Sequence[int] = (20, 30)
    lr_decay: float = 0.1
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    clip_grad: float = 0.0
    loss: str = "masked_mae"                # 'masked_mae' | 'masked_huber'
    null_val: float = 0.0
    precision: str = "fp32"                 # 'fp32' | 'tf32' | 'bf16' | 'fp16'
    seed: int = 42
    max_train_batches: Optional[int] = None  # for smoke tests
    max_eval_batches: Optional[int] = None
    log_every: int = 0                      # batches; 0 = epoch summaries only
    compile: bool = False
    resume: bool = False
    save_predictions: bool = False
    horizons: Sequence[int] = (3, 6, 12)


# =============================================================================
# Utilities
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunLogger:
    """Print + append to ``<run_dir>/train_log.txt`` with an explicit flush."""

    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "train_log.txt")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _atomic_save(obj, path: str) -> None:
    """Write to a temp file, then rename: an interrupted save (e.g. Colab
    disconnect) can never leave a truncated checkpoint behind."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _autocast(device: str, precision: str):
    if device.startswith("cuda") and precision in ("bf16", "fp16"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device == "cpu" and precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def build_optimizer(model: nn.Module, cfg: STTrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-8)


def build_scheduler(opt: torch.optim.Optimizer, cfg: STTrainConfig):
    if cfg.scheduler == "multistep":
        base = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=list(cfg.milestones), gamma=cfg.lr_decay)
    elif cfg.scheduler == "cosine":
        total = max(1, cfg.max_epochs - cfg.warmup_epochs)
        base = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=cfg.min_lr)
    else:
        return None
    if cfg.warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, total_iters=cfg.warmup_epochs)
        return torch.optim.lr_scheduler.SequentialLR(
            opt, [warm, base], milestones=[cfg.warmup_epochs])
    return base


def _forward(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(batch["x"], batch["tod"], batch["dow"], batch.get("exo"))


@torch.no_grad()
def predict(model: nn.Module, batcher, scaler, device: str, precision: str = "fp32",
            max_batches: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (predictions, labels) on the original scale, on CPU."""
    model.eval()
    preds: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for i, batch in enumerate(batcher):
        if max_batches is not None and i >= max_batches:
            break
        with _autocast(device, precision):
            out = _forward(model, batch)
        preds.append(scaler.inverse_transform(out.float()).cpu())
        labels.append(batch["y"].float().cpu())
    return torch.cat(preds), torch.cat(labels)


# =============================================================================
# Training
# =============================================================================

def fit(
    model: nn.Module,
    data: STDataBundle,
    cfg: STTrainConfig,
    run_dir: str,
    device: str = "cuda",
    extra_info: Optional[Dict] = None,
) -> Dict:
    """Train with early stopping, then evaluate the best weights on test."""
    log = RunLogger(run_dir)
    seed_everything(cfg.seed)

    if device.startswith("cuda"):
        allow_tf32 = cfg.precision in ("tf32", "bf16", "fp16")
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    model = model.to(device)
    fwd_model = torch.compile(model) if cfg.compile else model
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    loss_fn = masked_huber_loss if cfg.loss == "masked_huber" else masked_mae_loss
    use_scaler = device.startswith("cuda") and cfg.precision == "fp16"
    grad_scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    n_params = count_parameters(model)
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")

    start_epoch, best_val, bad_epochs = 0, math.inf, 0
    history: List[Dict] = []
    if cfg.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched is not None and ck.get("scheduler") is not None:
            sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val, bad_epochs, history = ck["best_val"], ck["bad_epochs"], ck["history"]
        # the shuffling generator lives on the CPU; map_location may have
        # moved its saved state to the GPU
        data.train.generator.set_state(ck["batch_rng"].cpu())
        log(f"Resumed from {last_path} at epoch {start_epoch + 1} (best val MAE {best_val:.4f})")
    else:
        log(f"Parameters: {n_params:,} | device {device} | precision {cfg.precision}")
        log(f"Train/val/test windows: {data.train.num_samples}/{data.val.num_samples}/"
            f"{data.test.num_samples} | batch {data.train.batch_size}")

    scaler = data.scaler
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        t0 = time.time()
        loss_sum = torch.zeros((), device=device)
        seen = 0
        for i, batch in enumerate(data.train):
            if cfg.max_train_batches is not None and i >= cfg.max_train_batches:
                break
            with _autocast(device, cfg.precision):
                out = _forward(fwd_model, batch)
            pred = scaler.inverse_transform(out.float())
            loss = loss_fn(pred, batch["y"], cfg.null_val)
            opt.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                if cfg.clip_grad > 0:
                    grad_scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
                loss.backward()
                if cfg.clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                opt.step()
            bsz = batch["y"].shape[0]
            loss_sum += loss.detach() * bsz
            seen += bsz
            if cfg.log_every and (i + 1) % cfg.log_every == 0:
                log(f"  epoch {epoch + 1} batch {i + 1}/{len(data.train)} "
                    f"loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
        train_loss = float(loss_sum.item()) / max(1, seen)
        train_time = time.time() - t0

        vp, vl = predict(fwd_model, data.val, scaler, device, cfg.precision, cfg.max_eval_batches)
        vm = masked_metrics(vp, vl, cfg.null_val)
        lr_now = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step()

        improved = vm["mae"] < best_val - 1e-5
        if improved:
            best_val, bad_epochs = vm["mae"], 0
            _atomic_save({"model": model.state_dict(), "epoch": epoch, "val_mae": best_val,
                          "scaler": scaler.state_dict()}, best_path)
        else:
            bad_epochs += 1

        rec = {"epoch": epoch + 1, "train_loss": train_loss, "val_mae": vm["mae"],
               "val_rmse": vm["rmse"], "val_mape": vm["mape"], "lr": lr_now,
               "epoch_time_s": round(time.time() - t0, 1), "train_time_s": round(train_time, 1)}
        history.append(rec)
        log(f"Epoch {epoch + 1:3d} | train {train_loss:.4f} | val MAE {vm['mae']:.4f} "
            f"RMSE {vm['rmse']:.4f} MAPE {vm['mape']:.2f}% | lr {lr_now:.1e} | "
            f"{rec['epoch_time_s']:.0f}s{' *' if improved else ''}")

        _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                      "scheduler": sched.state_dict() if sched is not None else None,
                      "epoch": epoch, "best_val": best_val, "bad_epochs": bad_epochs,
                      "history": history, "batch_rng": data.train.generator.get_state()},
                     last_path)
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

        if bad_epochs >= cfg.patience:
            log(f"Early stopping at epoch {epoch + 1} (no val improvement for {cfg.patience} epochs)")
            break

    # ---- Test with the best weights ------------------------------------
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        best_epoch = ck["epoch"] + 1
    else:
        best_epoch = len(history)
    # final test pass in full precision (tf32 matmuls at most), whatever the
    # training precision, so reported metrics carry no bf16 rounding noise
    eval_precision = "tf32" if cfg.precision in ("bf16", "fp16") else cfg.precision
    tp, tl = predict(model, data.test, scaler, device, eval_precision, cfg.max_eval_batches)
    test = horizon_metrics(tp, tl, cfg.horizons, cfg.null_val)
    log(format_horizon_table(test, title=f"TEST (best epoch {best_epoch})"))

    if cfg.save_predictions:
        np.savez_compressed(os.path.join(run_dir, "test_predictions.npz"),
                            pred=tp.numpy().astype(np.float16),
                            label=tl.numpy().astype(np.float16))

    peak_mem = (torch.cuda.max_memory_allocated() / 2**30
                if device.startswith("cuda") and torch.cuda.is_available() else None)
    results = {
        "test": {k: v for k, v in test.items() if k != "per_step"},
        "test_per_step": test["per_step"],
        "best_epoch": best_epoch, "best_val_mae": best_val,
        "epochs_run": len(history), "params": n_params,
        "mean_epoch_time_s": float(np.mean([h["epoch_time_s"] for h in history])) if history else None,
        "peak_gpu_mem_gb": peak_mem,
        "train_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "data": data.info,
    }
    if extra_info:
        results.update(extra_info)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    return results


# =============================================================================
# Naive baselines (sanity checks for the protocol)
# =============================================================================

def naive_baselines(data: STDataBundle, horizons: Sequence[int] = (3, 6, 12),
                    null_val: float = 0.0) -> Dict[str, Dict]:
    """Persistence (last observed value) and historical average by
    (day type, time-of-day slot) estimated on the training period."""
    scaler = data.scaler
    res = {}

    # Persistence
    preds, labels = [], []
    for batch in data.test:
        last = scaler.inverse_transform(batch["x"][:, -1, :, 0])            # (B, N)
        preds.append(last[:, None, :].expand_as(batch["y"]).cpu())
        labels.append(batch["y"].cpu())
    res["persistence"] = horizon_metrics(torch.cat(preds), torch.cat(labels), horizons, null_val)

    # Historical average over (day type, slot), zeros excluded
    y_all = data.train.t["y"].cpu().numpy()
    tod = data.train.t["tod"].cpu().numpy()
    dow = data.train.t["dow"].cpu().numpy()
    train_end = int(data.splits["train"][-1]) + data.train.in_steps
    n_types = int(dow.max()) + 1
    table = np.zeros((n_types, data.steps_per_day, y_all.shape[1]))
    vals = y_all[:train_end]
    valid = vals != null_val
    key = dow[:train_end] * data.steps_per_day + tod[:train_end]
    sums = np.zeros((n_types * data.steps_per_day, y_all.shape[1]))
    cnts = np.zeros_like(sums)
    np.add.at(sums, key, np.where(valid, vals, 0.0))
    np.add.at(cnts, key, valid.astype(np.float64))
    overall = vals[valid].mean()
    table = np.where(cnts > 0, sums / np.maximum(cnts, 1), overall)
    starts = data.splits["test"]
    in_s, out_s = data.test.in_steps, data.test.out_steps
    yi = starts[:, None] + np.arange(in_s, in_s + out_s)[None, :]
    ha_pred = table[dow[yi] * data.steps_per_day + tod[yi]]                   # (S, out, N)
    res["historical_average"] = horizon_metrics(ha_pred, y_all[yi], horizons, null_val)
    return res
