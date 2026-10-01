"""
Seed statistics and prediction ensembles for finished runs.

* :func:`aggregate` groups runs whose training options differ only by seed
  and reports mean +/- sample standard deviation per metric and horizon.
  Seed spread on these benchmarks (0.02-0.03 mph MAE) exceeds many published
  model-to-model gaps, so single runs should not be compared.
* :func:`ensemble` averages the saved test predictions of several runs
  (``--save-predictions``) and scores the average under the same masked
  protocol. Each member is re-scored first, so the fp16 storage error of
  the prediction files is reported alongside.
"""

import json
import os
import re
from collections.abc import Sequence

import numpy as np

HORIZON_KEYS = ("h3", "h6", "h12", "all")
METRICS = ("mae", "rmse", "mape")
# options that change how a run executes, not what it learns
EXECUTION_ARGS = {
    "seed", "run_name", "resume", "output_dir", "out_dir", "input_dir", "data_dir", "device",
    "save_predictions", "log_every", "compile", "precision", "max_train_batches", "max_eval_batches",
}


def find_runs(paths: Sequence[str]) -> list[str]:
    """Expand ``paths`` into run directories (folders holding a results.json with test metrics)."""
    runs: list[str] = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "results.json")):
            runs.append(p)
        elif os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                sub = os.path.join(p, name)
                if os.path.isfile(os.path.join(sub, "results.json")):
                    runs.append(sub)
    return [r for r in runs if "test" in _load(r)]


def _load(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
        return json.load(f)


def _config_key(result: dict, fallback: str) -> str:
    args = result.get("args")
    if not args:
        return fallback
    return json.dumps({k: v for k, v in sorted(args.items()) if k not in EXECUTION_ARGS}, default=str)


def _group_label(names: list[str]) -> str:
    prefix = os.path.commonprefix(names) if len(names) > 1 else names[0]
    return re.sub(r"(_s\d*)?_*$", "", prefix) or names[0]


def aggregate(run_dirs: Sequence[str]) -> list[dict]:
    """Group runs by configuration and compute mean / sample std per metric."""
    groups: dict[str, list[str]] = {}
    for r in run_dirs:
        groups.setdefault(_config_key(_load(r), r), []).append(r)
    out = []
    for runs in groups.values():
        results = [_load(r) for r in runs]
        names = [os.path.basename(os.path.normpath(r)) for r in runs]
        entry = {
            "label": _group_label(names),
            "runs": names,
            "seeds": [res.get("train_config", {}).get("seed") for res in results],
            "params": results[0].get("params"),
            "metrics": {},
        }
        for h in HORIZON_KEYS:
            entry["metrics"][h] = {}
            for m in METRICS:
                vals = np.array([res["test"][h][m] for res in results], dtype=np.float64)
                entry["metrics"][h][m] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "values": [round(float(v), 4) for v in vals],
                }
        out.append(entry)
    return sorted(out, key=lambda e: e["label"])


def format_aggregate(groups: list[dict]) -> str:
    lines = []
    for metric, unit in (("mae", ""), ("mape", "%")):
        lines.append(f"\nTest {metric.upper()} (mean ± sd over runs)")
        lines.append("| Configuration | runs | 15 min | 30 min | 60 min | average |")
        lines.append("|---|---|---|---|---|---|")
        for g in groups:
            cells = " | ".join(f"{g['metrics'][h][metric]['mean']:.3f}{unit} ± {g['metrics'][h][metric]['std']:.3f}"
                               for h in HORIZON_KEYS)
            lines.append(f"| {g['label']} | {len(g['runs'])} | {cells} |")
    return "\n".join(lines)


# =============================================================================
# Ensembles
# =============================================================================

def score(pred: np.ndarray, label: np.ndarray, horizons: Sequence[int] = (3, 6, 12),
          null_val: float = 0.0) -> dict[str, dict[str, float]]:
    """Masked MAE / RMSE / MAPE per horizon and pooled, computed step by step
    (same definition as :func:`visu_predict.metrics.horizon_metrics`, with
    far less memory for large test sets)."""
    sums = []
    for i in range(pred.shape[1]):
        p = pred[:, i].astype(np.float64)
        y = label[:, i].astype(np.float64)
        m = ~np.isnan(y) & (np.abs(y - null_val) > 1e-6)
        e = p[m] - y[m]
        ae = np.abs(e)
        sums.append((ae.sum(), (e ** 2).sum(), (ae / np.abs(y[m])).sum(), int(m.sum())))

    def fmt(a, s, r, n):
        return {"mae": a / n, "rmse": float(np.sqrt(s / n)), "mape": 100.0 * r / n, "count": n}

    out = {f"h{h}": fmt(*sums[h - 1]) for h in horizons if 1 <= h <= pred.shape[1]}
    out["all"] = fmt(*(sum(x[k] for x in sums) for k in range(4)))
    return out


def ensemble(run_dirs: Sequence[str]) -> dict:
    """Score the mean of the runs' saved test predictions."""
    if len(run_dirs) < 2:
        raise ValueError("an ensemble needs at least two runs")
    label0, acc, members = None, None, {}
    for r in run_dirs:
        path = os.path.join(r, "test_predictions.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - train with --save-predictions "
                                    "or create it with `visu-predict evaluate --save-predictions`")
        d = np.load(path)
        p, y = d["pred"].astype(np.float32), d["label"].astype(np.float32)
        if label0 is None:
            label0 = y
        elif y.shape != label0.shape or not np.array_equal(y, label0, equal_nan=True):
            raise ValueError(f"{r}: test labels differ from {run_dirs[0]} - not the same test set")
        acc = p.copy() if acc is None else acc + p
        name = os.path.basename(os.path.normpath(r))
        members[name] = {"recomputed": score(p, y)}
        res = _load(r) if os.path.exists(os.path.join(r, "results.json")) else {}
        if "test" in res:
            members[name]["recorded"] = {k: {m: res["test"][k][m] for m in METRICS} for k in HORIZON_KEYS}
        del p, y
    ens = score(acc / len(run_dirs), label0)
    deviation = max((abs(v["recomputed"][k][m] - v["recorded"][k][m])
                     for v in members.values() if "recorded" in v
                     for k in HORIZON_KEYS for m in ("mae", "mape")), default=None)
    mean_members = {k: {m: float(np.mean([v["recomputed"][k][m] for v in members.values()]))
                        for m in METRICS} for k in HORIZON_KEYS}
    return {"runs": [os.path.basename(os.path.normpath(r)) for r in run_dirs], "ensemble": ens,
            "mean_of_members": mean_members, "members": members, "storage_deviation": deviation}


def format_ensemble(result: dict) -> str:
    rows = [(name, v["recomputed"]) for name, v in result["members"].items()]
    rows += [("mean of runs", result["mean_of_members"]), ("ENSEMBLE", result["ensemble"])]
    lines = ["| | 15 min MAE / MAPE | 30 min MAE / MAPE | 60 min MAE / MAPE | average MAE / MAPE |",
             "|---|---|---|---|---|"]
    for name, r in rows:
        cells = " | ".join(f"{r[k]['mae']:.3f} / {r[k]['mape']:.2f}%" for k in HORIZON_KEYS)
        lines.append(f"| {name} | {cells} |")
    if result["storage_deviation"] is not None:
        lines.append(f"\nRe-scored members match their results.json within {result['storage_deviation']:.4f} "
                     "(fp16 storage of the prediction files).")
    return "\n".join(lines)
