"""
Masked forecasting metrics (benchmark protocol).

Follows the convention of the DCRNN / Graph WaveNet / STAEformer lineage so
numbers are directly comparable with published METR-LA / PEMS-BAY results:

* metrics are computed on the ORIGINAL scale (mph), never on scaled values;
* ground-truth entries equal to ``null_val`` (0.0 = sensor fault in METR-LA /
  PEMS-BAY) are excluded from MAE, RMSE and MAPE alike;
* results are reported per horizon (3 = 15 min, 6 = 30 min, 12 = 60 min) and
  pooled over all output steps ("average").

All functions accept numpy arrays or torch tensors shaped
``(samples, horizon, nodes)``.
"""

from collections.abc import Iterable

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _valid_mask(labels: np.ndarray, null_val: float | None) -> np.ndarray:
    mask = ~np.isnan(labels)
    if null_val is not None and not np.isnan(null_val):
        mask &= np.abs(labels - null_val) > 1e-6
    return mask


def masked_metrics(
    preds: ArrayLike,
    labels: ArrayLike,
    null_val: float | None = 0.0,
) -> dict[str, float]:
    """MAE, RMSE and MAPE (%) over all entries whose label is valid."""
    p = _to_numpy(preds).astype(np.float64)
    y = _to_numpy(labels).astype(np.float64)
    mask = _valid_mask(y, null_val)
    n = int(mask.sum())
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "count": 0}
    err = p[mask] - y[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.abs(y[mask])) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape, "count": n}


def horizon_metrics(
    preds: ArrayLike,
    labels: ArrayLike,
    horizons: Iterable[int] = (3, 6, 12),
    null_val: float | None = 0.0,
) -> dict[str, dict[str, float]]:
    """Per-horizon and pooled metrics.

    ``horizons`` are 1-based step indices (3 -> the 15-minute forecast on
    5-minute data). Returns ``{"h3": {...}, "h6": {...}, "h12": {...},
    "all": {...}, "per_step": [{...} x horizon]}``.
    """
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: preds {p.shape} vs labels {y.shape}")
    out: dict[str, dict[str, float]] = {}
    for h in horizons:
        if 1 <= h <= p.shape[1]:
            out[f"h{h}"] = masked_metrics(p[:, h - 1], y[:, h - 1], null_val)
    out["all"] = masked_metrics(p, y, null_val)
    out["per_step"] = [masked_metrics(p[:, i], y[:, i], null_val) for i in range(p.shape[1])]
    return out


def format_horizon_table(results: dict[str, dict[str, float]], title: str = "") -> str:
    """Render ``horizon_metrics`` output as a compact fixed-width table."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'horizon':>9} | {'MAE':>7} {'RMSE':>7} {'MAPE%':>7}")
    lines.append("-" * 36)
    for key in [k for k in results if k.startswith("h")] + ["all"]:
        if key not in results:
            continue
        r = results[key]
        label = "average" if key == "all" else f"{int(key[1:]) * 5} min"
        lines.append(f"{label:>9} | {r['mae']:7.3f} {r['rmse']:7.3f} {r['mape']:7.3f}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Differentiable losses (torch)
# -----------------------------------------------------------------------------

def masked_mae_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val: float | None = 0.0,
) -> torch.Tensor:
    """Mean absolute error over valid labels (the benchmark training loss)."""
    mask = ~torch.isnan(labels)
    if null_val is not None:
        mask &= (labels - null_val).abs() > 1e-6
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    labels = torch.nan_to_num(labels)
    return ((preds - labels).abs() * mask).sum() / denom


def masked_huber_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val: float | None = 0.0,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss over valid labels (used for flow datasets such as PEMS0X)."""
    mask = ~torch.isnan(labels)
    if null_val is not None:
        mask &= (labels - null_val).abs() > 1e-6
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    labels = torch.nan_to_num(labels)
    loss = torch.nn.functional.huber_loss(preds, labels, reduction="none", delta=delta)
    return (loss * mask).sum() / denom
