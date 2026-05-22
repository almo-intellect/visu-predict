"""GPU memory inspection helpers."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def gpu_memory_info() -> dict[str, Any]:
    """Per-device allocated/reserved/total memory (GB) and utilisation pct."""
    if not torch.cuda.is_available():
        return {}

    info: dict[str, Any] = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(i)
        allocated = torch.cuda.memory_allocated(i)
        info[f"gpu_{i}"] = {
            "name": props.name,
            "total_gb": total / 1e9,
            "reserved_gb": reserved / 1e9,
            "allocated_gb": allocated / 1e9,
            "free_gb": (total - reserved) / 1e9,
            "utilization_pct": 100.0 * allocated / total,
        }
    return info


def log_gpu_memory() -> None:
    """Emit a single INFO-level log line summarising GPU memory."""
    info = gpu_memory_info()
    if not info:
        return
    for gpu_id, stats in info.items():
        logger.info(
            "%s (%s): %.2f / %.2f GB allocated (%.1f%%)",
            gpu_id, stats["name"], stats["allocated_gb"], stats["total_gb"],
            stats["utilization_pct"],
        )


def best_available_device() -> torch.device:
    """Return a torch.device — cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
