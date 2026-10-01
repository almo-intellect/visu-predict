"""
Foundation utility module for the traffic prediction transformer project.

Provides feature-flag detection, reproducibility helpers, GPU utilities,
logging infrastructure, and result-persistence functions.
"""

import gc
import os
import random
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Feature-flag detection (module-level try/except imports)
# ---------------------------------------------------------------------------

# AMP / autocast support
# NOTE (patch): the previous probe instantiated torch.cuda.amp.autocast with a
# ``device_type`` keyword, which that class never accepted, so
# DEVICE_TYPE_SUPPORTED was always False and the deprecated torch.cuda.amp
# code paths were taken on every PyTorch version. We now detect the modern
# ``torch.amp`` API directly and expose make_autocast()/make_grad_scaler()
# helpers so call sites never touch deprecated APIs.
AMP_AVAILABLE = False
DEVICE_TYPE_SUPPORTED = False  # True when the modern torch.amp API exists
try:
    from torch.amp import autocast as _modern_autocast  # noqa: F401

    AMP_AVAILABLE = True
    DEVICE_TYPE_SUPPORTED = True
except ImportError:
    try:
        from torch.cuda.amp import autocast as _legacy_autocast  # noqa: F401

        AMP_AVAILABLE = True
    except ImportError:
        pass


def make_autocast(device: str = "cuda", enabled: bool = True):
    """Return the correct autocast context manager for this torch version.

    Uses ``torch.amp.autocast(device_type=...)`` when available (PyTorch
    >= 1.10 / 2.x), falling back to the legacy ``torch.cuda.amp.autocast``
    only on very old installs. Returns a no-op context when *enabled* is
    False or AMP is unavailable.
    """
    from contextlib import nullcontext

    if not enabled or not AMP_AVAILABLE:
        return nullcontext()
    device_type = str(device).split(":")[0]
    if device_type == "cpu":
        return nullcontext()
    if DEVICE_TYPE_SUPPORTED:
        return torch.amp.autocast(device_type=device_type)
    from torch.cuda.amp import autocast as _ac  # legacy fallback
    return _ac()


def make_grad_scaler(device: str = "cuda", enabled: bool = True):
    """Return a gradient scaler using the modern ``torch.amp.GradScaler``.

    Falls back to the legacy ``torch.cuda.amp.GradScaler`` on old installs.
    """
    device_type = str(device).split(":")[0]
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, enabled=enabled)
    from torch.cuda.amp import GradScaler as _Scaler  # legacy fallback
    return _Scaler(enabled=enabled)

# statsmodels
STATSMODELS_AVAILABLE = False
try:
    import statsmodels  # noqa: F401

    STATSMODELS_AVAILABLE = True
except ImportError:
    pass

# optuna
OPTUNA_AVAILABLE = False
try:
    import optuna  # noqa: F401

    OPTUNA_AVAILABLE = True
except ImportError:
    pass

# torch_geometric
TORCH_GEOMETRIC_AVAILABLE = False
try:
    from torch_geometric.nn import GATConv, GCNConv  # noqa: F401

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    pass

# Google Colab detection
IN_COLAB = False
try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Timezone constant
# ---------------------------------------------------------------------------
MAPUTO_TZ = pytz.timezone("Africa/Maputo")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_reproducibility_seed(seed: int = 42) -> None:
    """Set seeds across all RNG sources for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------
def generate_timestamp() -> str:
    """Return the current timestamp in Maputo timezone (``%Y%m%d_%H%M%S``)."""
    return datetime.now(MAPUTO_TZ).strftime("%Y%m%d_%H%M%S")


def get_maputo_timestamp(dataset_name: Optional[str] = None) -> str:
    """Return a Maputo-timezone timestamp, optionally prefixed with *dataset_name*.

    Parameters
    ----------
    dataset_name : str, optional
        If supplied the returned string is ``<dataset_name>_<timestamp>``;
        otherwise just ``<timestamp>``.
    """
    ts = generate_timestamp()
    if dataset_name is not None:
        return f"{dataset_name}_{ts}"
    return ts


# ---------------------------------------------------------------------------
# GPU utility functions
# ---------------------------------------------------------------------------
def get_gpu_memory_info() -> Dict[str, Any]:
    """Query per-device GPU memory stats.  Returns a dict keyed by ``gpu_<i>``."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    try:
        device_count = torch.cuda.device_count()
        gpu_info: Dict[str, Any] = {}
        for i in range(device_count):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            reserved_memory = torch.cuda.memory_reserved(i)
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = total_memory - reserved_memory
            gpu_info[f"gpu_{i}"] = {
                "total_memory_GB": total_memory / 1e9,
                "reserved_memory_GB": reserved_memory / 1e9,
                "allocated_memory_GB": allocated_memory / 1e9,
                "free_memory_GB": free_memory / 1e9,
                "utilization_pct": (allocated_memory / total_memory) * 100,
            }
        return gpu_info
    except Exception as e:
        return {"error": str(e)}


def find_optimal_batch_size(
    model: nn.Module,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    max_batch_size: int = 2048,
    start_batch: int = 32,
    device: str = "cuda",
) -> int:
    """Binary-search-style probe for the largest batch size that fits in GPU memory.

    Returns a safe batch size (80 % of the largest successful size, floored to
    *start_batch*).
    """
    if device == "cpu" or not torch.cuda.is_available():
        return 64

    model = model.to(device)
    optimal_batch_size = start_batch
    torch.cuda.empty_cache()
    gc.collect()

    print("Finding optimal batch size for GPU...")
    try:
        sample_input = sample_input.to(device)
        sample_target = sample_target.to(device)

        for batch_size in [
            2 ** i
            for i in range(
                int(np.log2(start_batch)), int(np.log2(max_batch_size)) + 1
            )
        ]:
            try:
                input_batch = sample_input.repeat(batch_size, 1, 1)
                target_batch = sample_target.repeat(batch_size, 1, 1)

                with make_autocast(device, enabled=True):
                    output = model(input_batch)
                    loss = nn.MSELoss()(output, target_batch)

                loss.backward()
                optimal_batch_size = batch_size

                # PATCH: gradients accumulated across probes previously; clear
                # them so each probe measures a clean backward pass.
                model.zero_grad(set_to_none=True)

                del input_batch, target_batch, output, loss
                torch.cuda.empty_cache()
                print(f"  Successfully tested batch size: {batch_size}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at batch size: {batch_size}")
                    break
                else:
                    raise
    except Exception as e:
        print(f"Error while finding optimal batch size: {e}")
        return 64
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()

    return max(int(optimal_batch_size * 0.8), start_batch)


def start_gpu_memory_monitor(
    config: Any, interval: int = 10
) -> List[bool]:
    """Launch a daemon thread that periodically logs GPU memory usage.

    Parameters
    ----------
    config :
        An object with at least a ``device`` attribute (str).
    interval : int
        Seconds between each log line.

    Returns
    -------
    list[bool]
        A one-element list used as a stop flag.  Pass it to
        :func:`stop_gpu_memory_monitor` to terminate the thread.
    """
    stop_flag: List[bool] = [False]

    def _monitor() -> None:
        while not stop_flag[0]:
            if torch.cuda.is_available() and getattr(config, "device", "cpu") != "cpu":
                info = get_gpu_memory_info()
                for gpu_name, stats in info.items():
                    if isinstance(stats, dict) and "utilization_pct" in stats:
                        print(
                            f"[GPU Monitor] {gpu_name}: "
                            f"allocated={stats['allocated_memory_GB']:.2f} GB, "
                            f"free={stats['free_memory_GB']:.2f} GB, "
                            f"utilization={stats['utilization_pct']:.1f}%"
                        )
            time.sleep(interval)

    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    return stop_flag


def stop_gpu_memory_monitor(stop_flag: List[bool]) -> None:
    """Signal the GPU-memory monitor thread to stop."""
    if stop_flag is not None:
        stop_flag[0] = True


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class TeeLogger:
    """Duplicate *stdout* so that output goes to both the terminal and a log file."""

    def __init__(self, filename: str) -> None:
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        """Delegate to the real terminal (tqdm and friends call this)."""
        return getattr(self.terminal, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self.terminal.fileno()

    def close(self) -> None:
        try:
            self.log.close()
        except Exception:
            pass


def setup_logging(base_output_dir: str) -> str:
    """Create a log directory under *base_output_dir* and redirect stdout via :class:`TeeLogger`.

    Returns
    -------
    str
        Absolute path to the created log file.
    """
    log_dir = os.path.join(base_output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = f"experiment_log_{generate_timestamp()}.txt"
    log_path = os.path.join(log_dir, log_filename)

    sys.stdout = TeeLogger(log_path)
    print(f"Logging to: {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def save_predictions_and_actuals(
    predictions: np.ndarray,
    actuals: np.ndarray,
    results_dir: str,
    filename: str,
    fold: Optional[int] = None,
) -> None:
    """Save *predictions* and *actuals* arrays side-by-side in a CSV file.

    Parameters
    ----------
    predictions, actuals :
        1-D or 2-D arrays of the same length.
    results_dir : str
        Directory in which to store the CSV.
    filename : str
        Base filename (a timestamp and fold info are appended).
    fold : int, optional
        Cross-validation fold index.
    """
    os.makedirs(results_dir, exist_ok=True)

    ts = generate_timestamp()
    fold_tag = f"_fold{fold}" if fold is not None else ""
    full_filename = f"{filename}{fold_tag}_{ts}.csv"
    filepath = os.path.join(results_dir, full_filename)

    predictions_flat = np.array(predictions).flatten()
    actuals_flat = np.array(actuals).flatten()

    df = pd.DataFrame(
        {"predictions": predictions_flat, "actuals": actuals_flat}
    )
    df.to_csv(filepath, index=False)
    print(f"Predictions and actuals saved to: {filepath}")


def save_experiment_results(
    config: Any,
    model_params: Dict[str, Any],
    metrics: Tuple[float, float, float, float],
    results_dir: str,
    fold: Optional[int] = None,
) -> None:
    """Persist experiment configuration and evaluation metrics to a CSV row.

    Parameters
    ----------
    config :
        A dataclass instance whose fields are serialised via ``dataclasses.asdict``.
    model_params : dict
        Arbitrary model hyper-parameters to record.
    metrics : tuple of four floats
        ``(mae, rmse, r2, mape)`` in that order.
    results_dir : str
        Directory in which to store the CSV.
    fold : int, optional
        Cross-validation fold index.
    """
    os.makedirs(results_dir, exist_ok=True)

    mae, rmse, r2, mape = metrics

    ts = generate_timestamp()
    fold_tag = f"_fold{fold}" if fold is not None else ""
    full_filename = f"experiment_results{fold_tag}_{ts}.csv"
    filepath = os.path.join(results_dir, full_filename)

    row: Dict[str, Any] = {}

    # Serialise the config dataclass
    try:
        config_dict = asdict(config)
    except Exception:
        config_dict = vars(config) if hasattr(config, "__dict__") else {}
    row.update({f"config_{k}": v for k, v in config_dict.items()})

    # Model parameters
    row.update({f"param_{k}": v for k, v in model_params.items()})

    # Metrics
    row["mae"] = mae
    row["rmse"] = rmse
    row["r2"] = r2
    row["mape"] = mape

    # Meta
    row["timestamp"] = ts
    if fold is not None:
        row["fold"] = fold

    df = pd.DataFrame([row])
    df.to_csv(filepath, index=False)
    print(f"Experiment results saved to: {filepath}")
