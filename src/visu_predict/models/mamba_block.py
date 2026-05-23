"""Optional Mamba / state-space temporal block.

Drop-in replacement for :class:`visu_predict.models.st_blocks.TemporalBlock`
that runs `Mamba <https://github.com/state-spaces/mamba>`_ along the time
axis. ``mamba_ssm`` is an optional, CUDA-only dependency; the import is
guarded so the rest of the package still imports cleanly when it isn't
installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None  # type: ignore[assignment]
    MAMBA_AVAILABLE = False


class MambaTemporalBlock(nn.Module):
    """Temporal mixing via a single ``Mamba`` block.

    Same interface as :class:`TemporalBlock`: input ``[B, T, N, d]``,
    output ``[B, T, N, d]``. Internally reshapes to ``[B*N, T, d]`` so each
    sensor's sequence is processed independently.

    Attributes:
        attn_weights: always ``None``; Mamba is not attention. Present for
            API parity with the attention-based blocks so callers can
            uniformly inspect ``block.attn_weights``.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is not installed. Install with `pip install visu-predict[mamba]` "
                "(requires CUDA)."
            )
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, num_sensors, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz * num_sensors, seq_len, d)
        x = self.mamba(x)
        return x.view(bsz, num_sensors, seq_len, d).permute(0, 2, 1, 3).contiguous()
