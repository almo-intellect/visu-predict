"""PatchTST-style temporal patch embedding for ``[B, T, N, d]`` tensors.

Splits the time axis into non-overlapping (or strided) patches of length ``P``
and projects each ``P * d`` flat patch back to ``d`` dimensions. This reduces
the effective sequence length passed to attention by a factor of ``P``,
giving each token richer information about a local temporal window. Inspired
by [PatchTST (ICLR 2023)](https://arxiv.org/abs/2211.14730).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split the time axis into patches and project each patch to ``d_model``.

    Args:
        d_model: per-cell channel dimension.
        patch_length: number of consecutive timesteps per patch.
        patch_stride: stride between patches. ``None`` means non-overlapping
            (stride == patch_length).
        d_out: output channel dimension (defaults to ``d_model`` to preserve
            STAE channel budgeting).
    """

    def __init__(
        self,
        d_model: int,
        patch_length: int,
        patch_stride: int | None = None,
        d_out: int | None = None,
    ) -> None:
        super().__init__()
        if patch_length <= 0:
            raise ValueError(f"patch_length must be positive, got {patch_length}")
        self.patch_length = patch_length
        self.patch_stride = patch_stride or patch_length
        self.d_model = d_model
        self.d_out = d_out or d_model
        self.proj = nn.Linear(patch_length * d_model, self.d_out)

    def num_patches(self, seq_length: int) -> int:
        return max(0, 1 + (seq_length - self.patch_length) // self.patch_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: tensor of shape ``[B, T, N, d_model]``.

        Returns:
            tensor of shape ``[B, num_patches, N, d_out]``.
        """
        bsz, seq_len, num_sensors, d = x.shape
        if d != self.d_model:
            raise ValueError(
                f"Input channel dim {d} doesn't match d_model {self.d_model}"
            )
        num_patches = self.num_patches(seq_len)
        if num_patches <= 0:
            raise ValueError(
                f"seq_length {seq_len} too short for patch_length {self.patch_length} "
                f"with stride {self.patch_stride}"
            )

        # [B, T, N, d] -> [B, N, T, d]
        x_perm = x.permute(0, 2, 1, 3)
        # unfold over the time axis: [B, N, num_patches, patch_length, d]
        x_patches = x_perm.unfold(dimension=2, size=self.patch_length, step=self.patch_stride)
        # Permute then merge patch_length*d for the linear projection:
        # unfold yields [B, N, num_patches, d, patch_length]; rearrange to
        # [B, N, num_patches, patch_length, d] then flatten the last two dims.
        x_patches = x_patches.permute(0, 1, 2, 4, 3).contiguous()
        x_flat = x_patches.view(bsz, num_sensors, num_patches, self.patch_length * d)
        projected = self.proj(x_flat)  # [B, N, num_patches, d_out]
        # back to [B, num_patches, N, d_out]
        return projected.permute(0, 2, 1, 3).contiguous()
