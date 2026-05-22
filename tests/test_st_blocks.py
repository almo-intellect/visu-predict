from __future__ import annotations

import pytest
import torch

from visu_predict.models.st_blocks import SpatialBlock, STAttnStack, TemporalBlock


def test_temporal_block_preserves_shape():
    block = TemporalBlock(d_model=16, num_heads=4)
    x = torch.randn(2, 6, 5, 16)
    out = block(x)
    assert out.shape == x.shape


def test_spatial_block_preserves_shape():
    block = SpatialBlock(d_model=16, num_heads=4)
    x = torch.randn(2, 6, 5, 16)
    out = block(x)
    assert out.shape == x.shape


def test_spatial_block_d_model_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        SpatialBlock(d_model=10, num_heads=4)


def test_spatial_block_accepts_attn_bias():
    block = SpatialBlock(d_model=16, num_heads=4)
    x = torch.randn(2, 3, 5, 16)
    bias = torch.zeros(1, 4, 5, 5)
    out = block(x, attn_bias=bias)
    assert out.shape == x.shape
    # attention weights captured
    assert block.attn_weights is not None
    assert block.attn_weights.shape[-2:] == (5, 5)


def test_st_attn_stack_alternates_and_preserves_shape():
    stack = STAttnStack(d_model=16, num_heads=4, num_layers=2, interleave_order="TS")
    x = torch.randn(1, 4, 3, 16)
    out = stack(x)
    assert out.shape == x.shape
    assert len(stack.temporal_blocks) == 2
    assert len(stack.spatial_blocks) == 2


def test_st_attn_stack_st_order_works():
    stack = STAttnStack(d_model=16, num_heads=4, num_layers=2, interleave_order="ST")
    x = torch.randn(1, 4, 3, 16)
    out = stack(x)
    assert out.shape == x.shape


def test_st_attn_stack_rejects_invalid_order():
    with pytest.raises(ValueError, match="interleave_order"):
        STAttnStack(d_model=16, num_heads=4, num_layers=2, interleave_order="XX")


def test_st_attn_stack_gradient_flow():
    stack = STAttnStack(d_model=8, num_heads=2, num_layers=1)
    x = torch.randn(1, 3, 2, 8, requires_grad=True)
    out = stack(x)
    out.sum().backward()
    assert x.grad is not None
    assert (x.grad != 0).any()
