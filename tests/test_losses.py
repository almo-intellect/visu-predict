from __future__ import annotations

import numpy as np
import torch

from visu_predict.training.losses import hybrid_loss, make_criterion, quantile_loss, robust_mape


def test_robust_mape_handles_zero_targets():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert np.isnan(robust_mape(y_true, y_pred))


def test_robust_mape_known_value():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 90.0])
    assert abs(robust_mape(y_true, y_pred) - 10.0) < 0.01


def test_hybrid_loss_between_mae_and_mse():
    output = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.5, 1.5]])
    mse = torch.nn.functional.mse_loss(output, target).item()
    mae = torch.nn.functional.l1_loss(output, target).item()
    blended = hybrid_loss(output, target, alpha=0.5).item()
    assert min(mse, mae) <= blended <= max(mse, mae) + 1e-6


def test_quantile_loss_positive():
    output = torch.randn(4, 3, 3)
    target = torch.randn(4, 3)
    loss = quantile_loss(output, target, [0.1, 0.5, 0.9])
    assert loss.item() >= 0


def test_make_criterion_returns_modules():
    assert isinstance(make_criterion("mae"), torch.nn.L1Loss)
    assert isinstance(make_criterion("mse"), torch.nn.MSELoss)
    assert isinstance(make_criterion("huber"), torch.nn.SmoothL1Loss)
