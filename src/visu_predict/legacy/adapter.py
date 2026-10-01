"""Run the V18 ``TrafficTransformer`` inside the V19 benchmark harness."""

from typing import Optional

import torch
import torch.nn as nn


class LegacyTransformerAdapter(nn.Module):
    """Runs the V18 ``TrafficTransformer`` inside the benchmark harness.

    Builds the legacy "sensors as channels" input ``(B, T, N + 2)`` from the
    scaled traffic plus time-of-day / day-of-week fractions, and returns
    ``(B, out, N)`` in the scaled space like :class:`visu_predict.STTransformer`.
    """

    def __init__(self, num_nodes: int, steps_per_day: int = 288, out_steps: int = 12,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 336, dropout: float = 0.05,
                 activation: str = "gelu") -> None:
        super().__init__()
        from .model import TrafficTransformer

        self.config = {"num_nodes": num_nodes, "steps_per_day": steps_per_day, "out_steps": out_steps,
                       "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                       "dim_feedforward": dim_feedforward, "dropout": dropout, "activation": activation}
        self.steps_per_day = steps_per_day
        self.net = TrafficTransformer(
            input_dim=num_nodes + 2, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, pred_length=out_steps, output_dim=num_nodes,
            num_sensors=num_nodes, decoder_type="linear", activation=activation,
        )

    def forward(self, x: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor,
                exo: Optional[torch.Tensor] = None) -> torch.Tensor:
        tod_f = (tod.float() / max(1, self.steps_per_day - 1))[..., None]
        dow_f = (dow.float().clamp(max=6) / 6.0)[..., None]
        feats = torch.cat([x[..., 0], tod_f, dow_f], dim=-1)    # (B, T, N + 2)
        return self.net(feats)
