"""RGB-only missing-modality condition predictor for SpecBrush Stage I.

The predictor maps degraded-color features to the material-condition space. It predicts
an embedding in the same material-condition space as z_m; it does not
reconstruct raw Raman/XRD spectra and does not use prototype retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CondPredictorConfig:
    in_dim: int = 128
    out_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.0


def _make_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers = []
    dim = int(in_dim)
    for _ in range(max(0, int(n_layers) - 1)):
        layers.extend([nn.Linear(dim, hidden_dim), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = int(hidden_dim)
    layers.append(nn.Linear(dim, int(out_dim)))
    return nn.Sequential(*layers)


class RGBConditionPredictor(nn.Module):
    """Predict the missing-modality material condition from degraded color."""

    def __init__(self, cfg: CondPredictorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = _make_mlp(
            cfg.in_dim, cfg.out_dim, cfg.hidden_dim, cfg.n_layers, cfg.dropout
        )

    def forward(self, z_color: torch.Tensor) -> torch.Tensor:
        if z_color.ndim != 2 or z_color.shape[-1] != self.cfg.in_dim:
            raise ValueError(
                f"RGBConditionPredictor expects (B,{self.cfg.in_dim}), got {tuple(z_color.shape)}"
            )
        return self.net(z_color)


# Backward-compatible import name for external scripts. The semantics are the
# Direct material-condition predictor, not a spectrum predictor.
ColorToSpecPredictor = RGBConditionPredictor
