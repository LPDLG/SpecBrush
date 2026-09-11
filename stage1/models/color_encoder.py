"""
Color encoder: small MLP that maps Lab (or RGB converted to Lab) into an embedding space.

We keep it lightweight because your paired fading dataset is small (9 materials × ~45 time points × 2 humidity experiments),
but we still want a learnable mapping so we can:
- align color embeddings with Raman embeddings (contrastive pretraining on your 40+ standard Raman library),
- predict missing modality embeddings from RGB/Lab (missing-modality inference).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ColorEncoderConfig:
    in_dim: int = 3          # Lab has 3 dims
    d_model: int = 128       # embedding dim (should match Raman encoder d_model for contrastive alignment)
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.0


class ColorEncoder(nn.Module):
    def __init__(self, cfg: ColorEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers = []
        dim = cfg.in_dim
        for i in range(int(cfg.n_layers) - 1):
            layers.append(nn.Linear(dim, cfg.hidden_dim))
            layers.append(nn.SiLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            dim = cfg.hidden_dim
        layers.append(nn.Linear(dim, cfg.d_model))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3) normalized Lab
        returns: (B,d_model)
        """
        if x.dim() != 2 or x.size(-1) != self.cfg.in_dim:
            raise ValueError(f"ColorEncoder expects (B,{self.cfg.in_dim}), got {tuple(x.shape)}")
        z = self.net(x)
        z = self.norm(z)
        return z
