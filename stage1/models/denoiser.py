
"""
Denoising network for conditional diffusion on short Lab sequences.

Input:
  x_t: (B, L, C)   (noised series)
  mask: (B, L, C)  (1=observed, 0=missing)
  t: (B,)          diffusion step
  cond: (B, D_cond) optional condition from Raman/XRD encoder

Output:
  eps_hat: (B, L, C) predicted noise
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_wrappers import SequenceMambaBlock


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) long/int
        returns: (B, dim)
        """
        if t.ndim != 1:
            t = t.view(-1)
        device = t.device
        half = self.dim // 2
        # exp range
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000.0) / max(half - 1, 1)))
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


@dataclass
class DenoiserConfig:
    in_channels: int = 3
    hidden_dim: int = 128
    n_layers: int = 4
    dropout: float = 0.0
    cond_dim: int = 0  # 0 means no conditioning


class MambaDenoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels
        self.hidden_dim = cfg.hidden_dim

        # Concatenate x_t and mask along channel dimension
        self.input_proj = nn.Linear(cfg.in_channels * 2, cfg.hidden_dim)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.cond_proj = None
        if cfg.cond_dim and cfg.cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(cfg.cond_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )

        self.blocks = nn.ModuleList(
            [SequenceMambaBlock(d_model=cfg.hidden_dim, dropout=cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.in_channels)

    def forward(
        self,
        x_t: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_t, mask: (B,L,C)
        t: (B,) long
        cond: (B,cond_dim) or None
        """
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be (B,L,C), got {tuple(x_t.shape)}")
        if mask.shape != x_t.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} != x_t shape {tuple(x_t.shape)}")

        h = torch.cat([x_t, mask], dim=-1)  # (B,L,2C)
        h = self.input_proj(h)              # (B,L,D)

        te = self.time_embed(t).unsqueeze(1)  # (B,1,D)
        h = h + te

        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("cond_dim>0 but cond is None")
            h = h + self.cond_proj(cond).unsqueeze(1)

        for blk in self.blocks:
            h = blk(h)

        h = self.out_norm(h)
        return self.output_proj(h)
