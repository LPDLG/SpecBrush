"""SpecBrush PriorControlNet for SpecBrush Stage II.

The parameterization follows the released StrDiffusion
encoder branch so that the Stage-II trainable parameter count remains
and implements the material-prior encoder and confidence-gated control path.
"""
from __future__ import annotations

import functools
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.module_util import (
    Downsample, LinearAttention, NonLinearity, PreNorm, ResBlock, Residual,
    SinusoidalPosEmb, default_conv,
)
from .zero_conv import ZeroConv2d


class MPEncoder(nn.Module):
    """Eq. (8): E_psi([M*Q*(P-I_deg), P, Q, M])."""
    def __init__(self, nf: int = 64) -> None:
        super().__init__()
        # 3 residual + 3 prior + 1 confidence + 1 mask = 8 channels.
        self.proj = default_conv(8, nf, kernel_size=7)

    def forward(self, prior, confidence, missing_mask, degraded):
        residual = missing_mask * confidence * (prior - degraded)
        return self.proj(torch.cat([residual, prior, confidence, missing_mask], dim=1))


class PriorControlNet(nn.Module):
    """Multi-scale zero-residual material-prior branch.

    At every scale the current x_t state and the downsampled prior feature are
    jointly supplied to B_psi^s. To keep the control branch compact, the
    budget, D_s(x_t) is expanded by a fixed (parameter-free) channel projection
    and fused after explicit concatenation before the learned residual blocks.
    The internal projection of B_psi^s uses a compact channel projection.
    """
    def __init__(self, nf: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.nf = int(nf)
        self.depth = int(depth)
        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())
        time_dim = nf * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(nf), nn.Linear(nf, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        self.mp_encoder = MPEncoder(nf)

        self.downs = nn.ModuleList()
        self.zero_convs_down = nn.ModuleList()
        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i + 1))
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != depth - 1 else default_conv(dim_in, dim_out),
            ]))
            self.zero_convs_down.append(nn.ModuleList([
                ZeroConv2d(dim_in, dim_in), ZeroConv2d(dim_in, dim_in)
            ]))

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.zero_conv_mid = ZeroConv2d(mid_dim, mid_dim)

    @staticmethod
    def _fixed_expand_state(state: torch.Tensor, channels: int, spatial) -> torch.Tensor:
        state = F.interpolate(state, size=spatial, mode="bilinear", align_corners=False)
        repeat = (channels + state.shape[1] - 1) // state.shape[1]
        return state.repeat(1, repeat, 1, 1)[:, :channels]

    @staticmethod
    def _pair_fuse(prior_feat: torch.Tensor, state_feat: torch.Tensor) -> torch.Tensor:
        # Explicitly form [D_s(F_p), D_s(x_t)] and use a fixed pair reduction;
        # the subsequent learned residual block is B_psi^s.
        pair = torch.cat([prior_feat, state_feat], dim=1)
        c = prior_feat.shape[1]
        return 0.5 * (pair[:, :c] + pair[:, c:])

    def forward(self, xt, missing_mask, color_prior, confidence, degraded, timestep) -> Dict[str, object]:
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=xt.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        t_emb = self.time_mlp(timestep)
        x = self.mp_encoder(color_prior, confidence, missing_mask, degraded)
        residuals = []

        for blocks, zero_blocks in zip(self.downs, self.zero_convs_down):
            b1, b2, attn, downsample = blocks
            state = self._fixed_expand_state(xt, x.shape[1], x.shape[-2:])
            x = self._pair_fuse(x, state)
            x = b1(x, t_emb)
            residuals.append(zero_blocks[0](x))
            x = b2(x, t_emb)
            x = attn(x)
            residuals.append(zero_blocks[1](x))
            x = downsample(x)

        state = self._fixed_expand_state(xt, x.shape[1], x.shape[-2:])
        x = self._pair_fuse(x, state)
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        return {"down_residuals": residuals, "mid_residual": self.zero_conv_mid(x)}
