"""MuCleaner: conditional-mean purification for SpecBrush Stage II.

Implements the MuCleaner conditional-mean purification path:
  mu_c = mu + sigmoid(G_phi([M,dM,|grad mu|])) *
               H_phi(T_phi^N(E_phi([mu*(1-M),1-M,M,dM])))
MuCleaner is optimized end-to-end by the Stage-II diffusion objective. No
self-supervised denoising, TV, perceptual, edge, or structural auxiliary loss is used.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pureformer_blocks import TransformerBlock


class MuCleaner(nn.Module):
    def __init__(
        self,
        dim: int = 32,
        num_blocks: int = 2,
        num_heads: int = 4,
        boundary_width: int = 3,
    ) -> None:
        super().__init__()
        self.boundary_width = int(boundary_width)
        # [mu * known (3), known (1), missing (1), boundary (1)] = 6.
        self.encoder = nn.Conv2d(6, dim, 3, 1, 1)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_expansion_factor=2.0,
                    attn_dilations=(1, 2),
                    ffn_dilations=(1,),
                    gate="relu_sigmoid",
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, 3, 3, 1, 1),
        )
        # [M, boundary, |grad mu|] -> RGB gate.
        self.gate = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, 3, 1),
        )

    def _boundary(self, missing: torch.Tensor) -> torch.Tensor:
        if self.boundary_width <= 0:
            return torch.zeros_like(missing)
        k = 2 * self.boundary_width + 1
        dilated = F.max_pool2d(missing, k, 1, self.boundary_width)
        eroded = 1.0 - F.max_pool2d(1.0 - missing, k, 1, self.boundary_width)
        return (dilated - eroded).clamp(0.0, 1.0)

    @staticmethod
    def _gradient_magnitude(mu: torch.Tensor) -> torch.Tensor:
        # Use channel-mean RGB gradient as the spatial gradient cue.
        y = mu.mean(dim=1, keepdim=True)
        gx = F.pad(y[:, :, :, 1:] - y[:, :, :, :-1], (0, 1, 0, 0))
        gy = F.pad(y[:, :, 1:, :] - y[:, :, :-1, :], (0, 0, 0, 1))
        return torch.sqrt(gx.square() + gy.square() + 1e-12)

    def forward(self, mu: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        if mu.dim() != 4 or mu.shape[1] != 3:
            raise ValueError(f"mu must be [B,3,H,W], got {tuple(mu.shape)}")
        if missing_mask.dim() != 4 or missing_mask.shape[1] != 1:
            raise ValueError("missing_mask must be [B,1,H,W] with 1=missing")
        missing = missing_mask.to(dtype=mu.dtype).clamp(0.0, 1.0)
        known = 1.0 - missing
        boundary = self._boundary(missing)
        grad = self._gradient_magnitude(mu)

        feat = self.encoder(torch.cat([mu * known, known, missing, boundary], dim=1))
        feat = self.blocks(feat)
        residual = self.residual_head(feat)
        gate = torch.sigmoid(self.gate(torch.cat([missing, boundary, grad], dim=1)))
        return (mu + gate * residual).clamp(0.0, 1.0)




