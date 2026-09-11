"""SpecBrush Mask-Gated Locality-Continuity (MGLC) block.

Implements MGLC with local/context branches, a two-way
softmax gate from [M_s, dM_s, D_s(Q), mean(H_s^p)], a confidence-aware
amplitude gate, boundary weighting only on the local branch, and a
zero-initialized output mapping.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)


class LocalContinuityBranch(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        return self.pw(self.dw5(F.gelu(self.dw3(x))))


class ContextExtensionBranch(nn.Module):
    """Large-range context path using DW/PW operations and channel reweighting."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        squeeze = max(channels // 8, 16)
        self.in_proj = nn.Conv2d(channels, channels, 1)
        self.dw_local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dw_h = nn.Conv2d(channels, channels, (1,9), padding=(0,4), groups=channels)
        self.dw_v = nn.Conv2d(channels, channels, (9,1), padding=(4,0), groups=channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeeze, 1),
            nn.GELU(),
            nn.Conv2d(squeeze, channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = F.gelu(self.in_proj(x))
        x = self.dw_local(x) + self.dw_h(x) + self.dw_v(x)
        x = F.gelu(x)
        return x * self.se(x)


class MGLCBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        gate_hidden: int = 16,
        boundary_width: int = 3,
        lambda_b: float = 1.0,
        branch_mode: str = "both",
        **_: object,
    ) -> None:
        super().__init__()
        if branch_mode not in {"both", "local_only", "context_only"}:
            raise ValueError(f"unsupported branch_mode: {branch_mode}")
        self.boundary_width = int(boundary_width)
        self.lambda_b = float(lambda_b)
        self.norm = LayerNorm2d(channels)
        self.local_branch = None if branch_mode == "context_only" else LocalContinuityBranch(channels)
        self.context_branch = None if branch_mode == "local_only" else ContextExtensionBranch(channels)
        self.gate_net = nn.Sequential(
            nn.Conv2d(4, gate_hidden, 3, padding=1), nn.GELU(), nn.Conv2d(gate_hidden, 2, 1)
        )
        self.amp_net = nn.Sequential(
            nn.Conv2d(4, gate_hidden, 3, padding=1), nn.GELU(), nn.Conv2d(gate_hidden, 1, 1)
        )
        # Zero initialization prevents the new branch
        # from perturbing the pretrained diffusion representation at iteration 0.
        self.zero_out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.zero_out.weight)
        if self.zero_out.bias is not None:
            nn.init.zeros_(self.zero_out.bias)

    def _resize(self, x: torch.Tensor, size: Tuple[int,int]) -> torch.Tensor:
        return F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False)

    def _boundary(self, mask: torch.Tensor) -> torch.Tensor:
        if self.boundary_width <= 0:
            return torch.zeros_like(mask)
        k = 2 * self.boundary_width + 1
        dilated = F.max_pool2d(mask, k, 1, self.boundary_width)
        eroded = 1.0 - F.max_pool2d(1.0-mask, k, 1, self.boundary_width)
        return (dilated-eroded).clamp(0.0,1.0)

    def forward(self, feat: torch.Tensor, missing_mask: Optional[torch.Tensor], confidence: Optional[torch.Tensor]=None):
        if missing_mask is None:
            return feat
        size = feat.shape[-2:]
        m = self._resize(missing_mask, size).clamp(0.0,1.0)
        dm = self._boundary(m)
        q = torch.ones_like(m) if confidence is None else self._resize(confidence, size).clamp(0.0,1.0)
        hbar = feat.mean(dim=1, keepdim=True)
        gate_input = torch.cat([m, dm, q, hbar], dim=1)
        g = torch.softmax(self.gate_net(gate_input), dim=1)
        a = torch.sigmoid(self.amp_net(gate_input))

        h = self.norm(feat)
        fused = torch.zeros_like(feat)
        if self.local_branch is not None:
            fused = fused + g[:,0:1] * self.local_branch(h) * (1.0 + self.lambda_b * dm)
        if self.context_branch is not None:
            fused = fused + g[:,1:2] * self.context_branch(h)
        return feat + self.zero_out(a * fused)
