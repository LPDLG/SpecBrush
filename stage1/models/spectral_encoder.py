"""Material-condition encoder used by Stage I of SpecBrush.

The complete material condition combines:
    z_m = Phi_m(Concat(E_c(c_deg), E_r(r), E_x(x_xrd)))
where E_c is the color encoder and E_r/E_x are Mamba spectral encoders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .mamba_wrappers import SequenceMambaBlock


class MambaSpectralEncoder(nn.Module):
    def __init__(
        self,
        spec_len: int,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.spec_len = int(spec_len)
        self.d_model = int(d_model)
        self.pooling = pooling
        self.in_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [SequenceMambaBlock(d_model=d_model, dropout=dropout) for _ in range(int(n_layers))]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.ndim != 2:
            raise ValueError(f"spec must be (B,L), got {tuple(spec.shape)}")
        x = self.in_proj(spec.unsqueeze(-1))
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x.mean(dim=1) if self.pooling == "mean" else x[:, -1, :]


@dataclass
class ConditionerConfig:
    use_raman: bool = True
    use_xrd: bool = True
    raman_len: int = 1024
    xrd_len: int = 2048
    d_model: int = 128
    n_layers: int = 4
    dropout: float = 0.0
    raman_peak_dim: int = 0
    xrd_peak_dim: int = 0
    material_dim: int = 128


class MultimodalConditioner(nn.Module):
    """Build the complete color+Raman+XRD material condition z_m."""

    def __init__(self, cfg: ConditionerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.raman_enc = (
            MambaSpectralEncoder(cfg.raman_len, cfg.d_model, cfg.n_layers, cfg.dropout)
            if cfg.use_raman else None
        )
        self.xrd_enc = (
            MambaSpectralEncoder(cfg.xrd_len, cfg.d_model, cfg.n_layers, cfg.dropout)
            if cfg.use_xrd else None
        )
        self.raman_peak_proj = (
            nn.Sequential(nn.Linear(cfg.raman_peak_dim, cfg.d_model), nn.SiLU(), nn.LayerNorm(cfg.d_model))
            if cfg.use_raman and int(cfg.raman_peak_dim) > 0 else None
        )
        self.xrd_peak_proj = (
            nn.Sequential(nn.Linear(cfg.xrd_peak_dim, cfg.d_model), nn.SiLU(), nn.LayerNorm(cfg.d_model))
            if cfg.use_xrd and int(cfg.xrd_peak_dim) > 0 else None
        )

        input_dim = cfg.d_model  # E_c(c_deg)
        if self.raman_enc is not None:
            input_dim += cfg.d_model
        if self.xrd_enc is not None:
            input_dim += cfg.d_model
        self.cond_dim = int(cfg.material_dim)
        self.fuse = nn.Sequential(
            nn.Linear(input_dim, self.cond_dim),
            nn.SiLU(),
            nn.LayerNorm(self.cond_dim),
        )

    def forward(
        self,
        z_color: torch.Tensor,
        raman: Optional[torch.Tensor],
        xrd: Optional[torch.Tensor],
        raman_peaks: Optional[torch.Tensor] = None,
        xrd_peaks: Optional[torch.Tensor] = None,
        return_embeds: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if z_color.ndim != 2 or z_color.shape[-1] != self.cfg.d_model:
            raise ValueError(
                f"z_color must be (B,{self.cfg.d_model}), got {tuple(z_color.shape)}"
            )

        embeds: Dict[str, torch.Tensor] = {"color": z_color}
        feats = [z_color]

        if self.raman_enc is not None:
            if raman is None:
                raise ValueError("Raman is required for the complete material condition")
            z_r = self.raman_enc(raman)
            if self.raman_peak_proj is not None and raman_peaks is not None:
                z_r = z_r + self.raman_peak_proj(raman_peaks)
            embeds["raman"] = z_r
            feats.append(z_r)

        if self.xrd_enc is not None:
            if xrd is None:
                raise ValueError("XRD is required for the complete material condition")
            z_x = self.xrd_enc(xrd)
            if self.xrd_peak_proj is not None and xrd_peaks is not None:
                z_x = z_x + self.xrd_peak_proj(xrd_peaks)
            embeds["xrd"] = z_x
            feats.append(z_x)

        z_m = self.fuse(torch.cat(feats, dim=-1))
        embeds["material"] = z_m
        return (z_m, embeds) if return_embeds else z_m
