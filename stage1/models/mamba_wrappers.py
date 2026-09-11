
"""
Light wrappers so the code can run even if `mamba_ssm` is not installed.

In your SSD-TS environment, you SHOULD have mamba_ssm==2.2.2 installed.
If not found, we fall back to GRU to keep the pipeline runnable (debug only).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def has_mamba() -> bool:
    try:
        import mamba_ssm  # noqa: F401
        return True
    except Exception:
        return False


class SequenceMambaBlock(nn.Module):
    """
    A block that processes (B, L, D) sequence.
    Uses Mamba if available, otherwise GRU fallback.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_mamba = has_mamba()
        self.dropout = nn.Dropout(dropout)

        if self.use_mamba:
            try:
                from mamba_ssm import Mamba  # type: ignore
                self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            except Exception:
                # alternate import path for some versions
                from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
                self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.norm = nn.LayerNorm(d_model)
            self.gru = None
        else:
            self.mamba = None
            self.norm = nn.LayerNorm(d_model)
            self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.use_mamba and self.mamba is not None:
            y = self.mamba(x)
        else:
            assert self.gru is not None
            y, _ = self.gru(x)
        y = self.dropout(y)
        return self.norm(x + y)
