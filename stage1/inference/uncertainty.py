"""Posterior-sampling uncertainty helpers for Stage-I diffusion inversion."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from utils.color_utils import LabNorm
from training.diffusion import p_sample_loop


@torch.no_grad()
def sample_with_confidence(denoiser, schedule, x0_norm: torch.Tensor, mask: torch.Tensor, cond, num_samples: int = 20) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    device = x0_norm.device
    lab_norm = LabNorm()
    x_obs = x0_norm * mask
    samples = []
    for _ in range(int(num_samples)):
        x_s = p_sample_loop(denoiser, schedule, x_obs=x_obs, obs_mask=mask, cond=cond)
        samples.append(x_s[:, 0, :].detach().cpu().numpy())
    arr = np.stack(samples, axis=0)
    mean_norm = np.mean(arr, axis=0)
    std_norm = np.std(arr, axis=0)
    mean_lab = lab_norm.denormalize(mean_norm)
    std_lab = std_norm * np.asarray([lab_norm.L_scale, lab_norm.ab_scale, lab_norm.ab_scale], dtype=np.float32)
    std_scalar = float(np.mean(np.linalg.norm(std_norm, axis=-1)))
    return mean_lab.astype(np.float32), std_lab.astype(np.float32), {
        'diffusion_std_norm_meanL2': std_scalar,
        'conf_diffusion': float(np.exp(-std_scalar)),
        'num_samples': int(num_samples),
    }

