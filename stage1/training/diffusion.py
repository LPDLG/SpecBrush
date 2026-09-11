
"""
DDPM-style diffusion utilities for masked time series imputation.

We follow the common "diffuse only missing entries" practice:
- observed entries are treated as condition and kept fixed through forward and reverse process.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class DiffusionConfig:
    T: int = 200
    beta_0: float = 1e-4
    beta_T: float = 0.02


class DiffusionSchedule:
    def __init__(self, cfg: DiffusionConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.T = int(cfg.T)

        betas = torch.linspace(cfg.beta_0, cfg.beta_T, self.T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        x0, noise: (B,L,C)
        t: (B,) long
        """
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    MSE over entries where mask==1.
    """
    loss = (pred - target) ** 2
    loss = loss * mask
    return loss.sum() / (mask.sum() + eps)


def diffusion_loss(
    model: nn.Module,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    obs_mask: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    *,
    return_x0_pred: bool = False,
    return_t: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    One-step training loss:
    - sample t ~ Uniform(0, T-1)
    - diffuse x0 -> x_t (noise added)
    - keep observed entries fixed
    - predict noise epsilon
    - compute MSE on missing entries only

    Optionally return:
      - x0_pred: reconstructed x0 from (x_t, eps_hat, t) (with observed entries re-imposed)
      - t: diffusion timestep tensor (B,)
    """
    device = x0.device
    B = x0.shape[0]
    t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = schedule.q_sample(x0, t, noise)
    # keep observed fixed
    x_t = x_t * (1.0 - obs_mask) + x0 * obs_mask

    eps_hat = model(x_t, obs_mask, t, cond)
    missing_mask = 1.0 - obs_mask
    loss = masked_mse(eps_hat, noise, missing_mask)

    if (not return_x0_pred) and (not return_t):
        return loss

    # x0 reconstruction: x0_hat = (x_t - sqrt(1-a_bar)*eps_hat) / sqrt(a_bar)
    ab = schedule.alpha_bar[t].view(-1, 1, 1)
    sqrt_ab = torch.sqrt(ab).clamp_min(1e-8)
    sqrt_omab = torch.sqrt(1.0 - ab).clamp_min(1e-8)
    x0_pred = (x_t - sqrt_omab * eps_hat) / sqrt_ab

    # re-impose observed (avoid drift on conditioned entries)
    x0_pred = x0_pred * (1.0 - obs_mask) + x0 * obs_mask

    outs = [loss]
    if return_x0_pred:
        outs.append(x0_pred)
    if return_t:
        outs.append(t)

    return tuple(outs)


@torch.no_grad()
def p_sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    obs_mask: torch.Tensor,
    t: int,
    cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    One reverse step: p(x_{t-1} | x_t)
    """
    B = x_t.shape[0]
    t_batch = torch.full((B,), t, device=x_t.device, dtype=torch.long)

    eps_hat = model(x_t, obs_mask, t_batch, cond)

    beta_t = schedule.betas[t_batch].view(-1, 1, 1)
    alpha_t = schedule.alphas[t_batch].view(-1, 1, 1)
    ab_t = schedule.alpha_bar[t_batch].view(-1, 1, 1)

    # DDPM mean
    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - ab_t)) * eps_hat)

    if t > 0:
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        x_prev = mean + sigma * noise
    else:
        x_prev = mean

    # re-impose observed values
    x_prev = x_prev * (1.0 - obs_mask) + x_t * obs_mask  # x_t already contains observed
    return x_prev


@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    schedule: DiffusionSchedule,
    x_obs: torch.Tensor,
    obs_mask: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Full reverse process from x_T ~ N(0,1) (for missing entries), while keeping observed fixed.
    Returns x_0 sample.
    """
    # initialize missing with noise, observed with x_obs
    x = torch.randn_like(x_obs)
    x = x * (1.0 - obs_mask) + x_obs * obs_mask

    for t in reversed(range(schedule.T)):
        x = p_sample(model, schedule, x, obs_mask, t=t, cond=cond)

    return x
