"""SpecBrush training entry point for SpecBrush Stage I.

Implements the Stage-I objectives:
  L_stage1 = L_inv + lambda_align * L_align
  L_inv    = E ||epsilon - epsilon_theta(X_t, t, O, z_c)||^2
  L_align  = ||sg(z_m) - Psi_eta(E_c(c_deg))||^2

No prototype retrieval, physics-cycle loss, color augmentation, or auxiliary
posterior/distillation objectives are used in the default release path.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import PigmentNPZDataset
from models.color_encoder import ColorEncoder, ColorEncoderConfig
from models.cond_predictor import CondPredictorConfig, RGBConditionPredictor
from models.denoiser import DenoiserConfig, MambaDenoiser
from models.spectral_encoder import ConditionerConfig, MultimodalConditioner
from training.diffusion import DiffusionConfig, DiffusionSchedule, diffusion_loss
from utils.config_utils import load_config
from utils.seed import set_seed


def _last_observed_color(x0: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    """Return c_deg from X0=[c_ori,c_deg] using the observation mask O."""
    observed = (obs_mask.mean(dim=-1) > 0.5).long()
    idx = torch.arange(x0.shape[1], device=x0.device).view(1, -1)
    last_idx = (idx * observed).max(dim=1).values
    return x0[torch.arange(x0.shape[0], device=x0.device), last_idx]


def _build_full_material_condition(
    batch: Dict[str, torch.Tensor],
    x0: torch.Tensor,
    obs_mask: torch.Tensor,
    color_encoder: ColorEncoder,
    conditioner: MultimodalConditioner,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    c_deg = _last_observed_color(x0, obs_mask)
    z_color = color_encoder(c_deg)
    raman = batch.get("raman")
    xrd = batch.get("xrd")
    z_m = conditioner(
        z_color,
        raman.to(device) if raman is not None else None,
        xrd.to(device) if xrd is not None else None,
        raman_peaks=batch.get("raman_peaks").to(device) if "raman_peaks" in batch else None,
        xrd_peaks=batch.get("xrd_peaks").to(device) if "xrd_peaks" in batch else None,
    )
    return z_color, z_m


def _available_full_mask(batch: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    """Mask samples with available Raman/XRD. Interpolated training data are full."""
    available = torch.ones(batch_size, dtype=torch.bool, device=device)
    if "has_raman" in batch:
        available &= batch["has_raman"].to(device).view(-1).bool()
    if "has_xrd" in batch:
        available &= batch["has_xrd"].to(device).view(-1).bool()
    return available


def _make_models(cfg: Dict, device: torch.device):
    mod = cfg.get("modality", {})
    color_cfg_raw = cfg.get("color_encoder", {})
    color_encoder = ColorEncoder(
        ColorEncoderConfig(
            in_dim=3,
            d_model=int(color_cfg_raw.get("d_model", mod.get("spec_d_model", 128))),
            hidden_dim=int(color_cfg_raw.get("hidden_dim", 256)),
            n_layers=int(color_cfg_raw.get("n_layers", 2)),
            dropout=float(color_cfg_raw.get("dropout", 0.0)),
        )
    ).to(device)

    conditioner = MultimodalConditioner(
        ConditionerConfig(
            use_raman=bool(mod.get("use_raman", True)),
            use_xrd=bool(mod.get("use_xrd", True)),
            raman_len=int(mod.get("raman_len", 1024)),
            xrd_len=int(mod.get("xrd_len", 2048)),
            d_model=int(mod.get("spec_d_model", 128)),
            n_layers=int(mod.get("spec_n_layers", 4)),
            dropout=float(mod.get("spec_dropout", 0.0)),
            raman_peak_dim=int(mod.get("raman_peak_dim", 0)),
            xrd_peak_dim=int(mod.get("xrd_peak_dim", 0)),
            material_dim=int(mod.get("material_dim", 128)),
        )
    ).to(device)

    mm = cfg.get("missing_modality", {})
    rgb_predictor = RGBConditionPredictor(
        CondPredictorConfig(
            in_dim=color_encoder.cfg.d_model,
            out_dim=conditioner.cond_dim,
            hidden_dim=int(mm.get("hidden_dim", 256)),
            n_layers=int(mm.get("n_layers", 2)),
            dropout=float(mm.get("dropout", 0.0)),
        )
    ).to(device)

    model_cfg = cfg.get("model", {})
    denoiser = MambaDenoiser(
        DenoiserConfig(
            in_channels=int(model_cfg.get("in_channels", 3)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            n_layers=int(model_cfg.get("n_layers", 4)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            cond_dim=conditioner.cond_dim,
        )
    ).to(device)
    return color_encoder, conditioner, rgb_predictor, denoiser


def _save_checkpoint(
    path: str,
    cfg: Dict,
    epoch: int,
    val_loss: float,
    color_encoder: nn.Module,
    conditioner: nn.Module,
    rgb_predictor: nn.Module,
    denoiser: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    payload = {
        "cfg": cfg,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "color_encoder": color_encoder.state_dict(),
        "conditioner": conditioner.state_dict(),
        "rgb_condition_predictor": rgb_predictor.state_dict(),
        "denoiser": denoiser.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


@torch.no_grad()
def _validate(
    loader: DataLoader,
    device: torch.device,
    schedule: DiffusionSchedule,
    color_encoder: ColorEncoder,
    conditioner: MultimodalConditioner,
    rgb_predictor: RGBConditionPredictor,
    denoiser: MambaDenoiser,
    lambda_align: float,
) -> Dict[str, float]:
    for module in (color_encoder, conditioner, rgb_predictor, denoiser):
        module.eval()
    total, inv_total, align_total, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        x0 = batch["x0"].to(device)
        obs_mask = batch["mask"].to(device)
        z_color, z_m = _build_full_material_condition(
            batch, x0, obs_mask, color_encoder, conditioner, device
        )
        z_rgb = rgb_predictor(z_color)
        # RGB-only condition is the deployment setting and therefore the
        # validation condition used for checkpoint selection.
        l_inv = diffusion_loss(denoiser, schedule, x0, obs_mask, z_rgb)
        l_align = F.mse_loss(z_rgb, z_m.detach())
        l = l_inv + float(lambda_align) * l_align
        b = int(x0.shape[0])
        total += float(l.item()) * b
        inv_total += float(l_inv.item()) * b
        align_total += float(l_align.item()) * b
        n += b
    denom = max(n, 1)
    return {
        "loss": total / denom,
        "inv": inv_total / denom,
        "align": align_total / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    set_seed(int(train_cfg.get("seed", 42)))

    ds_train = PigmentNPZDataset(
        cfg["data"]["train_npz"], index_csv=str(cfg["data"].get("train_index", ""))
    )
    ds_val = PigmentNPZDataset(
        cfg["data"]["val_npz"], index_csv=str(cfg["data"].get("val_index", ""))
    )
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=int(train_cfg.get("eval_batch_size", 64)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    color_encoder, conditioner, rgb_predictor, denoiser = _make_models(cfg, device)
    dcfg = cfg.get("diffusion", {})
    schedule = DiffusionSchedule(
        DiffusionConfig(
            T=int(dcfg.get("T", 200)),
            beta_0=float(dcfg.get("beta_0", 1e-4)),
            beta_T=float(dcfg.get("beta_T", 0.02)),
        ),
        device=device,
    )

    # Default optimizer/training settings: Adam, lr=1e-4, batch=64, 300 epochs.
    params = (
        list(color_encoder.parameters())
        + list(conditioner.parameters())
        + list(rgb_predictor.parameters())
        + list(denoiser.parameters())
    )
    optimizer = torch.optim.Adam(
        params,
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    mm = cfg.get("missing_modality", {})
    p_full = float(mm.get("p_full", 0.7))
    lambda_align = float(mm.get("lambda_align", 0.1))
    if not 0.0 <= p_full <= 1.0:
        raise ValueError("missing_modality.p_full must be in [0,1]")

    epochs = int(train_cfg.get("epochs", 300))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    eval_every = int(train_cfg.get("eval_every", 1))
    save_every = int(train_cfg.get("save_every", 20))
    save_dir = str(train_cfg.get("save_dir", "ckpt/lab_raman_xrd"))
    os.makedirs(save_dir, exist_ok=True)

    best_val = float("inf")
    global_step = 0
    for epoch in range(1, epochs + 1):
        for module in (color_encoder, conditioner, rgb_predictor, denoiser):
            module.train()

        for batch in train_loader:
            x0 = batch["x0"].to(device)
            obs_mask = batch["mask"].to(device)
            z_color, z_m = _build_full_material_condition(
                batch, x0, obs_mask, color_encoder, conditioner, device
            )
            z_rgb = rgb_predictor(z_color)

            # Eq. (3): z_c = delta*z_m + (1-delta)*Psi(E_c(c_deg)).
            # Samples lacking a measured/interpolated spectrum are forced onto
            # the RGB-only branch; normally the preprocessed training data are full.
            available = _available_full_mask(batch, x0.shape[0], device)
            use_full = (torch.rand(x0.shape[0], device=device) < p_full) & available
            z_c = torch.where(use_full[:, None], z_m, z_rgb)

            l_inv = diffusion_loss(denoiser, schedule, x0, obs_mask, z_c)
            l_align = F.mse_loss(z_rgb, z_m.detach())
            loss = l_inv + lambda_align * l_align

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            global_step += 1

            if global_step % log_every == 0:
                print(
                    f"[epoch {epoch:03d} step {global_step:07d}] "
                    f"loss={loss.item():.6f} inv={l_inv.item():.6f} "
                    f"align={l_align.item():.6f} full_frac={use_full.float().mean().item():.3f}"
                )

        if epoch % eval_every == 0:
            metrics = _validate(
                val_loader,
                device,
                schedule,
                color_encoder,
                conditioner,
                rgb_predictor,
                denoiser,
                lambda_align,
            )
            print(
                f"[epoch {epoch:03d}] val_rgb_only={metrics['loss']:.6f} "
                f"inv={metrics['inv']:.6f} align={metrics['align']:.6f}"
            )
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                _save_checkpoint(
                    os.path.join(save_dir, "best_model.pt"), cfg, epoch, best_val,
                    color_encoder, conditioner, rgb_predictor, denoiser, optimizer,
                )

        if save_every > 0 and epoch % save_every == 0:
            _save_checkpoint(
                os.path.join(save_dir, f"epoch_{epoch:03d}.pt"), cfg, epoch, best_val,
                color_encoder, conditioner, rgb_predictor, denoiser, optimizer,
            )

    _save_checkpoint(
        os.path.join(save_dir, "latest_model.pt"), cfg, epochs, best_val,
        color_encoder, conditioner, rgb_predictor, denoiser, optimizer,
    )


if __name__ == "__main__":
    main()
