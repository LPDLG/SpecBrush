"""SpecBrush Stage-II training model for SpecBrush.

Only PriorControlNet, MuCleaner and MGLC are optimized. The pretrained
StrDiffusion UNet is frozen. The sole optimization objective is the pixel-space
IR-SDE noise-prediction loss.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.networks as networks
from .base_model import BaseModel
from .mu_denoiser import MuCleaner

logger = logging.getLogger("base")


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.train_opt = opt.get("train", {}) or {}
        self.rank = -1

        self.model, self.dis = networks.define_G(opt)
        self.model = self.model.to(self.device)
        self.dis = self.dis.to(self.device)
        for p in self.dis.parameters():
            p.requires_grad_(False)
        self.dis.eval()

        gpu_ids = opt.get("gpu_ids", None)
        if gpu_ids is not None and len(gpu_ids) > 1:
            self.model = DataParallel(self.model, device_ids=gpu_ids, output_device=gpu_ids[0])

        mu_opt = opt.get("mu_cleaner", {}) or {}
        self.use_mu_cleaner = bool(mu_opt.get("enabled", True))
        self.mu_cleaner = None
        if self.use_mu_cleaner:
            self.mu_cleaner = MuCleaner(
                dim=int(mu_opt.get("dim", 32)),
                num_blocks=int(mu_opt.get("num_blocks", 2)),
                num_heads=int(mu_opt.get("num_heads", 4)),
                boundary_width=int(mu_opt.get("boundary_width", 3)),
            ).to(self.device)

        self.load()
        self._freeze_pretrained_backbone()

        self.log_dict = OrderedDict()
        self._training_debug: Dict[str, torch.Tensor] = {}
        if self.is_train:
            self._build_optimizer()

    @staticmethod
    def _unwrap(network):
        return network.module if isinstance(network, (DataParallel, DistributedDataParallel)) else network

    def _freeze_pretrained_backbone(self) -> None:
        """Freeze StrDiffusion and leave only the three SpecBrush control modules trainable."""
        if not bool(self.train_opt.get("freeze_backbone", True)):
            raise ValueError("SpecBrush Stage II requires train.freeze_backbone=true")
        module = self._unwrap(self.model)
        trainable_prefixes = (
            "prior_controlnet.",
            "prior_gates.",
            "prior_gate_mid.",
            "mglc_dec.",
            "mglc_mid.",
        )
        for name, param in module.named_parameters():
            param.requires_grad_(name.startswith(trainable_prefixes))

        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        mu_params = sum(p.numel() for p in self.mu_cleaner.parameters()) if self.mu_cleaner is not None else 0
        logger.info(
            "[PaperAlign] frozen StrDiffusion params=%d; PriorControlNet+MGLC params=%d; MuCleaner params=%d",
            frozen, trainable, mu_params,
        )

    def _build_optimizer(self) -> None:
        module = self._unwrap(self.model)
        params = [p for p in module.parameters() if p.requires_grad]
        if self.mu_cleaner is not None:
            params += list(self.mu_cleaner.parameters())
        if not params:
            raise RuntimeError("No trainable Stage-II control parameters")
        self._trainable_params = params

        self.optimizer = torch.optim.Adam(
            params,
            lr=float(self.train_opt.get("lr_new", 1e-6)),
            betas=(
                float(self.train_opt.get("beta1", 0.9)),
                float(self.train_opt.get("beta2", 0.99)),
            ),
            weight_decay=float(self.train_opt.get("weight_decay", 0.0)),
        )
        self.optimizers.append(self.optimizer)
        if str(self.train_opt.get("lr_scheme", "MultiStepLR")) != "MultiStepLR":
            raise ValueError("SpecBrush release uses MultiStepLR")
        self.schedulers.append(
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[int(x) for x in self.train_opt.get("lr_steps", [100000, 150000])],
                gamma=float(self.train_opt.get("lr_gamma", 0.5)),
            )
        )

    def compute_mu_clean(self, raw_mu: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """Eq. (11), with gradients retained for end-to-end Stage-II training."""
        if self.mu_cleaner is None:
            return raw_mu
        return self.mu_cleaner(raw_mu, missing_mask)

    def feed_data(
        self,
        state,
        condition_mu,
        GT,
        mask_known,
        S_sde,
        S_GT,
        S_LQ,
        color_prior=None,
        confidence=None,
        original_degraded=None,
        **_: object,
    ):
        self.state = state.to(self.device)
        self.condition = condition_mu.to(self.device)
        self.state_0 = GT.to(self.device)
        self.mask = mask_known.to(self.device)
        self.S_sde = S_sde
        self.S_GT = S_GT.to(self.device)
        self.S_LQ = S_LQ.to(self.device)
        self.color_prior = color_prior.to(self.device) if color_prior is not None else None
        self.confidence = confidence.to(self.device) if confidence is not None else None
        self.original_degraded = (
            original_degraded.to(self.device) if original_degraded is not None else self.condition
        )
        self._training_debug = {
            "condition_mu": self.condition.detach(),
            "training_target": self.state_0.detach(),
            "color_prior": self.color_prior.detach() if self.color_prior is not None else None,
            "confidence": self.confidence.detach() if self.confidence is not None else None,
            "mask_known": self.mask.detach(),
            "mask_hole": (1.0 - self.mask).detach(),
            "original_degraded": self.original_degraded.detach(),
        }

    def optimize_parameters(self, step, timesteps, sde=None):
        if sde is None:
            raise ValueError("Stage-II optimization requires IRSDE")
        self.log_dict = OrderedDict()
        self.model.train()
        if self.mu_cleaner is not None:
            self.mu_cleaner.train()

        timesteps = timesteps.to(self.device)
        sde.set_mu(self.condition)

        # Preserve the original StrDiffusion structure condition, but do not add
        # any structure loss: it is only an input to the frozen backbone.
        _, s_state = self.S_sde.generate_random_states_texture(
            x0=self.S_GT,
            mu=self.S_LQ * self.mask,
            timesteps=timesteps,
        )
        s_optimum = self.S_sde.reverse_optimum_step(s_state, self.S_GT, timesteps)

        kwargs = {
            "mask": 1.0 - self.mask,  # PriorControlNet/MGLC convention: 1=missing.
            "color_prior": self.color_prior,
            "confidence": self.confidence,
            "observed_degraded": self.original_degraded,
        }
        model_output = sde.noise_fn(
            self.state,
            timesteps.squeeze(),
            s_optimum,
            **kwargs,
        )
        predicted_noise = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        target_noise = sde.get_real_noise(self.state, self.state_0, timesteps).detach()

        # Eq. (16): the ONLY Stage-II training loss.
        loss = F.mse_loss(predicted_noise, target_noise)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip = float(self.train_opt.get("grad_clip", 1.0))
        if clip > 0:
            nn.utils.clip_grad_norm_(self._trainable_params, clip)
        self.optimizer.step()

        with torch.no_grad():
            hole = (1.0 - self.mask).expand_as(predicted_noise)
            known = self.mask.expand_as(predicted_noise)
            sq = (predicted_noise - target_noise).square()
            hole_mse = (sq * hole).sum() / hole.sum().clamp_min(1.0)
            known_mse = (sq * known).sum() / known.sum().clamp_min(1.0)
        self.log_dict["loss"] = float(loss.item())
        self.log_dict["loss_main"] = float(loss.item())
        self.log_dict["loss_total"] = float(loss.item())
        self.log_dict["loss_hole"] = float(hole_mse.item())
        self.log_dict["loss_known"] = float(known_mse.item())
        self.log_dict["mask_hole_ratio"] = float((1.0 - self.mask).mean().item())

    def get_current_log(self):
        return self.log_dict

    def get_current_training_debug(self):
        return self._training_debug

    def load(self):
        path = self.opt.get("path", {}).get("pretrain_model_G", None)
        if not path or not os.path.exists(path):
            if path:
                logger.warning("Stage-II pretrained checkpoint not found yet: %s", path)
            return
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "params_ema", "params"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        model_state, mu_state = {}, {}
        for key, value in checkpoint.items():
            k = key[7:] if key.startswith("module.") else key
            if k.startswith("mu_cleaner."):
                mu_state[k[len("mu_cleaner."):]] = value
            elif k.startswith("mu_denoiser."):
                mu_state[k[len("mu_denoiser."):]] = value
            else:
                model_state[k] = value
        result = self._unwrap(self.model).load_state_dict(model_state, strict=False)
        logger.info(
            "Loaded Stage-II G: missing=%d unexpected=%d",
            len(result.missing_keys), len(result.unexpected_keys),
        )
        if self.mu_cleaner is not None and mu_state:
            self.mu_cleaner.load_state_dict(mu_state, strict=False)

    def save(self, iter_label):
        os.makedirs(self.opt["path"]["models"], exist_ok=True)
        path = os.path.join(self.opt["path"]["models"], f"{iter_label}_G.pth")
        state = {k: v.detach().cpu() for k, v in self._unwrap(self.model).state_dict().items()}
        if self.mu_cleaner is not None:
            for key, value in self.mu_cleaner.state_dict().items():
                state[f"mu_cleaner.{key}"] = value.detach().cpu()
        torch.save(state, path)

    def save_training_state(self, epoch, iter_step, label=None, extra_state=None):
        os.makedirs(self.opt["path"]["training_state"], exist_ok=True)
        state = {
            "epoch": epoch,
            "iter": iter_step,
            "schedulers": [s.state_dict() for s in self.schedulers],
            "optimizers": [o.state_dict() for o in self.optimizers],
        }
        if extra_state:
            state.update(extra_state)
        name = label if label is not None else str(iter_step)
        torch.save(state, os.path.join(self.opt["path"]["training_state"], f"{name}.state"))

    def resume_training(self, resume_state):
        super().resume_training(resume_state)

