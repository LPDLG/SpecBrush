"""GT-free released inference model for SpecBrush Stage II."""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torch.nn.parallel import DataParallel

import models.networks as networks
from color_prior_generator import ColorPriorGenerator
from .base_model import BaseModel
from .mu_denoiser import MuCleaner

logger = logging.getLogger("base")


class DenoisingModel(BaseModel):
    """Run the 200-step IR-SDE restoration without any GT-dependent decision."""

    def __init__(self, opt):
        super().__init__(opt)
        self.dataset_opt = next(iter(opt["datasets"].values()))
        self.inference_opt = opt.get("inference", {}) or {}

        self.model, self.models, self.dis = networks.define_G(opt)
        self.model = self.model.to(self.device)
        self.models = self.models.to(self.device)
        self.dis = self.dis.to(self.device)
        self.dis.eval()
        for p in self.dis.parameters():
            p.requires_grad_(False)

        gpu_ids = opt.get("gpu_ids", None)
        if gpu_ids is not None and len(gpu_ids) > 1:
            self.model = DataParallel(self.model, device_ids=gpu_ids)
            self.models = DataParallel(self.models, device_ids=gpu_ids)

        mu_opt = opt.get("mu_cleaner", {}) or {}
        self.mu_cleaner = None
        if bool(mu_opt.get("enabled", True)):
            self.mu_cleaner = MuCleaner(
                dim=int(mu_opt.get("dim", 32)),
                num_blocks=int(mu_opt.get("num_blocks", 2)),
                num_heads=int(mu_opt.get("num_heads", 4)),
                boundary_width=int(mu_opt.get("boundary_width", 3)),
            ).to(self.device)

        lut_path = self.dataset_opt.get("lut_path", None)
        if not lut_path:
            raise ValueError("SpecBrush inference requires the offline Stage-I LUT")
        self.color_prior_generator = ColorPriorGenerator(
            lut_path=lut_path,
            alpha=float(self.dataset_opt.get("lut_alpha", 0.85)),
            beta=float(self.dataset_opt.get("lut_beta", 0.15)),
            inpaint_method=str(self.dataset_opt.get("lut_inpaint_method", "telea")),
            inpaint_mask_dilate=int(self.dataset_opt.get("prior_inpaint_mask_dilate", 3)),
        )

        self.load()
        self.model.eval()
        self.models.eval()
        if self.mu_cleaner is not None:
            self.mu_cleaner.eval()

        self.output = None
        self.debug_outputs: Dict[str, torch.Tensor] = {}
        self.sample_name = None

    @staticmethod
    def _unwrap(network):
        return network.module if isinstance(network, DataParallel) else network

    def _load_combined_stage2(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict):
            for container_key in ("state_dict", "params_ema", "params"):
                if container_key in checkpoint and isinstance(checkpoint[container_key], dict):
                    checkpoint = checkpoint[container_key]
                    break
        model_state, mu_state = OrderedDict(), OrderedDict()
        for key, value in checkpoint.items():
            k = key[7:] if key.startswith("module.") else key
            if k.startswith("mu_cleaner."):
                mu_state[k[len("mu_cleaner."):]] = value
            elif k.startswith("mu_denoiser."):
                # Accept old prefix only to make migration errors explicit via missing keys.
                mu_state[k[len("mu_denoiser."):]] = value
            else:
                model_state[k] = value
        result = self._unwrap(self.model).load_state_dict(model_state, strict=False)
        logger.info(
            "Loaded SpecBrush Stage-II checkpoint: missing=%d unexpected=%d",
            len(result.missing_keys), len(result.unexpected_keys),
        )
        if self.mu_cleaner is not None:
            if not mu_state:
                raise RuntimeError(
                    "SpecBrush Stage-II checkpoint must contain mu_cleaner.* weights. "
                    "Use a checkpoint produced by the current release for deployment inference."
                )
            self.mu_cleaner.load_state_dict(mu_state, strict=True)

    def load(self):
        paths = self.opt.get("path", {}) or {}
        g_path = paths.get("pretrain_model_G", None)
        gs_path = paths.get("pretrain_model_Gs", None)
        if g_path and os.path.exists(g_path):
            self._load_combined_stage2(g_path)
        elif g_path:
            logger.warning("Stage-II checkpoint not found: %s", g_path)

        if gs_path and os.path.exists(gs_path):
            state = torch.load(gs_path, map_location=self.device)
            if isinstance(state, dict):
                for container_key in ("state_dict", "params_ema", "params"):
                    if container_key in state and isinstance(state[container_key], dict):
                        state = state[container_key]
                        break
            cleaned = OrderedDict()
            for key, value in state.items():
                cleaned[key[7:] if key.startswith("module.") else key] = value
            result = self._unwrap(self.models).load_state_dict(cleaned, strict=False)
            logger.info(
                "Loaded frozen StrDiffusion structure model: missing=%d unexpected=%d",
                len(result.missing_keys), len(result.unexpected_keys),
            )
        elif gs_path:
            logger.warning("StrDiffusion structure checkpoint not found: %s", gs_path)

    def feed_data(
        self,
        degraded: torch.Tensor,
        mask_known: torch.Tensor,
        mask_hole: torch.Tensor,
        color_prior: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        sample_name: Optional[str] = None,
    ) -> None:
        self.degraded = degraded.to(self.device).float()
        self.mask_known = mask_known.to(self.device).float().clamp(0.0, 1.0)
        self.mask_hole = mask_hole.to(self.device).float().clamp(0.0, 1.0)
        deviation = (self.mask_known + self.mask_hole - 1.0).abs().max().item()
        if deviation > 1e-4:
            raise ValueError(f"mask_known/mask_hole are not complementary: {deviation:.6f}")
        self.color_prior = color_prior.to(self.device).float() if color_prior is not None else None
        self.confidence = confidence.to(self.device).float() if confidence is not None else None
        self.sample_name = sample_name

    def _generate_prior_if_needed(self):
        if self.color_prior is not None and self.confidence is not None:
            return self.color_prior, self.confidence
        generated = self.color_prior_generator.generate_tensor(
            self.degraded,
            self.mask_hole,
            device=self.device,
            method=str(self.dataset_opt.get("prior_method", "quality")),
            debug=False,
        )
        prior, confidence = generated
        if self.color_prior is not None:
            prior = self.color_prior
        if self.confidence is not None:
            confidence = self.confidence
        return prior, confidence

    @staticmethod
    def _edge_from_rgb(image: torch.Tensor) -> torch.Tensor:
        """Build the frozen StrDiffusion structure condition from observed RGB only."""
        edges = []
        for i in range(image.shape[0]):
            rgb = image[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
            gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
            edge = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
            edges.append(torch.from_numpy(edge)[None, ...])
        return torch.stack(edges, dim=0).to(image.device, image.dtype)

    @torch.no_grad()
    def test(self, sde, structure_sde) -> torch.Tensor:
        """Run the complete GT-free reverse process for exactly T steps."""
        requested_steps = int(self.inference_opt.get("sampling_steps", 200))
        if requested_steps != int(sde.T):
            raise ValueError(
                f"SpecBrush inference requires sampling_steps==SDE.T; "
                f"got {requested_steps} vs {sde.T}"
            )
        if requested_steps != 200:
            raise ValueError(f"Stage-II inference requires 200 steps; got {requested_steps}")

        prior, confidence = self._generate_prior_if_needed()

        # Eq. (11): replace the original masked degraded mean with MuCleaner.
        raw_mu = self.degraded * self.mask_known
        mu_c = self.mu_cleaner(raw_mu, self.mask_hole) if self.mu_cleaner is not None else raw_mu
        sde.set_mu(mu_c)

        # Preserve the original frozen StrDiffusion structure branch without GT.
        # Its observed condition is derived only from the degraded RGB image.
        edge = self._edge_from_rgb(self.degraded)
        structure_mu = edge * self.mask_known
        structure_sde.set_mu(structure_mu)
        structure_state = structure_sde.noise_state(structure_mu)
        state = sde.noise_state(mu_c)

        for t in reversed(range(1, requested_steps + 1)):
            structure_score = structure_sde.score_fn(structure_state, t)
            structure_state = structure_sde.reverse_sde_step(
                structure_state, structure_score, t
            )
            score = sde.score_fn(
                state,
                t,
                structure_state,
                mask=self.mask_hole,
                color_prior=prior,
                confidence=confidence,
                observed_degraded=self.degraded,
            )
            state = sde.reverse_sde_step(state, score, t)

        output = state.clamp(0.0, 1.0)
        if bool(self.inference_opt.get("preserve_known_pixels", True)):
            output = output * self.mask_hole + self.degraded * self.mask_known

        self.output = output
        self.debug_outputs = {
            "degraded": self.degraded,
            "mask_hole": self.mask_hole,
            "color_prior": prior,
            "confidence": confidence,
            "mu_c": mu_c,
            "output": output,
        }
        return output

    def get_current_visuals(self):
        return {key: value.detach().float().cpu() for key, value in self.debug_outputs.items()}
