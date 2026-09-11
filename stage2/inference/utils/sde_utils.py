# -*- coding: utf-8 -*-
"""SDE utilities with official-compatible enhanced inference support.

This file preserves the original ``reverse_sde(...)`` entry point used by the
official test code. For the final BrushNet/MGLC pipeline, texture inference
still enters here but uses explicit ``mask_known`` / ``mask_hole`` handling.
"""

import abc
import logging
import math
import os

import numpy as np
import torch
import torchvision.utils as tvutils
from scipy import integrate
from tqdm import tqdm

logger = logging.getLogger("base")

class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def ode_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)#, self.drift(x, t),self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t)

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)

    def reverse_ode_step(self, x, score, t):
        return x - self.ode_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

        return x


#############################################################################


class IRSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, max_sigma, T=100, schedule='cosine', eps=0.01,  device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
            
        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)
        
        self.mu = 0.
        self.model = None

    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    #####################################

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, t):
        return self.thetas[t] * (self.mu - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def ode_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t]**2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def _extract_model_output(self, output):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def score_fn(self, x, t, S=None, **kwargs):
        # need to pre-set mu and score_model
        if S is None:
            noise = self.model(x, self.mu, t, **kwargs)
        else:
            noise = self.model(x, self.mu, t, S, **kwargs)
        noise = self._extract_model_output(noise)
        return self.get_score_from_noise(noise, t)

    def noise_fn(self, x, t, S=None, **kwargs):
        # need to pre-set mu and score_model
        if S is None:
            noise = self.model(x, self.mu, t, **kwargs)
        else:
            noise = self.model(x, self.mu, t, S, **kwargs)
        return self._extract_model_output(noise)

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t)**2

    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x
    

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1), torch.tensor(beta)], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    
    def get_beta_schedule(self,beta_schedule='linear', *, beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=100):
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if beta_schedule == "quad":
            betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        elif beta_schedule == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  
            betas = 1.0 / np.linspace(
                num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas
    

    def _save_state_image(self, tensor, save_dir, index):
        os.makedirs(save_dir, exist_ok=True)
        tvutils.save_image(tensor.data, f"{save_dir}/state_{index}.png", normalize=False)

    def _validate_enhanced_masks(self, mask_known, mask_hole):
        if mask_known is None or mask_hole is None:
            raise ValueError("Enhanced inference requires both mask_known and mask_hole.")
        if mask_known.shape != mask_hole.shape:
            raise ValueError(
                f"Enhanced inference mask shape mismatch: mask_known={tuple(mask_known.shape)}, "
                f"mask_hole={tuple(mask_hole.shape)}"
            )
        deviation = torch.max(torch.abs(mask_known + mask_hole - 1.0)).item()
        if deviation > 1e-4:
            raise ValueError(
                f"mask_known and mask_hole must be complementary in enhanced inference; max deviation={deviation:.6f}"
            )

    def _masked_hole_stats(self, tensor, mask_hole):
        """Return simple hole-region safety stats for inference-only guards."""
        if tensor is None or mask_hole is None:
            return 0.0, 0.0, 0.0, 0.0
        x = tensor.detach().float()
        mask = mask_hole.detach().float()
        if mask.shape[1] != x.shape[1]:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        denom = mask.sum().clamp_min(1.0)
        mean = (x * mask).sum() / denom
        values = x[mask > 0.5]
        if values.numel() > 0:
            min_val = values.min()
            max_val = values.max()
        else:
            min_val = x.new_tensor(0.0)
            max_val = x.new_tensor(0.0)
        white = ((x > 0.95).all(dim=1, keepdim=True).float() * mask_hole[:, :1]).sum()
        white_ratio = white / mask_hole[:, :1].sum().clamp_min(1.0)
        return (
            float(mean.item()),
            float(white_ratio.item()),
            float(min_val.item()),
            float(max_val.item()),
        )

    def _masked_abs_mean(self, tensor, mask_hole):
        if tensor is None or mask_hole is None:
            return 0.0
        x = tensor.detach().float().abs()
        mask = mask_hole.detach().float()
        if mask.shape[1] != x.shape[1]:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        return float((x * mask).sum().div(mask.sum().clamp_min(1.0)).item())

    def _masked_dark_ratio(self, tensor, mask_hole, threshold=0.12):
        if tensor is None or mask_hole is None:
            return 0.0
        x = tensor.detach().float()
        mask = mask_hole.detach().float()
        if x.shape[1] >= 3:
            luminance = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            luminance = x[:, :1]
        dark = (luminance < threshold).float() * mask[:, :1]
        return float(dark.sum().div(mask[:, :1].sum().clamp_min(1.0)).item())

    def _log_reverse_trajectory(self, tag, t, x, score, mask_hole):
        mean, white, min_val, max_val = self._masked_hole_stats(x, mask_hole)
        score_abs = self._masked_abs_mean(score, mask_hole)
        logger.info(
            "[Trajectory Debug] %s t=%d hole(mean=%.4f,min=%.4f,max=%.4f,white=%.4f) score_abs_mean=%.4f",
            tag,
            t,
            mean,
            min_val,
            max_val,
            white,
            score_abs,
        )

    def _safe_discriminator_candidate(self, candidate, reference, mask_hole):
        """Keep discriminator guidance from selecting obvious color outliers.

        The discriminator branch is legacy proposal selection.  In the mural
        target-domain pipeline the discriminator is not a color prior and must
        not be allowed to override the SDE/BrushNet trajectory with numerically
        unstable, all-white, or unsupported all-dark hole states.  This guard
        does not change the SDE
        update formula; it only rejects discriminator proposals that are clearly
        outside the target-domain hole range.
        """
        if candidate is None or not torch.isfinite(candidate).all():
            return False
        cand_mean, cand_white, cand_min, cand_max = self._masked_hole_stats(candidate, mask_hole)
        ref_mean, _, _, _ = self._masked_hole_stats(reference, mask_hole)
        cand_dark = self._masked_dark_ratio(candidate, mask_hole)
        ref_dark = self._masked_dark_ratio(reference, mask_hole)
        if cand_min < -1.0 or cand_max > 2.0:
            return False
        if cand_white > 0.25 and cand_mean > ref_mean + 0.08:
            return False
        if cand_mean > 0.95 and cand_mean > ref_mean + 0.12:
            return False
        # If the color/GT/LUT reference has almost no dark content in the hole,
        # reject discriminator proposals that introduce a new dark blob.  This
        # targets the baseline-checkpoint artifact where the original stochastic
        # sampler may select a dark patch in otherwise uniform mural regions.
        if ref_mean > 0.45 and ref_dark < 0.03 and cand_dark > 0.08:
            return False
        if ref_mean > 0.60 and cand_min < 0.05 and cand_dark > ref_dark + 0.05:
            return False
        return True

    def _reverse_sde_enhanced(
        self,
        xt,
        T,
        save_states,
        save_dir,
        mask_known,
        mask_hole,
        S_sde,
        S_GT,
        S_LQ,
        S_LQs,
        restore_S_guidance,
        deterministic_reverse,
        known_area_projection,
        **brushnet_kwargs,
    ):
        x_original = xt.clone().to(self.device)
        structure_state = S_LQ.clone().to(self.device) if S_LQ is not None else None
        early_start = max(1, int(0.4 * T))

        # Early stage follows the legacy restore_S_guidance path: update the
        # structure SDE and feed its state to the texture network.
        for t in tqdm(reversed(range(early_start, T + 1))):
            structure_tensor = None
            if restore_S_guidance and S_sde is not None and S_GT is not None and S_LQs is not None:
                xs_optimum = S_sde.generate_states(
                    x0=S_GT.to(self.device) * mask_known,
                    mu=S_LQs.to(self.device) * mask_known,
                    timesteps=t - 1,
                )
                if structure_state is None:
                    structure_state = xs_optimum
                else:
                    structure_state = xs_optimum * mask_known + structure_state * (1 - mask_known)
                structure_score = S_sde.score_fn(structure_state, t)
                structure_state = S_sde.reverse_sde_step(structure_state, structure_score, t)
                structure_tensor = structure_state

            score_original = self.score_fn(x_original, t, structure_tensor, **brushnet_kwargs)
            if deterministic_reverse:
                x_original = self.reverse_sde_step_mean(x_original, score_original, t)
            else:
                x_original = self.reverse_sde_step(x_original, score_original, t)

            # NOTE: known_area_projection is intentionally NOT applied inside the loop.
            if save_states:
                interval = max(1, self.T // 100)
                if t % interval == 0:
                    self._save_state_image(x_original, save_dir, t // interval)
                    if (t // interval) in {100, 75, 50, 25, 10, 5, 1}:
                        self._log_reverse_trajectory(
                            "enhanced_early",
                            t,
                            x_original,
                            score_original,
                            mask_hole,
                        )

        # Late stage restores original StrDiffusion behavior: feed current texture
        # grayscale as S.  This lets holes refine texture instead of being forced by
        # an external known-only edge map.
        for t in tqdm(reversed(range(1, early_start))):
            structure_tensor = torch.mean(x_original, dim=1, keepdim=True) if restore_S_guidance else None
            score_original = self.score_fn(x_original, t, structure_tensor, **brushnet_kwargs)
            if deterministic_reverse:
                x_original = self.reverse_sde_step_mean(x_original, score_original, t)
            else:
                x_original = self.reverse_sde_step(x_original, score_original, t)

            if save_states:
                interval = max(1, self.T // 100)
                if t % interval == 0:
                    self._save_state_image(x_original, save_dir, t // interval)
                    if (t // interval) in {100, 75, 50, 25, 10, 5, 1}:
                        self._log_reverse_trajectory(
                            "enhanced_late",
                            t,
                            x_original,
                            score_original,
                            mask_hole,
                        )

        # Return the raw model prediction without compositing.
        # denoising_model.test() handles the final known/hole composite.
        return x_original

    def _reverse_sde_enhanced_with_discriminator(
        self,
        xt,
        T,
        save_states,
        save_dir,
        mask_known,
        mask_hole,
        S_sde,
        S_GT,
        S_LQ,
        S_LQs,
        dis,
        restore_S_guidance,
        deterministic_reverse,
        known_area_projection,
        **brushnet_kwargs,
    ):
        x_original = xt.clone().to(self.device)
        xs = S_LQ.clone().to(self.device)
        early_start = max(1, int(0.4 * T))
        guard_reference = brushnet_kwargs.get("color_prior", None)
        if guard_reference is None:
            guard_reference = self.mu
        guard_rejects = 0

        for t in tqdm(reversed(range(early_start, T + 1))):
            structure_tensor = None
            if restore_S_guidance and S_sde is not None and S_GT is not None and S_LQs is not None:
                xs_optimum = S_sde.generate_states(
                    x0=S_GT.to(self.device) * mask_known,
                    mu=S_LQs.to(self.device) * mask_known,
                    timesteps=t - 1,
                )
                xs = xs_optimum * mask_known + xs * (1 - mask_known)
                scores = S_sde.score_fn(xs, t)
                xs = S_sde.reverse_sde_step(xs, scores, t)
                structure_tensor = xs

            score_original = self.score_fn(x_original, t, structure_tensor, **brushnet_kwargs)
            if deterministic_reverse:
                x_nominal = self.reverse_sde_step_mean(x_original, score_original, t)
            else:
                x_nominal = self.reverse_sde_step(x_original, score_original, t)
            x_updated = x_nominal
            # No per-step known_area_projection — see _reverse_sde_enhanced comment.

            if (
                dis is not None
                and structure_tensor is not None
                and restore_S_guidance
            ):
                d_current = dis(
                    torch.tensor(t, device=self.device).reshape(1,),
                    x_updated.detach() * mask_known,
                    structure_tensor.detach(),
                ).view(-1)
                u_max = 6
                u_min = 3
                jump = 5
                re = 5
                step = re if t % jump == 0 else 0
                if step + t > T:
                    step = T - t + 1
                xs_t = structure_tensor
                for i in range(1, u_max):
                    if step == 0:
                        break
                    xs1 = xs_t
                    for j in range(step):
                        xs1 = S_sde.forward_step(xs1, t - 1 + j)
                    for z in reversed(range(j + 1)):
                        xs1 = xs_optimum * mask_known + xs1 * (1 - mask_known)
                        scores = S_sde.score_fn(xs1, t + z)
                        xs1 = S_sde.reverse_sde_step(xs1, scores, t + z)
                    score_tmp = self.score_fn(x_original, t, xs1, **brushnet_kwargs)
                    if deterministic_reverse:
                        x_tmp = self.reverse_sde_step_mean(x_original, score_tmp, t)
                    else:
                        x_tmp = self.reverse_sde_step(x_original, score_tmp, t)
                    if not self._safe_discriminator_candidate(x_tmp, guard_reference, mask_hole):
                        guard_rejects += 1
                        continue
                    d_proposal = dis(
                        torch.tensor(t, device=self.device).reshape(1,),
                        x_tmp.detach() * mask_known,
                        xs1.detach(),
                    ).view(-1)
                    if i >= u_min and d_proposal >= d_current:
                        break
                    if d_proposal < d_current:
                        x_updated = x_tmp
                        xs_t = xs1
                        d_current = d_proposal
                    else:
                        x_blend = (x_updated + x_tmp) / 2
                        if self._safe_discriminator_candidate(x_blend, guard_reference, mask_hole):
                            x_updated = x_blend
                            xs_t = (xs1 + xs_t) / 2
                        else:
                            guard_rejects += 1
                xs = xs_optimum * mask_known + xs_t * (1 - mask_known)

            if not self._safe_discriminator_candidate(x_updated, guard_reference, mask_hole):
                guard_rejects += 1
                x_updated = x_nominal
            x_original = x_updated

            if save_states:
                interval = max(1, self.T // 100)
                if t % interval == 0:
                    self._save_state_image(x_original, save_dir, t // interval)

        for t in tqdm(reversed(range(1, early_start))):
            # Match the original discriminator-guided StrDiffusion late stage:
            # feed current texture grayscale as S so the texture branch can refine
            # hole content instead of being constrained by a known-only edge map.
            structure_tensor = torch.mean(x_original, dim=1, keepdim=True) if restore_S_guidance else None
            score_original = self.score_fn(x_original, t, structure_tensor, **brushnet_kwargs)
            if deterministic_reverse:
                x_original = self.reverse_sde_step_mean(x_original, score_original, t)
            else:
                x_original = self.reverse_sde_step(x_original, score_original, t)
            # No per-step known_area_projection ? see _reverse_sde_enhanced comment.
            if save_states:
                interval = max(1, self.T // 100)
                if t % interval == 0:
                    self._save_state_image(x_original, save_dir, t // interval)

        if guard_rejects:
            logger.info(
                "[DiscriminatorGuard] rejected_candidates=%d using color/GT/LUT reference safety range",
                guard_rejects,
            )

        # Return the raw model prediction without compositing.
        # denoising_model.test() handles the final known/hole composite.
        return x_original

    def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state',GT = None, mask = None, S_sde = None, S_GT = None, S_LQ = None, dis = None, S_LQs = None, **kwargs):
        T = self.T if T < 0 else T

        if kwargs.get("enhanced_inference", False):
            if mask is None:
                raise ValueError("Enhanced inference requires mask=mask_known in reverse_sde(...).")
            if "mask_hole" not in kwargs or kwargs["mask_hole"] is None:
                raise ValueError("Enhanced inference requires mask_hole in reverse_sde(...).")
            mask_known = mask.to(self.device).float()
            mask_hole = kwargs["mask_hole"].to(self.device).float()
            deterministic_reverse = bool(kwargs.get("deterministic_reverse", True))
            known_area_projection = bool(kwargs.get("known_area_projection", True))
            self._validate_enhanced_masks(mask_known, mask_hole)
            # BrushNet / MGLC must always receive the hole mask, never mask_known.
            brushnet_kwargs = {
                "mask": mask_hole,
                "color_prior": kwargs.get("color_prior").to(self.device)
                if kwargs.get("color_prior") is not None
                else None,
                "confidence": kwargs.get("confidence").to(self.device)
                if kwargs.get("confidence") is not None
                else None,
            }
            brushnet_kwargs = {
                key: value for key, value in brushnet_kwargs.items() if value is not None
            }

            if kwargs.get("discriminator_guidance", False):
                return self._reverse_sde_enhanced_with_discriminator(
                    xt.to(self.device),
                    T,
                    save_states,
                    save_dir,
                    mask_known,
                    mask_hole,
                    S_sde,
                    S_GT,
                    S_LQ,
                    S_LQs,
                    dis,
                    kwargs.get("restore_S_guidance", False),
                    deterministic_reverse,
                    known_area_projection,
                    **brushnet_kwargs,
                )

            return self._reverse_sde_enhanced(
                xt.to(self.device),
                T,
                save_states,
                save_dir,
                mask_known,
                mask_hole,
                S_sde,
                S_GT,
                S_LQ,
                S_LQs,
                kwargs.get("restore_S_guidance", False),
                deterministic_reverse,
                known_area_projection,
                **brushnet_kwargs,
            )

        S_GT = S_GT.cuda()
        GT = GT.cuda()
        S_LQ = S_LQ.cuda()
        mask = mask.cuda()
        
        xt = xt.cuda()
        x_original = xt.clone()
        xs = S_LQ.clone()

        # Adaptive Resampling Strategy #
        for t in tqdm(reversed(range(int(0.4*T), int(T+1)))):# Early Stage
            xs_optimum = S_sde.generate_states(x0=S_GT.cuda() * mask.cuda(), mu=S_LQs.cuda() * mask.cuda(), timesteps = t-1)
            xs = xs_optimum * mask.cuda() + xs * (1 - mask.cuda())
            scores = S_sde.score_fn(xs, t)
            xs = S_sde.reverse_sde_step(xs, scores, t)
            xs_t = xs
    
            score_original = self.score_fn(x_original, t, xs, **kwargs)
            x_updated = self.reverse_sde_step(x_original, score_original, t)
    
            D_n = dis(torch.tensor(t).reshape(1,), x_updated.detach() * mask.cuda(), xs.detach()).view(-1)
            u_max = 6
            u_min = 3
            jump = 5
            re = 5
            step = 0
            if t % jump == 0:
                step = re
            if step + t > T:
                step = T - t + 1
            for i in range(1,u_max):
                if step != 0:
                    xs1 = xs_t
                    for j in range(0,step):
                        xs1 = S_sde.forward_step(xs1,t-1+j)
                    for z in reversed(range(0,j+1)):
                        xs1 = xs_optimum * mask.cuda() + xs1 * (1 - mask.cuda())
                        scores = S_sde.score_fn(xs1, t+z)
                        xs1 = S_sde.reverse_sde_step(xs1, scores, t+z)
                    score = self.score_fn(x_original, t, xs1, **kwargs)
                    x_tmp = self.reverse_sde_step(x_original, score, t)
                    D_p = dis(torch.tensor(t).reshape(1,), x_tmp.detach() * mask.cuda(), xs1.detach()).view(-1)
                    if i >= u_min:
                        if D_p < D_n:
                            x_updated = x_tmp
                            xs_t = xs1
                        else: 
                            break
                    else:
                        if D_p < D_n:
                            x_updated = x_tmp
                            xs_t = xs1
                        else: 
                            x_updated = (x_updated + x_tmp)/2
                            xs_t = (xs1 + xs_t)/2
                else:
                    break
            x_original = x_updated
            xs = xs_optimum * mask.cuda() + xs_t * (1 - mask.cuda())
            
        for t in tqdm(reversed(range(1, int(0.4*T)))):# Late Stage
            xs = torch.mean(x_original, dim=1, keepdim=True)
            score_original = self.score_fn(x_original, t, xs, **kwargs)
            x_original = self.reverse_sde_step(x_original, score_original, t)
            
        return GT.cuda() * mask.cuda() + x_original * (1 - mask.cuda())

    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,):
        shape = xt.shape

        def to_flattened_numpy(x):
          """Flatten a torch tensor `x` and convert it to numpy."""
          return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
          """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
          return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                     rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(2, self.T + 1, (batch, 1, 1, 1)).long()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32)

    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma

                
