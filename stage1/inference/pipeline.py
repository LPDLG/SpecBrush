"""SpecBrush Stage-I inference for SpecBrush.

Deployment inference is RGB-only:
    c_deg -> E_c -> Psi_eta -> z_c -> masked diffusion inversion.
Raman/XRD are accepted only by the optional ``full`` analysis route.
For LeadAging sequence evaluation, K=20 posterior samples are aggregated and
adjacent time points are smoothed by Kalman-RTS before reporting the inversion.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data.dataset import PigmentNPZDataset
from inference.kalman_rts import kalman_rts_smooth_lab
from inference.uncertainty import sample_with_confidence
from models.color_encoder import ColorEncoder, ColorEncoderConfig
from models.cond_predictor import CondPredictorConfig, RGBConditionPredictor
from models.denoiser import DenoiserConfig, MambaDenoiser
from models.spectral_encoder import ConditionerConfig, MultimodalConditioner
from training.diffusion import DiffusionConfig, DiffusionSchedule
from utils.color_utils import LabNorm, delta_e2000, lab_to_rgb, rgb_to_lab


def _last_observed_color(x0: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    observed = (obs_mask.mean(dim=-1) > 0.5).long()
    idx = torch.arange(x0.shape[1], device=x0.device).view(1, -1)
    last_idx = (idx * observed).max(dim=1).values
    return x0[torch.arange(x0.shape[0], device=x0.device), last_idx]


def load_checkpoint(ckpt_path: str, device: torch.device) -> Dict[str, object]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt.get("cfg", {}))
    mod = cfg.get("modality", {})
    color_raw = cfg.get("color_encoder", {})

    color_encoder = ColorEncoder(
        ColorEncoderConfig(
            in_dim=3,
            d_model=int(color_raw.get("d_model", mod.get("spec_d_model", 128))),
            hidden_dim=int(color_raw.get("hidden_dim", 256)),
            n_layers=int(color_raw.get("n_layers", 2)),
            dropout=float(color_raw.get("dropout", 0.0)),
        )
    ).to(device)
    color_encoder.load_state_dict(ckpt["color_encoder"], strict=True)

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
    conditioner.load_state_dict(ckpt["conditioner"], strict=True)

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
    predictor_state = ckpt.get("rgb_condition_predictor", ckpt.get("cond_predictor"))
    if predictor_state is None:
        raise KeyError("checkpoint is missing rgb_condition_predictor")
    rgb_predictor.load_state_dict(predictor_state, strict=True)

    model_raw = cfg.get("model", {})
    denoiser = MambaDenoiser(
        DenoiserConfig(
            in_channels=int(model_raw.get("in_channels", 3)),
            hidden_dim=int(model_raw.get("hidden_dim", 128)),
            n_layers=int(model_raw.get("n_layers", 4)),
            dropout=float(model_raw.get("dropout", 0.0)),
            cond_dim=conditioner.cond_dim,
        )
    ).to(device)
    denoiser.load_state_dict(ckpt["denoiser"], strict=True)

    diff = cfg.get("diffusion", {})
    schedule = DiffusionSchedule(
        DiffusionConfig(
            T=int(diff.get("T", 200)),
            beta_0=float(diff.get("beta_0", 1e-4)),
            beta_T=float(diff.get("beta_T", 0.02)),
        ),
        device=device,
    )
    for module in (color_encoder, conditioner, rgb_predictor, denoiser):
        module.eval()
    return {
        "cfg": cfg,
        "color_encoder": color_encoder,
        "conditioner": conditioner,
        "rgb_predictor": rgb_predictor,
        "denoiser": denoiser,
        "schedule": schedule,
    }


def _resolve_condition(
    bundle: Dict[str, object],
    x0: torch.Tensor,
    obs_mask: torch.Tensor,
    batch: Optional[Dict[str, torch.Tensor]],
    condition: str,
    device: torch.device,
) -> torch.Tensor:
    c_deg = _last_observed_color(x0, obs_mask)
    z_color = bundle["color_encoder"](c_deg)
    if condition == "rgb_only":
        return bundle["rgb_predictor"](z_color)
    if condition == "full":
        if batch is None:
            raise ValueError("full condition requires Raman/XRD batch data")
        return bundle["conditioner"](
            z_color,
            batch.get("raman").to(device) if "raman" in batch else None,
            batch.get("xrd").to(device) if "xrd" in batch else None,
            raman_peaks=batch.get("raman_peaks").to(device) if "raman_peaks" in batch else None,
            xrd_peaks=batch.get("xrd_peaks").to(device) if "xrd_peaks" in batch else None,
        )
    raise ValueError(f"unknown condition: {condition}")


def _sample_batch(
    bundle: Dict[str, object],
    x0: torch.Tensor,
    obs_mask: torch.Tensor,
    cond: torch.Tensor,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    return sample_with_confidence(
        bundle["denoiser"], bundle["schedule"], x0, obs_mask, cond,
        num_samples=int(num_samples),
    )


def _apply_rts_by_sequence(
    predictions: np.ndarray,
    variances: np.ndarray,
    exp_ids: np.ndarray,
    patch_ids: np.ndarray,
    times: np.ndarray,
    process_std_lab: float,
    measurement_floor_lab: float,
) -> np.ndarray:
    out = predictions.copy()
    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx, key in enumerate(zip(exp_ids.tolist(), patch_ids.tolist())):
        groups.setdefault((int(key[0]), int(key[1])), []).append(idx)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda i: int(times[i]))
        smoothed, _ = kalman_rts_smooth_lab(
            predictions[ordered],
            variances[ordered],
            process_std_lab=process_std_lab,
            measurement_floor_lab=measurement_floor_lab,
        )
        out[ordered] = smoothed
    return out


@torch.no_grad()
def evaluate_test(
    bundle: Dict[str, object],
    test_npz: str,
    device: torch.device,
    condition: str = "rgb_only",
    num_samples: int = 20,
    apply_rts: bool = True,
    batch_size: int = 64,
) -> Dict[str, float]:
    ds = PigmentNPZDataset(test_npz)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    lab_norm = LabNorm()

    pred_list, var_list, gt_list = [], [], []
    exp_list, patch_list, time_list = [], [], []
    confidence_list = []
    for batch in loader:
        x0 = batch["x0"].to(device)
        obs_mask = batch["mask"].to(device)
        cond = _resolve_condition(bundle, x0, obs_mask, batch, condition, device)
        mean_lab, std_lab, sample_info = _sample_batch(
            bundle, x0, obs_mask, cond, num_samples
        )
        gt_lab = lab_norm.denormalize(x0[:, 0, :].detach().cpu().numpy())
        pred_list.append(mean_lab)
        var_list.append(std_lab ** 2)
        gt_list.append(gt_lab)
        confidence_list.extend([float(sample_info["conf_diffusion"])] * x0.shape[0])

        b = x0.shape[0]
        exp_list.extend(batch.get("exp_id", torch.zeros(b, dtype=torch.long)).cpu().numpy().tolist())
        patch_list.extend(batch.get("patch_id", torch.arange(b, dtype=torch.long)).cpu().numpy().tolist())
        time_list.extend(batch.get("t", torch.arange(b, dtype=torch.long)).cpu().numpy().tolist())

    pred = np.concatenate(pred_list, axis=0)
    var = np.concatenate(var_list, axis=0)
    gt = np.concatenate(gt_list, axis=0)

    infer_cfg = bundle["cfg"].get("inference", {})
    if apply_rts and len(pred) > 1:
        pred = _apply_rts_by_sequence(
            pred,
            var,
            np.asarray(exp_list),
            np.asarray(patch_list),
            np.asarray(time_list),
            process_std_lab=float(infer_cfg.get("kalman_process_std_lab", 2.0)),
            measurement_floor_lab=float(infer_cfg.get("kalman_measurement_floor_lab", 1.0)),
        )

    delta_e = np.asarray([delta_e2000(p, g) for p, g in zip(pred, gt)], dtype=np.float64)
    return {
        "deltaE2000_mean": float(delta_e.mean()),
        "deltaE2000_std": float(delta_e.std()),
        "num_samples_K": int(num_samples),
        "kalman_rts": bool(apply_rts),
        "condition": condition,
        "confidence_mean": float(np.mean(confidence_list)) if confidence_list else float("nan"),
    }


@torch.no_grad()
def infer_single_rgb(
    bundle: Dict[str, object],
    rgb: np.ndarray,
    device: torch.device,
    num_samples: int = 20,
) -> Dict[str, object]:
    """RGB-only deployment inference for one degraded color."""
    rgb = np.asarray(rgb, dtype=np.float32).reshape(3)
    lab = rgb_to_lab(rgb[None, :])[0]
    norm = LabNorm()
    x0_lab = np.stack([lab, lab], axis=0)[None, ...]
    x0 = torch.from_numpy(norm.normalize(x0_lab).astype(np.float32)).to(device)
    # X0=[c_ori,c_deg], O=[0,1]: degraded color is fixed, target color sampled.
    obs_mask = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], device=device)
    cond = _resolve_condition(bundle, x0, obs_mask, None, "rgb_only", device)
    mean_lab, std_lab, info = _sample_batch(bundle, x0, obs_mask, cond, num_samples)
    mean_lab = mean_lab[0]
    std_lab = std_lab[0]
    pred_rgb = lab_to_rgb(mean_lab[None, :])[0]
    return {
        "rgb": pred_rgb.tolist(),
        "lab": mean_lab.tolist(),
        "std_lab": std_lab.tolist(),
        "confidence": float(info["conf_diffusion"]),
        "num_samples_K": int(num_samples),
        "condition": "rgb_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rgb", default="", help="R,G,B in 0-255")
    parser.add_argument("--test_npz", default="")
    parser.add_argument("--condition", choices=["rgb_only", "full"], default="rgb_only")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--no_rts", action="store_true", help="disable RTS only for ablation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundle = load_checkpoint(args.ckpt, device)
    infer_cfg = bundle["cfg"].get("inference", {})
    k = int(args.num_samples if args.num_samples is not None else infer_cfg.get("num_samples", 20))

    if args.rgb:
        rgb = np.asarray([float(v) for v in args.rgb.split(",")], dtype=np.float32)
        if rgb.size != 3:
            raise ValueError("--rgb must be 'R,G,B'")
        print(json.dumps(infer_single_rgb(bundle, rgb, device, k), ensure_ascii=False, indent=2))
        return

    if args.test_npz:
        use_rts = bool(infer_cfg.get("use_kalman_rts_for_ordered_aging_sequences", True)) and not args.no_rts
        stats = evaluate_test(
            bundle,
            args.test_npz,
            device,
            condition=args.condition,
            num_samples=k,
            apply_rts=use_rts,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    raise ValueError("provide --rgb or --test_npz")


if __name__ == "__main__":
    main()
