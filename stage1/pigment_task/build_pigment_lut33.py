"""Build the offline 33^3 SpecBrush pigment LUT from Stage-I aging predictions.

SpecBrush sequence:
  1) RGB-only Stage-I inversion for each observed LeadAging color.
  2) K=20 posterior sampling -> mean and variance.
  3) Kalman-RTS smoothing across adjacent aging time points of each pigment run.
  4) Organize the smoothed color correspondences into a 3-D RGB LUT.

The scattered-to-grid interpolation is implemented with a stable numerical routine;
this release uses local inverse-distance weighting over the nearest observed
aging colors to form LUT vertices, while retaining posterior uncertainty as
q_lut.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from data.dataset import PigmentNPZDataset
from inference.kalman_rts import kalman_rts_smooth_lab
from inference.pipeline import _last_observed_color, _resolve_condition, load_checkpoint
from inference.uncertainty import sample_with_confidence
from utils.color_utils import LabNorm, lab_to_rgb


def _collect_aging_predictions(
    bundle: Dict[str, object],
    aging_npz: str,
    device: torch.device,
    num_samples: int,
    batch_size: int,
):
    ds = PigmentNPZDataset(aging_npz)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    norm = LabNorm()

    degraded_lab, predicted_lab, predicted_var = [], [], []
    exp_ids, patch_ids, times = [], [], []
    for batch in loader:
        x0 = batch["x0"].to(device)
        obs_mask = batch["mask"].to(device)
        c_deg_norm = _last_observed_color(x0, obs_mask)
        cond = _resolve_condition(bundle, x0, obs_mask, batch, "rgb_only", device)
        mean_lab, std_lab, _ = sample_with_confidence(
            bundle["denoiser"], bundle["schedule"], x0, obs_mask, cond,
            num_samples=num_samples,
        )
        degraded_lab.append(norm.denormalize(c_deg_norm.detach().cpu().numpy()))
        predicted_lab.append(mean_lab)
        predicted_var.append(std_lab ** 2)
        b = x0.shape[0]
        exp_ids.extend(batch.get("exp_id", torch.zeros(b, dtype=torch.long)).cpu().numpy().tolist())
        patch_ids.extend(batch.get("patch_id", torch.arange(b, dtype=torch.long)).cpu().numpy().tolist())
        times.extend(batch.get("t", torch.arange(b, dtype=torch.long)).cpu().numpy().tolist())

    deg_lab = np.concatenate(degraded_lab, axis=0).astype(np.float32)
    pred_lab = np.concatenate(predicted_lab, axis=0).astype(np.float32)
    var_lab = np.concatenate(predicted_var, axis=0).astype(np.float32)
    return (
        deg_lab,
        pred_lab,
        var_lab,
        np.asarray(exp_ids, dtype=np.int64),
        np.asarray(patch_ids, dtype=np.int64),
        np.asarray(times, dtype=np.int64),
    )


def _smooth_groups(
    pred_lab: np.ndarray,
    var_lab: np.ndarray,
    exp_ids: np.ndarray,
    patch_ids: np.ndarray,
    times: np.ndarray,
    process_std_lab: float,
    measurement_floor_lab: float,
) -> Tuple[np.ndarray, np.ndarray]:
    smoothed = pred_lab.copy()
    smoothed_var = var_lab.copy()
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i, key in enumerate(zip(exp_ids.tolist(), patch_ids.tolist())):
        groups.setdefault((int(key[0]), int(key[1])), []).append(i)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        order = sorted(indices, key=lambda i: int(times[i]))
        m, v = kalman_rts_smooth_lab(
            pred_lab[order],
            var_lab[order],
            process_std_lab=process_std_lab,
            measurement_floor_lab=measurement_floor_lab,
        )
        smoothed[order] = m
        smoothed_var[order] = v
    return smoothed, smoothed_var


def _idw_lut(
    degraded_rgb: np.ndarray,
    target_lab: np.ndarray,
    target_var_lab: np.ndarray,
    grid_size: int,
    neighbors: int,
    chunk_size: int,
):
    grid = np.linspace(0.0, 255.0, grid_size, dtype=np.float32)
    rr, gg, bb = np.meshgrid(grid, grid, grid, indexing="ij")
    queries = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)
    source = degraded_rgb.astype(np.float32)
    target = target_lab.astype(np.float32)

    # Convert Lab posterior variance to a bounded LUT-vertex reliability.
    scale = np.asarray([100.0, 128.0, 128.0], dtype=np.float32)
    std_norm = np.sqrt(np.maximum(target_var_lab, 0.0)) / scale[None, :]
    source_conf = np.exp(-np.linalg.norm(std_norm, axis=-1)).astype(np.float32)

    lut_lab = np.empty((queries.shape[0], 3), dtype=np.float32)
    lut_conf = np.empty((queries.shape[0],), dtype=np.float32)
    k = max(1, min(int(neighbors), source.shape[0]))
    eps = 1e-6

    for start in range(0, queries.shape[0], chunk_size):
        q = queries[start:start + chunk_size]
        d2 = ((q[:, None, :] - source[None, :, :]) ** 2).sum(axis=-1)
        nearest = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        d2k = np.take_along_axis(d2, nearest, axis=1)
        # Inverse-distance weights keep exact/near measured colors dominant.
        w = 1.0 / (np.sqrt(d2k) + 1.0)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), eps)
        tk = target[nearest]
        ck = source_conf[nearest]
        lut_lab[start:start + len(q)] = (w[..., None] * tk).sum(axis=1)
        lut_conf[start:start + len(q)] = (w * ck).sum(axis=1)

    lut_lab = lut_lab.reshape(grid_size, grid_size, grid_size, 3)
    lut_rgb = lab_to_rgb(lut_lab.reshape(-1, 3)).reshape(
        grid_size, grid_size, grid_size, 3
    ).astype(np.uint8)
    lut_conf = np.clip(lut_conf.reshape(grid_size, grid_size, grid_size), 0.0, 1.0)
    return grid, lut_rgb, lut_lab, lut_conf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpt/lab_raman_xrd/best_model.pt")
    parser.add_argument("--aging_npz", default="data/pigment_npz/train.npz")
    parser.add_argument("--out_npz", default="pigment_lut33.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--grid_size", type=int, default=33)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--kalman_process_std_lab", type=float, default=2.0)
    parser.add_argument("--kalman_measurement_floor_lab", type=float, default=1.0)
    args = parser.parse_args()

    if int(args.num_samples) != 20:
        print(f"[warning] default setting is K=20; requested K={args.num_samples}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundle = load_checkpoint(args.ckpt, device)

    deg_lab, pred_lab, var_lab, exp_ids, patch_ids, times = _collect_aging_predictions(
        bundle, args.aging_npz, device, int(args.num_samples), int(args.batch_size)
    )
    pred_smooth, var_smooth = _smooth_groups(
        pred_lab,
        var_lab,
        exp_ids,
        patch_ids,
        times,
        process_std_lab=float(args.kalman_process_std_lab),
        measurement_floor_lab=float(args.kalman_measurement_floor_lab),
    )
    degraded_rgb = lab_to_rgb(deg_lab).astype(np.float32)
    grid, lut_rgb, lut_lab, lut_conf = _idw_lut(
        degraded_rgb,
        pred_smooth,
        var_smooth,
        grid_size=int(args.grid_size),
        neighbors=int(args.neighbors),
        chunk_size=int(args.chunk_size),
    )
    np.savez_compressed(
        args.out_npz,
        grid=grid,
        lut_rgb=lut_rgb,
        lut_lab=lut_lab,
        lut_conf=lut_conf,
        meta=np.asarray({
            "axis_order": "RGB",
            "num_samples_K": int(args.num_samples),
            "kalman_rts": True,
            "aging_npz": str(args.aging_npz),
            "checkpoint": str(args.ckpt),
            "grid_size": int(args.grid_size),
            "interpolator": "local_inverse_distance",
            "neighbors": int(args.neighbors),
        }, dtype=object),
    )
    print(f"saved released LUT: {args.out_npz} shape={lut_rgb.shape} K={args.num_samples} RTS=True")


if __name__ == "__main__":
    main()
