# -*- coding: utf-8 -*-
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from color_prior_generator import ColorPriorGenerator
except ImportError:
    from ..color_prior_generator import ColorPriorGenerator

IMG_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
    '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF',
}


def _list_image_paths(root: Optional[str]) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1] in IMG_EXTENSIONS:
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


def _build_stem_map(paths: List[str]) -> Dict[str, str]:
    return {os.path.splitext(os.path.basename(path))[0]: path for path in paths}


def _stem_candidates(stem: str) -> List[str]:
    candidates = [stem]
    suffixes = ['_mask_merge', '_merge', '_masked', '_degraded', '_input']
    for suffix in suffixes:
        if stem.endswith(suffix):
            candidates.append(stem[: -len(suffix)])
    dedup: List[str] = []
    for item in candidates:
        if item and item not in dedup:
            dedup.append(item)
    return dedup


class MuralPairedInpaintingDataset(Dataset):
    """Paired mural dataset.

    Loads three aligned assets by stem:
    - GT / full reference image from dataroot_GT
    - observed current-domain image (hole already whitened) from dataroot_degraded
    - mask from dataroot_mask

    This removes the old synthetic target generation and the old GT/mask shuffle.
    """

    def __init__(
        self,
        opt: dict,
        lut_path: str,
        gt_mode: str = 'paired_true',
        prior_method: str = 'fast',
        debug_mode: bool = False,
        debug_dir: str = './debug_logs',
        lut_alpha: float = 0.7,
        lut_beta: float = 0.3,
        lut_inpaint_method: str = 'telea',
        lut_delta_gain: float = 1.0,
    ):
        super().__init__()
        self.opt = opt
        self.gt_mode = gt_mode
        self.prior_method = prior_method.lower()
        self.debug_mode = bool(debug_mode)
        self.debug_dir = debug_dir
        self.GT_size = int(opt.get('GT_size', 256))
        self.use_flip = bool(opt.get('use_flip', True))
        self.use_rot = bool(opt.get('use_rot', True))
        self.max_crop_retry = int(opt.get('max_crop_retry', 20))
        self.min_hole_ratio = float(opt.get('min_hole_ratio', 0.005))
        self.max_hole_ratio = float(opt.get('max_hole_ratio', 0.80))
        self._crop_retry_fail_count = 0
        self.refine_mask_from_observed_white = bool(opt.get('refine_mask_from_observed_white', False))
        self.mask_white_refine_threshold = float(opt.get('mask_white_refine_threshold', 0.95))
        self.mask_white_refine_dilate = int(opt.get('mask_white_refine_dilate', 6))
        self.mask_white_refine_expand = int(opt.get('mask_white_refine_expand', 0))
        self._mask_refine_log_count = 0

        self.color_prior_gen = ColorPriorGenerator(
            lut_path=lut_path,
            alpha=lut_alpha,
            beta=lut_beta,
            inpaint_method=lut_inpaint_method,
            lut_delta_gain=max(0.0, float(lut_delta_gain)),
            inpaint_mask_dilate=opt.get('prior_inpaint_mask_dilate', opt.get('inpaint_mask_dilate', 3)),
        )

        self.gt_root = opt.get('dataroot_GT', '')
        self.degraded_root = opt.get('dataroot_degraded', '')
        self.mask_root = opt.get('dataroot_mask', '')

        self.gt_map = _build_stem_map(_list_image_paths(self.gt_root))
        self.degraded_map = _build_stem_map(_list_image_paths(self.degraded_root))
        self.mask_map = _build_stem_map(_list_image_paths(self.mask_root))

        if not self.gt_map:
            raise FileNotFoundError(f'No GT images found under {self.gt_root}')
        if not self.degraded_map:
            raise FileNotFoundError(f'No degraded images found under {self.degraded_root}')
        if not self.mask_map:
            raise FileNotFoundError(f'No masks found under {self.mask_root}')

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                'No paired training samples found. '
                f'GT={self.gt_root}, degraded={self.degraded_root}, mask={self.mask_root}'
            )

        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)

        print('[MuralPairedInpaintingDataset] initialized')
        print(f'  - samples: {len(self.samples)}')
        print(f'  - GT root: {self.gt_root}')
        print(f'  - degraded root: {self.degraded_root}')
        print(f'  - mask root: {self.mask_root}')
        print(f'  - gt_mode: {self.gt_mode}')
        print(
            '  - refine_mask_from_observed_white: '
            f'{self.refine_mask_from_observed_white} '
            f'(threshold={self.mask_white_refine_threshold:.3f}, '
            f'dilate={self.mask_white_refine_dilate}, expand={self.mask_white_refine_expand})'
        )

    def _build_supervision_gt(self, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Build the supervision target from the paired GT itself.

        This keeps paired supervision semantics intact while allowing the whole
        chain to move into a recolored target domain when requested.
        """
        mode = str(self.gt_mode).lower()
        if mode in {'paired_true', 'raw', 'gt', 'paired_gt'}:
            return gt
        if mode in {'full', 'gt_full', 'full_gt', 'paired_full', 'paired_gt_full'}:
            return self.color_prior_gen.build_target(gt, mask, mode='full')
        if mode in {'partial', 'gt_partial', 'partial_gt', 'paired_partial', 'paired_gt_partial'}:
            return self.color_prior_gen.build_target(gt, mask, mode='partial')
        raise ValueError(
            f"Unsupported paired gt_mode={self.gt_mode!r}; "
            "expected paired_true|full|partial (or gt_* aliases)."
        )

    def _resolve_mask_path(self, stem: str) -> Optional[str]:
        for base in _stem_candidates(stem):
            for key in (f'{base}_mask', base):
                path = self.mask_map.get(key)
                if path is not None:
                    return path
        return None

    def _resolve_gt_path(self, stem: str) -> Optional[str]:
        for key in _stem_candidates(stem):
            path = self.gt_map.get(key)
            if path is not None:
                return path
        return None

    def _build_samples(self) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        missing_gt = 0
        missing_mask = 0
        for degraded_stem, degraded_path in sorted(self.degraded_map.items()):
            gt_path = self._resolve_gt_path(degraded_stem)
            mask_path = self._resolve_mask_path(degraded_stem)
            if gt_path is None:
                missing_gt += 1
                continue
            if mask_path is None:
                missing_mask += 1
                continue
            samples.append({
                'stem': degraded_stem,
                'degraded_path': degraded_path,
                'gt_path': gt_path,
                'mask_path': mask_path,
            })
        print(f'[MuralPairedInpaintingDataset] paired samples={len(samples)}, missing_gt={missing_gt}, missing_mask={missing_mask}')
        return samples

    def _load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: str, target_hw: Tuple[int, int]) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Failed to read mask: {path}')
        if mask.shape[:2] != target_hw:
            mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
        return ((mask > 127).astype(np.uint8) * 255)

    def _refine_mask_from_observed(self, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.refine_mask_from_observed_white:
            return mask

        mask_bin = (mask > 127).astype(np.uint8)
        if mask_bin.max() == 0:
            return mask

        threshold_uint8 = int(np.clip(round(self.mask_white_refine_threshold * 255.0), 0, 255))
        white_like = (degraded[:, :, 0] >= threshold_uint8) & (
            degraded[:, :, 1] >= threshold_uint8
        ) & (degraded[:, :, 2] >= threshold_uint8)
        white_like = white_like.astype(np.uint8)

        if self.mask_white_refine_dilate > 0:
            k = 2 * self.mask_white_refine_dilate + 1
            kernel = np.ones((k, k), dtype=np.uint8)
            near_hole = cv2.dilate(mask_bin, kernel, iterations=1)
        else:
            near_hole = mask_bin

        refined = np.maximum(mask_bin, white_like * near_hole)
        if self.mask_white_refine_expand > 0:
            k = 2 * self.mask_white_refine_expand + 1
            kernel = np.ones((k, k), dtype=np.uint8)
            refined = cv2.dilate(refined, kernel, iterations=1)

        if self._mask_refine_log_count < 5:
            original_ratio = float(mask_bin.mean())
            refined_ratio = float(refined.mean())
            if refined_ratio > original_ratio + 1e-4:
                print(
                    '[MuralPairedInpaintingDataset] refined mask from observed white boundary: '
                    f'ratio {original_ratio:.4f} -> {refined_ratio:.4f}, '
                    f'threshold={self.mask_white_refine_threshold:.3f}, '
                    f'dilate={self.mask_white_refine_dilate}, expand={self.mask_white_refine_expand}'
                )
                self._mask_refine_log_count += 1

        return (refined * 255).astype(np.uint8)

    def _resize_triplet_if_needed(self, gt: np.ndarray, degraded: np.ndarray, mask: np.ndarray):
        h, w = gt.shape[:2]
        crop_size = self.GT_size
        if h >= crop_size and w >= crop_size:
            return gt, degraded, mask
        scale = max(crop_size / h, crop_size / w) * 1.1
        new_h, new_w = int(h * scale), int(w * scale)
        gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        degraded = cv2.resize(degraded, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return gt, degraded, ((mask > 127).astype(np.uint8) * 255)

    def _random_crop_triplet(self, gt: np.ndarray, degraded: np.ndarray, mask: np.ndarray):
        gt, degraded, mask = self._resize_triplet_if_needed(gt, degraded, mask)
        h, w = gt.shape[:2]
        crop_size = self.GT_size

        best = None
        best_ratio = None
        best_score = float('inf')

        def eval_crop(y: int, x: int):
            gt_crop = gt[y:y+crop_size, x:x+crop_size]
            degraded_crop = degraded[y:y+crop_size, x:x+crop_size]
            mask_crop = mask[y:y+crop_size, x:x+crop_size]
            hole_ratio = float(np.mean(mask_crop > 127))
            if hole_ratio < self.min_hole_ratio:
                score = self.min_hole_ratio - hole_ratio
            elif hole_ratio > self.max_hole_ratio:
                score = hole_ratio - self.max_hole_ratio
            else:
                score = 0.0
            return gt_crop, degraded_crop, mask_crop, hole_ratio, score

        def try_crop(y: int, x: int):
            nonlocal best, best_ratio, best_score
            y = int(np.clip(y, 0, h - crop_size))
            x = int(np.clip(x, 0, w - crop_size))
            gt_crop, degraded_crop, mask_crop, hole_ratio, score = eval_crop(y, x)
            if score <= 0.0:
                return gt_crop, degraded_crop, mask_crop, True
            if score < best_score:
                best = (gt_crop, degraded_crop, mask_crop)
                best_ratio = hole_ratio
                best_score = score
            return None, None, None, False

        for _ in range(max(1, self.max_crop_retry)):
            y = random.randint(0, h - crop_size)
            x = random.randint(0, w - crop_size)
            gt_crop, degraded_crop, mask_crop, ok = try_crop(y, x)
            if ok:
                return gt_crop, degraded_crop, mask_crop

        hole_ys, hole_xs = np.where(mask > 127)
        if hole_ys.size > 0:
            for _ in range(max(1, self.max_crop_retry)):
                idx = random.randrange(hole_ys.size)
                cy, cx = int(hole_ys[idx]), int(hole_xs[idx])
                y = cy - random.randint(0, crop_size - 1)
                x = cx - random.randint(0, crop_size - 1)
                gt_crop, degraded_crop, mask_crop, ok = try_crop(y, x)
                if ok:
                    return gt_crop, degraded_crop, mask_crop

        self._crop_retry_fail_count += 1
        if self.debug_mode or self._crop_retry_fail_count <= 5:
            print(
                '[MuralPairedInpaintingDataset] crop retry fallback(best after guided): '
                f'hole_ratio={best_ratio:.4f}, expected '
                f'[{self.min_hole_ratio:.4f}, {self.max_hole_ratio:.4f}], '
                f'max_crop_retry={self.max_crop_retry}, guided_retry={self.max_crop_retry}'
            )
        return best

    def _augment_all(self, gt, degraded, mask, color_prior, confidence, conf_lut):
        if self.use_flip and random.random() > 0.5:
            gt = np.fliplr(gt).copy()
            degraded = np.fliplr(degraded).copy()
            mask = np.fliplr(mask).copy()
            color_prior = np.fliplr(color_prior).copy()
            confidence = np.fliplr(confidence).copy()
            conf_lut = np.fliplr(conf_lut).copy()
        if self.use_rot:
            k = random.randint(0, 3)
            if k > 0:
                gt = np.rot90(gt, k).copy()
                degraded = np.rot90(degraded, k).copy()
                mask = np.rot90(mask, k).copy()
                color_prior = np.rot90(color_prior, k).copy()
                confidence = np.rot90(confidence, k).copy()
                conf_lut = np.rot90(conf_lut, k).copy()
        return gt, degraded, mask, color_prior, confidence, conf_lut

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image.copy())

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        gt = self._load_image(sample['gt_path'])
        degraded = self._load_image(sample['degraded_path'])
        if degraded.shape[:2] != gt.shape[:2]:
            degraded = cv2.resize(degraded, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = self._load_mask(sample['mask_path'], gt.shape[:2])
        mask = self._refine_mask_from_observed(degraded, mask)

        gt, degraded, mask = self._random_crop_triplet(gt, degraded, mask)

        prior_result = self.color_prior_gen.generate(degraded, mask, method=self.prior_method)
        color_prior = np.clip(prior_result['color_prior'], 0, 255).astype(np.uint8)
        confidence = prior_result['confidence'].astype(np.float32)
        conf_lut = prior_result['conf_lut'].astype(np.float32)

        gt, degraded, mask, color_prior, confidence, conf_lut = self._augment_all(
            gt, degraded, mask, color_prior, confidence, conf_lut
        )

        gt_true = gt.copy()
        gt_supervision = self._build_supervision_gt(gt_true, mask)

        gt_gray = cv2.cvtColor(gt_supervision, cv2.COLOR_RGB2GRAY)
        gt_edge = cv2.Canny(gt_gray, 50, 150)

        return {
            'degraded': self._to_tensor(degraded),
            'degraded_full': self._to_tensor(degraded),
            'GT': self._to_tensor(gt_supervision),
            'gt': self._to_tensor(gt_supervision),
            'GT_true': self._to_tensor(gt_true),
            'gt_true': self._to_tensor(gt_true),
            'GT_gray': torch.from_numpy((gt_gray.astype(np.float32) / 255.0)[None, ...]),
            'GT_edge': torch.from_numpy((gt_edge.astype(np.float32) / 255.0)[None, ...]),
            'mask': torch.from_numpy((mask.astype(np.float32) / 255.0)[None, ...]),
            'color_prior': self._to_tensor(color_prior),
            'confidence': torch.from_numpy(confidence[None, ...].astype(np.float32)),
            'conf_lut': torch.from_numpy(conf_lut[None, ...].astype(np.float32)),
            'mode': 'paired_true',
            'path': sample['degraded_path'],
            'GT_path': sample['gt_path'],
            'mask_path': sample['mask_path'],
        }

    def __len__(self) -> int:
        return len(self.samples)
