# -*- coding: utf-8 -*-
"""壁画推理数据集。

用法:
    在 options/test/ir-sde-brushnet.yml 中配置:
        datasets:
          test:
            mode: mural_inference
            dataroot_degraded: /path/to/degraded
            dataroot_mask: /path/to/mask
            dataroot_GT: /path/to/gt    # 可选

Mask 语义:
    - 输入 mask 白色/255 表示待修复区域
    - 输出 mask_hole: 1 表示待修复区域
    - 输出 mask_known: 1 表示已知区域

命名规则:
    - 样本主键始终使用输入图像 stem
    - mask 默认匹配 <stem>_mask.*
    - 若找不到 <stem>_mask.*，兼容回退到 <stem>.*
    - GT / color_prior / confidence 保持同 stem 匹配
"""

import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.utils.data as data

try:
    from color_prior_generator import ColorPriorGenerator
except ImportError:
    from ..color_prior_generator import ColorPriorGenerator

READ_IMAGE_MSG = '无法读取图像'
READ_MASK_MSG = '无法读取掩码'
READ_CONFIDENCE_MSG = '无法读取置信度图'
NO_DEGRADED_MSG = 'dataroot_degraded 下没有可用图像。'
NO_MASK_MSG = 'dataroot_mask 下没有可用掩码。'
MISSING_MASK_MSG = '缺少对应掩码'


IMG_EXTENSIONS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tif',
    '.tiff',
    '.JPG',
    '.JPEG',
    '.PNG',
    '.BMP',
    '.TIF',
    '.TIFF',
}


def _list_image_paths(root: Optional[str]) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            ext = os.path.splitext(filename)[1]
            if ext in IMG_EXTENSIONS:
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


def _build_stem_map(paths: List[str]) -> Dict[str, str]:
    mapping = {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        mapping[stem] = path
    return mapping


def _load_rgb_image(path: str, target_hw=None) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"{READ_IMAGE_MSG}: {path}")
    if target_hw is not None and image.shape[:2] != target_hw:
        image = cv2.resize(image, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def _load_mask(path: str, target_hw) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"{READ_MASK_MSG}: {path}")
    if mask.shape[:2] != target_hw:
        mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return mask


def _load_confidence(path: str, target_hw) -> np.ndarray:
    confidence = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if confidence is None:
        raise FileNotFoundError(f"{READ_CONFIDENCE_MSG}: {path}")
    if confidence.shape[:2] != target_hw:
        confidence = cv2.resize(
            confidence,
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return confidence.astype(np.float32) / 255.0


class MuralInferenceDataset(data.Dataset):
    """用于最终增强版推理的真实数据集。"""

    def __init__(self, opt: dict):
        super().__init__()
        self.opt = opt
        self.gt_mode = opt.get('gt_mode', 'partial')
        self.degraded_root = opt.get('dataroot_degraded')
        self.mask_root = opt.get('dataroot_mask')
        self.refine_mask_from_observed_white = bool(opt.get('refine_mask_from_observed_white', False))
        self.mask_white_refine_threshold = float(opt.get('mask_white_refine_threshold', 0.95))
        self.mask_white_refine_dilate = int(opt.get('mask_white_refine_dilate', 6))
        self.mask_white_refine_expand = int(opt.get('mask_white_refine_expand', 0))
        self._mask_refine_log_count = 0

        self.degraded_paths = _list_image_paths(self.degraded_root)
        if not self.degraded_paths:
            raise FileNotFoundError(NO_DEGRADED_MSG)

        self.mask_map = _build_stem_map(_list_image_paths(self.mask_root))
        if not self.mask_map:
            raise FileNotFoundError(NO_MASK_MSG)

        self.gt_map = _build_stem_map(_list_image_paths(opt.get('dataroot_GT')))
        self.color_prior_map = _build_stem_map(_list_image_paths(opt.get('dataroot_color_prior')))
        self.confidence_map = _build_stem_map(_list_image_paths(opt.get('dataroot_confidence')))
        self.color_prior_generator = None
        lut_path = opt.get('lut_path')
        if lut_path:
            self.color_prior_generator = ColorPriorGenerator(
                lut_path=lut_path,
                alpha=opt.get('lut_alpha', 0.7),
                beta=opt.get('lut_beta', 0.3),
                inpaint_method=opt.get('lut_inpaint_method', 'telea'),
                inpaint_mask_dilate=opt.get('prior_inpaint_mask_dilate', 3),
            )

    def _build_eval_gt(self, gt: np.ndarray, mask_hole: np.ndarray):
        mode = str(self.gt_mode).lower()
        if mode in {'paired_true', 'raw', 'gt', 'paired_gt'}:
            return gt, False
        if self.color_prior_generator is None:
            raise RuntimeError(
                f"gt_mode={self.gt_mode!r} requires lut_path / ColorPriorGenerator in the inference dataset."
            )
        if mode in {'full', 'gt_full', 'full_gt', 'paired_full', 'paired_gt_full'}:
            return self.color_prior_generator.build_target(gt, (mask_hole * 255.0).round().astype(np.uint8), mode='full'), True
        if mode in {'partial', 'gt_partial', 'partial_gt', 'paired_partial', 'paired_gt_partial'}:
            return self.color_prior_generator.build_target(gt, (mask_hole * 255.0).round().astype(np.uint8), mode='partial'), True
        raise ValueError(
            f"Unsupported inference gt_mode={self.gt_mode!r}; expected paired_true|full|partial."
        )

    def _refine_mask_from_observed(self, degraded: np.ndarray, mask_hole: np.ndarray, stem: str) -> np.ndarray:
        if not self.refine_mask_from_observed_white:
            return mask_hole

        mask_bin = (mask_hole > 0.5).astype(np.uint8)
        if mask_bin.max() == 0:
            return mask_hole

        threshold = float(np.clip(self.mask_white_refine_threshold, 0.0, 1.0))
        white_like = (
            (degraded[:, :, 0] >= threshold)
            & (degraded[:, :, 1] >= threshold)
            & (degraded[:, :, 2] >= threshold)
        ).astype(np.uint8)

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
                    '[MuralInferenceDataset] refined mask from observed white boundary: '
                    f'stem={stem} ratio {original_ratio:.4f}->{refined_ratio:.4f} '
                    f'threshold={self.mask_white_refine_threshold:.3f} '
                    f'dilate={self.mask_white_refine_dilate} expand={self.mask_white_refine_expand}'
                )
                self._mask_refine_log_count += 1

        return refined.astype(np.float32)

    def __len__(self) -> int:
        return len(self.degraded_paths)

    def _resolve_mask_path(self, stem: str) -> str:
        preferred_keys = [f'{stem}_mask', stem]
        for key in preferred_keys:
            path = self.mask_map.get(key)
            if path is not None:
                return path

        example_keys = sorted(self.mask_map.keys())[:10]
        raise FileNotFoundError(
            f"{MISSING_MASK_MSG}: image stem='{stem}', expected mask stem '{stem}_mask' "
            f"(preferred) or '{stem}' in '{self.mask_root}'. Example keys: {example_keys}"
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        degraded_path = self.degraded_paths[index]
        stem = os.path.splitext(os.path.basename(degraded_path))[0]

        degraded = _load_rgb_image(degraded_path)
        height, width = degraded.shape[:2]

        mask_path = self._resolve_mask_path(stem)
        mask_hole = _load_mask(mask_path, (height, width))
        mask_hole = self._refine_mask_from_observed(degraded, mask_hole, stem)
        mask_known = 1.0 - mask_hole

        sample = {
            'degraded': torch.from_numpy(np.transpose(degraded, (2, 0, 1))).float(),
            'mask_hole': torch.from_numpy(mask_hole[None, ...]).float(),
            'mask_known': torch.from_numpy(mask_known[None, ...]).float(),
            'degraded_path': degraded_path,
            'mask_path': mask_path,
            'stem': stem,
            'gt_mode': self.gt_mode,
        }

        if stem in self.gt_map:
            gt_raw = _load_rgb_image(self.gt_map[stem], (height, width))
            gt_eval_u8, gt_is_target_like = self._build_eval_gt(
                (gt_raw * 255.0).round().astype(np.uint8),
                mask_hole,
            )
            gt_eval = gt_eval_u8.astype(np.float32) / 255.0
            sample['GT'] = torch.from_numpy(np.transpose(gt_eval, (2, 0, 1))).float()
            sample['GT_true'] = torch.from_numpy(np.transpose(gt_raw, (2, 0, 1))).float()
            sample['gt_already_target_like'] = bool(gt_is_target_like)
            sample['GT_path'] = self.gt_map[stem]

        if stem in self.color_prior_map:
            color_prior = _load_rgb_image(self.color_prior_map[stem], (height, width))
            sample['color_prior'] = torch.from_numpy(np.transpose(color_prior, (2, 0, 1))).float()
            sample['color_prior_path'] = self.color_prior_map[stem]

        if stem in self.confidence_map:
            confidence = _load_confidence(self.confidence_map[stem], (height, width))
            sample['confidence'] = torch.from_numpy(confidence[None, ...]).float()
            sample['confidence_path'] = self.confidence_map[stem]

        return sample
