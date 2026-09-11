# -*- coding: utf-8 -*-
"""
color_prior_generator.py - 颜色先验与置信度生成器

功能说明：
---------
本模块整合LUT颜色映射和缺失区域填充，生成高质量的颜色先验图和置信度图。

核心功能：
----------
1. 对已知像素区域应用LUT三线性插值进行颜色映射
2. 对Mask缺失区域使用多尺度cv2.inpaint进行填充
3. 融合LUT置信度和修复置信度，生成最终置信度图

多尺度修复策略：
----------------
- 如果 Mask 区域占比 > 30%：先下采样图像进行修复，再上采样回原分辨率
- 这样可以保证大面积缺失时的全局颜色一致性

置信度融合公式：
---------------
Conf_final = α * Conf_LUT + β * Conf_Inpaint

其中：
- α (alpha): LUT置信度权重，默认0.7
- β (beta): 修复置信度权重，默认0.3
- Mask区域的 Conf_Inpaint 显著低于已知区域

Author: Auto-generated for BrushNet Integration
"""

import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple

# 导入LUT处理器
import sys
import os
# 添加当前目录到路径以支持相对导入
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from lut_processor import LUTProcessor


class ColorPriorGenerator:
    """
    颜色先验与置信度生成器
    
    该类负责：
    1. 加载并应用LUT进行颜色映射
    2. 对缺失区域进行多尺度修复填充
    3. 生成融合的置信度图
    
    Attributes:
        lut (LUTProcessor): LUT处理器实例
        alpha (float): LUT置信度权重
        beta (float): 修复置信度权重
        inpaint_method (int): cv2.inpaint 方法 (TELEA 或 NS)
        large_mask_threshold (float): 大面积缺失阈值（默认0.3）
        inpaint_conf_known (float): 已知区域的修复置信度
        inpaint_conf_inpainted (float): 修复区域的修复置信度
    """
    
    def __init__(
        self,
        lut_path: str,
        alpha: float = 0.7,
        beta: float = 0.3,
        inpaint_method: str = 'telea',
        large_mask_threshold: float = 0.3,
        inpaint_conf_known: float = 1.0,
        inpaint_conf_inpainted: float = 0.3,
        inpaint_mask_dilate: int = 3,
        lut_delta_gain: float = 1.0
    ):
        """
        初始化颜色先验生成器
        
        Args:
            lut_path: LUT `.npz` 文件路径
            alpha: LUT置信度权重 (默认0.7)
            beta: 修复置信度权重 (默认0.3)
            inpaint_method: 修复方法 'telea' 或 'ns' (默认'telea')
            large_mask_threshold: 大面积缺失阈值 (默认0.3, 即30%)
            inpaint_conf_known: 已知区域的修复置信度 (默认1.0)
            inpaint_conf_inpainted: 修复区域的修复置信度 (默认0.3)
        """
        # 加载LUT处理器
        self.lut = LUTProcessor(lut_path)
        
        # 置信度融合参数
        self.alpha = alpha
        self.beta = beta
        
        # 多尺度修复参数
        self.large_mask_threshold = large_mask_threshold
        # Internal safety dilation for cv2.inpaint masks. This does not change
        # public hole/known semantics; it only prevents white/gray placeholder
        # borders and downsampled thin mask strokes from being used as known colors.
        self.inpaint_mask_dilate = max(0, int(inpaint_mask_dilate))
        self.lut_delta_gain = max(0.0, float(lut_delta_gain))
        
        # 修复方法选择
        if inpaint_method.lower() == 'telea':
            self.inpaint_method = cv2.INPAINT_TELEA
        elif inpaint_method.lower() == 'ns':
            self.inpaint_method = cv2.INPAINT_NS
        else:
            raise ValueError(f"不支持的修复方法: {inpaint_method}，请使用 'telea' 或 'ns'")
        
        # 修复置信度参数
        self.inpaint_conf_known = inpaint_conf_known
        self.inpaint_conf_inpainted = inpaint_conf_inpainted
        
        print(f"[ColorPriorGenerator] 初始化完成:")
        print(f"  - α (LUT权重): {self.alpha}")
        print(f"  - β (修复权重): {self.beta}")
        print(f"  - 修复方法: {inpaint_method}")
        print(f"  - 大面积阈值: {self.large_mask_threshold * 100:.0f}%")
        print(f"  - inpaint mask dilate: {self.inpaint_mask_dilate}px")
        print(f"  - LUT delta gain: {self.lut_delta_gain:.3f}")

    def feather_blend(
        self,
        original: np.ndarray,
        transformed: np.ndarray,
        mask: np.ndarray,
        feather_radius: int = 7,
    ) -> np.ndarray:
        """Blend ``transformed`` into ``original`` inside ``mask`` with a soft edge."""
        mask_float = (mask.astype(np.float32) / 255.0)
        if feather_radius > 0:
            kernel = 2 * feather_radius + 1
            mask_float = cv2.GaussianBlur(mask_float, (kernel, kernel), sigmaX=0, sigmaY=0)
        mask_float = np.clip(mask_float, 0.0, 1.0)
        mask_3ch = mask_float[:, :, np.newaxis]
        blended = (
            original.astype(np.float32) * (1.0 - mask_3ch)
            + transformed.astype(np.float32) * mask_3ch
        )
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def _apply_lut_delta_gain(self, image: np.ndarray, lut_image: np.ndarray) -> np.ndarray:
        """Amplify LUT chroma delta while keeping the source luminance stable."""
        image_u8 = np.clip(image, 0, 255).astype(np.uint8)
        lut_u8 = np.clip(lut_image, 0, 255).astype(np.uint8)
        if abs(self.lut_delta_gain - 1.0) < 1e-6:
            return lut_u8

        orig_lab = cv2.cvtColor(
            cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR),
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)
        lut_lab = cv2.cvtColor(
            cv2.cvtColor(lut_u8, cv2.COLOR_RGB2BGR),
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)
        out_lab = orig_lab.copy()
        out_lab[..., 1] = np.clip(
            orig_lab[..., 1] + (lut_lab[..., 1] - orig_lab[..., 1]) * self.lut_delta_gain,
            0,
            255,
        )
        out_lab[..., 2] = np.clip(
            orig_lab[..., 2] + (lut_lab[..., 2] - orig_lab[..., 2]) * self.lut_delta_gain,
            0,
            255,
        )
        out_bgr = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    def build_target(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        mode: str = 'full',
        feather_radius: int = 7,
    ) -> np.ndarray:
        """Build the training target using the same rules as the mural dataset.

        Args:
            image: [H, W, 3] uint8 reference degraded image.
            mask: [H, W] uint8 hole mask, 255 = hole.
            mode: 'full' or 'partial'.
            feather_radius: soft blending radius used by partial mode.

        Returns:
            [H, W, 3] uint8 target image aligned with training supervision.
        """
        lut_only, _ = self.lut.trilinear_interpolate(image)
        lut_only = self._apply_lut_delta_gain(image, lut_only)
        if mode == 'full':
            return lut_only
        if mode == 'partial':
            return self.feather_blend(image, lut_only, mask, feather_radius=feather_radius)
        raise ValueError(f"未知的目标模式: {mode}")
    
    def _calculate_mask_ratio(self, mask: np.ndarray) -> float:
        """
        计算Mask区域占比
        
        Args:
            mask: [H, W] uint8 掩码（255=缺失, 0=已知）
            
        Returns:
            ratio: 缺失区域占比 (0.0 ~ 1.0)
        """
        total_pixels = mask.size
        masked_pixels = np.sum(mask > 127)
        return masked_pixels / total_pixels

    def _normalize_hole_mask(self, mask: np.ndarray) -> np.ndarray:
        """Return a uint8 hole mask where 255=hole and 0=known."""
        if mask is None:
            return mask
        mask_uint8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask.copy()
        if mask_uint8.max() <= 1:
            mask_uint8 = mask_uint8 * 255
        return ((mask_uint8 > 127).astype(np.uint8) * 255)

    def _expand_inpaint_mask(self, mask: np.ndarray) -> np.ndarray:
        """Dilate the hole mask used by cv2.inpaint, preserving the public mask semantics.

        The final returned color_prior is still written only to the original hole
        pixels.  This expanded mask is only an internal safety mask for inpaint:
        it prevents white/gray placeholder borders, anti-aliased mask edges, and
        thin holes lost during downsampling from being used as known colors.
        """
        mask_uint8 = self._normalize_hole_mask(mask)
        if mask_uint8 is None or self.inpaint_mask_dilate <= 0 or not np.any(mask_uint8):
            return mask_uint8
        k = 2 * self.inpaint_mask_dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask_uint8, kernel, iterations=1)

    def _prepare_lut_input(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        为 LUT/Lab 颜色先验准备 mask-aware 输入。

        mural 的 observed_degraded 在 hole 区通常被填成白色/黑色；这些像素不是
        真实观测，不能参与 Lab 亮度、色差平滑和 LUT 置信度计算。这里仅在进入
        现有 color-prior 主算法之前，用已有 inpaint 逻辑把 hole 预填成来自
        known 区的上下文颜色；known 区保持原图不变。

        Args:
            image: [H, W, 3] uint8/float RGB 图像。
            mask: [H, W] uint8 掩码，255/1=hole，0=known。

        Returns:
            image_for_lut: [H, W, 3] uint8 RGB 图像，known 区与输入一致，
                hole 区为上下文预填结果。
        """
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        if mask is None:
            return image_uint8.copy()

        mask_uint8 = self._normalize_hole_mask(mask)
        if not np.any(mask_uint8):
            return image_uint8.copy()

        # cv2.inpaint 会忽略 mask 内原始像素值；这里用已有多尺度策略只生成
        # LUT/Lab 的安全输入，不改变返回接口，也不对 hole 做硬阈值改色。
        filled = self.multi_scale_inpaint(image_uint8, mask_uint8)
        image_for_lut = image_uint8.copy()
        mask_bool = mask_uint8 > 0
        image_for_lut[mask_bool] = filled[mask_bool]
        return image_for_lut
    
    def multi_scale_inpaint(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        inpaint_radius: int = 3
    ) -> np.ndarray:
        """
        多尺度修复策略
        
        策略说明：
        ---------
        1. 计算Mask区域占比
        2. 如果 > large_mask_threshold：
           - 将图像下采样到1/2尺寸
           - 在低分辨率上进行修复
           - 上采样回原分辨率
           - 再次进行精细修复
        3. 否则直接使用cv2.inpaint
        
        Args:
            image: [H, W, 3] uint8 输入图像 (BGR或RGB)
            mask: [H, W] uint8 掩码 (255=缺失, 0=已知)
            inpaint_radius: 修复半径 (默认3)
            
        Returns:
            inpainted: [H, W, 3] uint8 修复后的图像
        """
        H, W = image.shape[:2]
        # Use a slightly expanded mask internally. Final callers still write
        # back only to the original hole mask, but cv2.inpaint must not see
        # white/gray placeholder borders or downsampled thin strokes as known.
        mask = self._expand_inpaint_mask(mask)
        if mask is None or not np.any(mask):
            return image.copy()
        mask_ratio = self._calculate_mask_ratio(mask)
        
        # 确保mask是uint8类型
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        if mask_ratio > self.large_mask_threshold:
            # ============================================================
            # 大面积缺失：使用多尺度策略
            # ============================================================
            
            # 第一阶段：在1/2分辨率上修复（获取全局颜色一致性）
            scale_factor = 0.5
            small_H, small_W = int(H * scale_factor), int(W * scale_factor)
            
            # 下采样
            small_image = cv2.resize(image, (small_W, small_H), interpolation=cv2.INTER_AREA)
            # INTER_NEAREST may drop thin/anti-aliased hole strokes while
            # downsampling, turning white placeholders into false known pixels.
            # Resize with area coverage and keep any covered pixel as hole.
            small_mask = cv2.resize(mask, (small_W, small_H), interpolation=cv2.INTER_AREA)
            small_mask = ((small_mask > 0).astype(np.uint8) * 255)
            
            # 低分辨率修复
            small_inpainted = cv2.inpaint(
                small_image, small_mask, 
                inpaintRadius=inpaint_radius * 2,  # 低分辨率用更大半径
                flags=self.inpaint_method
            )
            
            # 上采样回原分辨率
            upscaled = cv2.resize(small_inpainted, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # 将上采样结果填充到原图的mask区域
            merged = image.copy()
            mask_bool = mask > 127
            merged[mask_bool] = upscaled[mask_bool]
            
            # 第二阶段：在原分辨率上精细修复边缘
            # 创建边缘mask（只修复边缘过渡区域）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated = cv2.dilate(mask, kernel, iterations=1)
            eroded = cv2.erode(mask, kernel, iterations=1)
            edge_mask = dilated - eroded
            
            # 精细修复边缘
            final_inpainted = cv2.inpaint(
                merged, edge_mask,
                inpaintRadius=inpaint_radius,
                flags=self.inpaint_method
            )
            
            return final_inpainted
        
        else:
            # ============================================================
            # 小面积缺失：直接修复
            # ============================================================
            return cv2.inpaint(
                image, mask,
                inpaintRadius=inpaint_radius,
                flags=self.inpaint_method
            )
    
    def get_spatial_confidence(self, mask: np.ndarray) -> np.ndarray:
        """
        计算空间置信度（基于距离变换）
        
        边界区域置信度高，中心区域置信度低
        
        Args:
            mask: [H, W] uint8 掩码 (255=缺失, 0=已知)
            
        Returns:
            spatial_conf: [H, W] float32 空间置信度 (0.1 ~ 1.0)
        """
        # 距离变换：计算每个像素到最近边界的距离
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_dist = float(dist_map.max()) + 1e-8
        
        # 归一化距离
        norm_dist = dist_map / max_dist
        
        # 置信度 = 1 - 归一化距离（边界高，中心低）
        spatial_conf = 1.0 - norm_dist
        spatial_conf = np.clip(spatial_conf, 0.1, 1.0)  # 最小0.1
        
        # 已知区域置信度 = 1.0
        spatial_conf[mask == 0] = 1.0
        
        return spatial_conf.astype(np.float32)
    
    def smooth_delta_bilateral(
        self, 
        delta: np.ndarray, 
        sigma_color: float = 10.0,
        sigma_space: float = 8.0
    ) -> np.ndarray:
        """
        对色差进行双边滤波平滑
        
        Args:
            delta: [H, W] float32 色差图
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差
            
        Returns:
            smoothed: [H, W] float32 平滑后的色差
        """
        delta = delta.astype(np.float32)
        return cv2.bilateralFilter(delta, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    def smooth_delta_multiscale(
        self, 
        delta: np.ndarray, 
        down: int = 1, 
        sigma: float = 2.0
    ) -> np.ndarray:
        """
        多尺度平滑（参考t6.py/t7.py）
        
        Args:
            delta: [H, W] float32 色差图
            down: 下采样层数
            sigma: 高斯模糊标准差
            
        Returns:
            smoothed: [H, W] float32 平滑后的色差
        """
        x = delta.astype(np.float32)
        for _ in range(max(0, down)):
            x = cv2.pyrDown(x)
        x = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)
        for _ in range(max(0, down)):
            x = cv2.pyrUp(x)
        
        # 确保尺寸匹配
        H, W = delta.shape[:2]
        x = x[:H, :W]
        return x
    
    def generate(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        method: str = 'fast',
        debug: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        生成颜色先验和置信度图
        
        Args:
            image: [H, W, 3] uint8 输入RGB图像 (0-255)
            mask: [H, W] uint8 掩码 (255=缺失需要修复, 0=已知)
            method: 生成方法
                - 'fast': 快速版（训练用）- 双边滤波平滑
                - 'quality': 高质量版（推理用）- 多尺度+导向滤波
            debug: 是否返回调试信息
            
        Returns:
            result: dict 包含以下键：
                - 'color_prior': [H, W, 3] float32 颜色先验图 (0-255)
                - 'confidence': [H, W] float32 融合置信度图 (0-1)
                - 'conf_lut': [H, W] float32 LUT原始置信度 (0-1)
                - 'conf_inpaint': [H, W] float32 修复区域置信度 (0-1)
        """
        if method == 'fast':
            return self.generate_fast(image, mask, debug)
        elif method == 'quality':
            return self.generate_quality(image, mask, debug)
        else:
            raise ValueError(f"未知的生成方法: {method}，请使用 'fast' 或 'quality'")
    
    def generate_fast(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        debug: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        快速版颜色先验生成（训练用）
        
        特点：
        - Lab 空间处理，保持原图亮度 L
        - 双边滤波平滑色差 delta(a,b)
        - 空间置信度（边界高，中心低）
        - cv2.inpaint 填充 mask 区域
        """
        # 验证输入
        assert image.ndim == 3 and image.shape[2] == 3
        assert mask.ndim == 2
        assert image.shape[:2] == mask.shape
        
        H, W = image.shape[:2]
        mask = self._normalize_hole_mask(mask)
        mask_bool = mask > 127
        image_for_lut = self._prepare_lut_input(image, mask)
        
        # ============================================================
        # Step 1: RGB 转 Lab
        # 注意：Lab/LUT 输入必须 mask-aware，避免 observed white/black hole
        # 污染亮度 L、色差平滑和 LUT confidence。
        # ============================================================
        img_bgr = cv2.cvtColor(image_for_lut, cv2.COLOR_RGB2BGR)
        orig_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_orig = orig_lab[..., 0]
        
        # ============================================================
        # Step 2: LUT 映射
        # ============================================================
        color_prior_lut, conf_lut = self.lut.trilinear_interpolate(image_for_lut)
        
        # 转换到 Lab 空间
        lut_bgr = cv2.cvtColor(np.clip(color_prior_lut, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        mapped_lab = cv2.cvtColor(lut_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # 保持原图亮度
        mapped_lab[..., 0] = L_orig
        
        # ============================================================
        # Step 3: 计算色差 delta(a, b) 并平滑
        # ============================================================
        da = mapped_lab[..., 1] - orig_lab[..., 1]
        db = mapped_lab[..., 2] - orig_lab[..., 2]
        
        # 双边滤波平滑
        da_smooth = self.smooth_delta_bilateral(da, sigma_color=10.0, sigma_space=8.0)
        db_smooth = self.smooth_delta_bilateral(db, sigma_color=10.0, sigma_space=8.0)
        
        # ============================================================
        # Step 4: 应用平滑后的色差生成新的 Lab 图像
        # ============================================================
        da_smooth = da_smooth * self.lut_delta_gain
        db_smooth = db_smooth * self.lut_delta_gain
        new_lab = orig_lab.copy()
        new_lab[..., 1] = np.clip(orig_lab[..., 1] + da_smooth, 0, 255)
        new_lab[..., 2] = np.clip(orig_lab[..., 2] + db_smooth, 0, 255)
        
        # 转回 RGB
        new_bgr = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        color_prior_full = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # ============================================================
        # Step 5: 对 mask 区域进行修复填充
        # ============================================================
        color_prior_uint8 = np.clip(color_prior_full, 0, 255).astype(np.uint8)
        color_prior_inpainted = self.multi_scale_inpaint(color_prior_uint8, mask)
        color_prior_inpainted = color_prior_inpainted.astype(np.float32)
        
        # 组合：已知区域用 LUT 结果，mask 区域用修复结果
        color_prior = color_prior_full.copy()
        color_prior[mask_bool] = color_prior_inpainted[mask_bool]
        
        # ============================================================
        # Step 6: 计算置信度（空间置信度 × LUT置信度加权）
        # ============================================================
        spatial_conf = self.get_spatial_confidence(mask)
        
        confidence = np.zeros_like(conf_lut)
        
        # 已知区域：置信度 = 1.0
        confidence[~mask_bool] = 1.0
        
        # 缺失区域：空间置信度 × (α * LUT置信度 + β * inpaint置信度)
        confidence[mask_bool] = spatial_conf[mask_bool] * (
            self.alpha * conf_lut[mask_bool] + 
            self.beta * self.inpaint_conf_inpainted
        )
        
        confidence = np.clip(confidence, 0, 1)
        
        # 构建返回结果
        result = {
            'color_prior': color_prior,
            'confidence': confidence,
            'conf_lut': conf_lut,
            'conf_inpaint': spatial_conf
        }
        
        if debug:
            result['color_prior_lut_raw'] = color_prior_lut
            result['color_prior_lut'] = color_prior_full
            result['color_prior_inpainted'] = color_prior_inpainted
            result['image_for_lut'] = image_for_lut
            result['mask_ratio'] = self._calculate_mask_ratio(mask)
            result['spatial_conf'] = spatial_conf
            result['inpaint_mask'] = self._expand_inpaint_mask(mask)
            result['inpaint_mask_ratio'] = self._calculate_mask_ratio(result['inpaint_mask'])

        return result
    
    def generate_quality(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        debug: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        高质量版颜色先验生成（推理用）
        
        特点：
        - Lab 空间处理，保持原图亮度 L
        - 多尺度预平滑 + 导向滤波（如果可用）
        - 空间置信度（边界高，中心低）
        - KMeans 调色板软分配（可选扩展，当前使用 inpaint）
        """
        # 验证输入
        assert image.ndim == 3 and image.shape[2] == 3
        assert mask.ndim == 2
        assert image.shape[:2] == mask.shape
        
        H, W = image.shape[:2]
        mask = self._normalize_hole_mask(mask)
        mask_bool = mask > 127
        mask_01 = (mask / 255.0).astype(np.float32)
        image_for_lut = self._prepare_lut_input(image, mask)
        
        # ============================================================
        # Step 1: RGB 转 Lab
        # 注意：Lab/LUT 输入必须 mask-aware，避免 observed white/black hole
        # 污染亮度 L、色差平滑和 LUT confidence。
        # ============================================================
        img_bgr = cv2.cvtColor(image_for_lut, cv2.COLOR_RGB2BGR)
        orig_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_orig = orig_lab[..., 0]
        guide = (L_orig / (L_orig.max() + 1e-6)).astype(np.float32)
        
        # ============================================================
        # Step 2: LUT 映射
        # ============================================================
        color_prior_lut, conf_lut = self.lut.trilinear_interpolate(image_for_lut)
        
        lut_bgr = cv2.cvtColor(np.clip(color_prior_lut, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        mapped_lab = cv2.cvtColor(lut_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        mapped_lab[..., 0] = L_orig
        
        # ============================================================
        # Step 3: 计算色差并进行多尺度 + 边缘感知平滑
        # ============================================================
        da = mapped_lab[..., 1] - orig_lab[..., 1]
        db = mapped_lab[..., 2] - orig_lab[..., 2]
        
        # 多尺度预平滑
        da = self.smooth_delta_multiscale(da, down=1, sigma=2.0)
        db = self.smooth_delta_multiscale(db, down=1, sigma=2.0)
        
        # One-level Gaussian-pyramid smoothing (sigma=2.0)
        # followed by bilateral filtering (d=-1, sigmaColor=10, sigmaSpace=16).
        da_smooth = self.smooth_delta_bilateral(da, sigma_color=10.0, sigma_space=16.0)
        db_smooth = self.smooth_delta_bilateral(db, sigma_color=10.0, sigma_space=16.0)
        
        # ============================================================
        # Step 4: 应用平滑后的色差
        # ============================================================
        da_smooth = da_smooth * self.lut_delta_gain
        db_smooth = db_smooth * self.lut_delta_gain
        new_lab = orig_lab.copy()
        new_lab[..., 1] = np.clip(orig_lab[..., 1] + da_smooth, 0, 255)
        new_lab[..., 2] = np.clip(orig_lab[..., 2] + db_smooth, 0, 255)
        
        new_bgr = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        color_prior_full = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # ============================================================
        # Step 5: 对 mask 区域进行修复填充
        # ============================================================
        color_prior_uint8 = np.clip(color_prior_full, 0, 255).astype(np.uint8)
        color_prior_inpainted = self.multi_scale_inpaint(color_prior_uint8, mask)
        color_prior_inpainted = color_prior_inpainted.astype(np.float32)
        
        color_prior = color_prior_full.copy()
        color_prior[mask_bool] = color_prior_inpainted[mask_bool]
        
        # ============================================================
        # Step 6: 计算置信度
        # ============================================================
        spatial_conf = self.get_spatial_confidence(mask)
        
        confidence = np.zeros_like(conf_lut)
        confidence[~mask_bool] = 1.0
        confidence[mask_bool] = spatial_conf[mask_bool] * (
            self.alpha * conf_lut[mask_bool] + 
            self.beta * self.inpaint_conf_inpainted
        )
        confidence = np.clip(confidence, 0, 1)
        
        result = {
            'color_prior': color_prior,
            'confidence': confidence,
            'conf_lut': conf_lut,
            'conf_inpaint': spatial_conf
        }
        
        if debug:
            result['color_prior_lut_raw'] = color_prior_lut
            result['color_prior_lut'] = color_prior_full
            result['color_prior_inpainted'] = color_prior_inpainted
            result['image_for_lut'] = image_for_lut
            result['mask_ratio'] = self._calculate_mask_ratio(mask)
            result['spatial_conf'] = spatial_conf
            result['inpaint_mask'] = self._expand_inpaint_mask(mask)
            result['inpaint_mask_ratio'] = self._calculate_mask_ratio(result['inpaint_mask'])

        return result
    
    def generate_tensor(
        self,
        image_tensor,
        mask_tensor,
        device=None,
        method: str = 'fast',
        debug: bool = False,
    ):
        """
        对PyTorch张量生成颜色先验和置信度图
        
        Args:
            image_tensor: [B, 3, H, W] torch.Tensor，范围 [0, 1]
            mask_tensor: [B, 1, H, W] torch.Tensor，范围 [0, 1]（1=缺失）
            device: 输出设备
            
        Returns:
            color_prior: [B, 3, H, W] torch.Tensor，范围 [0, 1]
            confidence: [B, 1, H, W] torch.Tensor，范围 [0, 1]
            debug_tensors: 仅当 debug=True 时返回，包含中间量张量
        """
        import torch
        
        if device is None:
            device = image_tensor.device
        
        B = image_tensor.shape[0]
        
        # 转换到numpy
        img_np = image_tensor.detach().cpu().numpy()  # [B, 3, H, W]
        img_np = np.transpose(img_np, (0, 2, 3, 1))    # [B, H, W, 3]
        img_np = (img_np * 255).astype(np.uint8)
        
        mask_np = mask_tensor.detach().cpu().numpy()   # [B, 1, H, W]
        mask_np = mask_np[:, 0, :, :]                  # [B, H, W]
        mask_np = (mask_np * 255).astype(np.uint8)
        
        # 批量处理
        color_priors = []
        confidences = []
        debug_buffers = {
            'color_prior_lut_raw': [],
            'color_prior_lut': [],
            'color_prior_inpainted': [],
            'image_for_lut': [],
            'conf_lut': [],
            'conf_inpaint': [],
            'inpaint_mask': [],
        } if debug else None
        
        for i in range(B):
            result = self.generate(img_np[i], mask_np[i], method=method, debug=debug)
            color_priors.append(result['color_prior'])
            confidences.append(result['confidence'])
            if debug:
                for key in debug_buffers.keys():
                    debug_buffers[key].append(result[key])
        
        # 转回tensor
        color_prior = np.stack(color_priors, axis=0)   # [B, H, W, 3]
        color_prior = np.transpose(color_prior, (0, 3, 1, 2)) / 255.0  # [B, 3, H, W]
        
        confidence = np.stack(confidences, axis=0)     # [B, H, W]
        confidence = confidence[:, np.newaxis, :, :]   # [B, 1, H, W]
        
        color_prior = torch.from_numpy(color_prior.astype(np.float32)).to(device)
        confidence = torch.from_numpy(confidence.astype(np.float32)).to(device)
        
        if not debug:
            return color_prior, confidence

        debug_tensors = {}
        for key, values in debug_buffers.items():
            stacked = np.stack(values, axis=0)
            if stacked.ndim == 4:
                stacked = np.transpose(stacked, (0, 3, 1, 2))
                if key.startswith('color_prior') or key == 'image_for_lut':
                    stacked = stacked / 255.0
            elif stacked.ndim == 3:
                stacked = stacked[:, np.newaxis, :, :]
            debug_tensors[key] = torch.from_numpy(stacked.astype(np.float32)).to(device)

        return color_prior, confidence, debug_tensors



# =============================================================================
# 单元测试
# =============================================================================
if __name__ == "__main__":
    import sys
    
    # 测试用占位LUT路径
    LUT_PATH = "./example_lut.npz"
    
    print("=" * 60)
    print("Color Prior Generator 单元测试")
    print("=" * 60)
    
    # 确保LUT文件存在（复用lut_processor的测试文件）
    if not os.path.exists(LUT_PATH):
        print(f"\n[警告] LUT文件未找到: {LUT_PATH}")
        print("请先运行 lut_processor.py 创建测试LUT")
        sys.exit(1)
    
    # 测试初始化
    print("\n[测试1] 初始化生成器...")
    gen = ColorPriorGenerator(
        LUT_PATH,
        alpha=0.7,
        beta=0.3,
        inpaint_method='telea'
    )
    
    # 创建测试数据
    print("\n[测试2] 创建测试数据...")
    test_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # 小面积mask
    small_mask = np.zeros((256, 256), dtype=np.uint8)
    small_mask[100:150, 100:150] = 255  # 约10%缺失
    
    # 大面积mask
    large_mask = np.zeros((256, 256), dtype=np.uint8)
    large_mask[50:200, 50:200] = 255  # 约35%缺失
    
    # 测试小面积修复
    print("\n[测试3] 小面积修复...")
    result_small = gen.generate(test_img, small_mask, debug=True)
    print(f"  Mask占比: {result_small['mask_ratio']*100:.1f}%")
    print(f"  颜色先验形状: {result_small['color_prior'].shape}")
    print(f"  置信度形状: {result_small['confidence'].shape}")
    print(f"  置信度范围: [{result_small['confidence'].min():.4f}, {result_small['confidence'].max():.4f}]")
    
    # 测试大面积修复（多尺度）
    print("\n[测试4] 大面积修复（多尺度策略）...")
    result_large = gen.generate(test_img, large_mask, debug=True)
    print(f"  Mask占比: {result_large['mask_ratio']*100:.1f}%")
    print(f"  颜色先验形状: {result_large['color_prior'].shape}")
    print(f"  置信度形状: {result_large['confidence'].shape}")
    
    # 验证置信度在mask区域较低
    mask_bool = large_mask > 127
    conf_in_mask = result_large['confidence'][mask_bool].mean()
    conf_out_mask = result_large['confidence'][~mask_bool].mean()
    print(f"  Mask内平均置信度: {conf_in_mask:.4f}")
    print(f"  Mask外平均置信度: {conf_out_mask:.4f}")
    
    assert conf_in_mask < conf_out_mask, "修复区域置信度应低于已知区域"
    
    print("\n✓ 所有测试通过!")
