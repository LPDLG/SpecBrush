# -*- coding: utf-8 -*-
"""
lut_processor.py - 颜色映射表（LUT）三线性插值处理模块

功能说明：
---------
本模块实现对预训练颜色映射表（LUT `.npz` 文件）的加载与应用。
核心算法使用三线性插值（Trilinear Interpolation），避免最近邻查找导致的颜色断层。

LUT 数据结构：
-------------
- grid: [N,] float32 - 网格坐标点（0~255等间隔划分）
- lut_rgb: [N, N, N, 3] uint8 - RGB颜色映射
- lut_lab: [N, N, N, 3] float32 - Lab颜色映射  
- lut_conf: [N, N, N] float32 - 置信度映射（0~1）
- lut_std: [N, N, N] float32 - 预测标准差

轴序：lut[R_index, G_index, B_index]

Author: Auto-generated for BrushNet Integration
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict


class LUTProcessor:
    """
    颜色映射表处理器 - 三线性插值实现
    
    该类加载预训练的颜色LUT，并通过三线性插值将输入RGB图像映射到
    预测的原始颜色空间，同时输出对应的置信度图。
    
    Attributes:
        grid (np.ndarray): [N,] 网格坐标点
        lut_rgb (np.ndarray): [N,N,N,3] RGB颜色映射表
        lut_conf (np.ndarray): [N,N,N] 置信度映射表
        grid_size (int): 网格大小（默认33）
        grid_step (float): 网格步长（255.0 / (grid_size - 1)）
    """
    
    def __init__(self, lut_path: str):
        """
        初始化LUT处理器
        
        Args:
            lut_path: LUT `.npz` 文件路径
            
        Raises:
            FileNotFoundError: 如果LUT文件不存在
            KeyError: 如果LUT文件缺少必要的键
        """
        if not os.path.exists(lut_path):
            raise FileNotFoundError(
                f"[LUTProcessor] LUT文件不存在: {lut_path}\n"
                f"请确保 LUT `.npz` 文件已放置在正确位置。"
            )
        
        # 加载LUT数据
        data = np.load(lut_path, allow_pickle=True)
        
        # 验证必要的键
        required_keys = ['grid', 'lut_rgb', 'lut_conf']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"[LUTProcessor] LUT文件缺少必要的键: {key}")
        
        # 存储LUT数据
        self.grid = data['grid'].astype(np.float32)  # [N,] 网格坐标
        self.lut_rgb = data['lut_rgb']                # [N,N,N,3] RGB映射
        self.lut_conf = data['lut_conf'].astype(np.float32)  # [N,N,N] 置信度
        
        # 可选数据
        self.lut_lab = data.get('lut_lab', None)      # Lab颜色空间（可选）
        self.lut_std = data.get('lut_std', None)      # 标准差（可选）
        
        # 计算网格参数
        self.grid_size = len(self.grid)  # 通常为33
        self.grid_step = self.grid[1] - self.grid[0]  # 网格步长
        
        # 加载元数据（如果存在）
        self.meta = data.get('meta', {})
        if isinstance(self.meta, np.ndarray):
            self.meta = self.meta.item() if self.meta.ndim == 0 else {}
        
        print(f"[LUTProcessor] 加载完成:")
        print(f"  - 网格大小: {self.grid_size}")
        print(f"  - 网格范围: [{self.grid[0]:.1f}, {self.grid[-1]:.1f}]")
        print(f"  - 网格步长: {self.grid_step:.4f}")
        print(f"  - LUT形状: {self.lut_rgb.shape}")
    
    def trilinear_interpolate(
        self, 
        rgb_image: np.ndarray,
        return_lab: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对输入RGB图像进行三线性插值颜色映射
        
        三线性插值原理：
        ---------------
        对于输入像素 (r, g, b)，首先计算其在LUT网格中的连续坐标，
        然后找到包围它的8个网格顶点，通过加权平均得到插值结果。
        
        权重计算基于距离各顶点的归一化距离。
        
        Args:
            rgb_image: [H, W, 3] uint8 输入RGB图像（0-255）
            return_lab: 是否同时返回Lab颜色空间映射结果
            
        Returns:
            color_prior: [H, W, 3] float32 颜色先验图（0-255范围）
            confidence: [H, W] float32 置信度图（0-1范围）
            
        Tensor Shape 变化：
        ------------------
        输入: [H, W, 3] uint8
        归一化网格坐标: [H, W, 3] float32
        8个顶点索引: 各 [H, W, 3] int32
        插值权重: [H, W, 8] float32
        输出颜色: [H, W, 3] float32
        输出置信度: [H, W] float32
        """
        # 确保输入格式正确
        assert rgb_image.ndim == 3 and rgb_image.shape[2] == 3, \
            f"输入图像形状应为 [H, W, 3]，实际为 {rgb_image.shape}"
        
        H, W = rgb_image.shape[:2]
        rgb = rgb_image.astype(np.float32)  # [H, W, 3]
        
        # ============================================================
        # Step 1: 计算在LUT网格中的连续坐标
        # ============================================================
        # LUT网格从 grid[0] 到 grid[-1]（通常是0到255）
        # 归一化到 [0, grid_size-1] 范围
        # coords[..., c] 表示通道 c 在网格中的连续索引
        coords = rgb / self.grid_step  # [H, W, 3]
        
        # 裁剪到有效范围，防止索引越界
        coords = np.clip(coords, 0, self.grid_size - 1 - 1e-5)
        
        # ============================================================
        # Step 2: 计算8个包围顶点的索引
        # ============================================================
        # 下界索引 (floor)
        idx_low = np.floor(coords).astype(np.int32)  # [H, W, 3]
        # 上界索引 (ceil)，注意不要超过最大索引
        idx_high = np.minimum(idx_low + 1, self.grid_size - 1)  # [H, W, 3]
        
        # 分离R, G, B通道索引
        r0, g0, b0 = idx_low[..., 0], idx_low[..., 1], idx_low[..., 2]
        r1, g1, b1 = idx_high[..., 0], idx_high[..., 1], idx_high[..., 2]
        
        # ============================================================
        # Step 3: 计算插值权重
        # ============================================================
        # 小数部分作为权重
        frac = coords - idx_low.astype(np.float32)  # [H, W, 3]
        fr, fg, fb = frac[..., 0], frac[..., 1], frac[..., 2]  # 各 [H, W]
        
        # 计算8个顶点的权重（三线性插值公式）
        # w_ijk = (1-fr if i=0 else fr) * (1-fg if j=0 else fg) * (1-fb if k=0 else fb)
        w000 = (1 - fr) * (1 - fg) * (1 - fb)  # [H, W]
        w001 = (1 - fr) * (1 - fg) * fb
        w010 = (1 - fr) * fg * (1 - fb)
        w011 = (1 - fr) * fg * fb
        w100 = fr * (1 - fg) * (1 - fb)
        w101 = fr * (1 - fg) * fb
        w110 = fr * fg * (1 - fb)
        w111 = fr * fg * fb
        
        # ============================================================
        # Step 4: 从LUT采样并加权求和
        # ============================================================
        # 采样8个顶点的RGB值
        v000 = self.lut_rgb[r0, g0, b0].astype(np.float32)  # [H, W, 3]
        v001 = self.lut_rgb[r0, g0, b1].astype(np.float32)
        v010 = self.lut_rgb[r0, g1, b0].astype(np.float32)
        v011 = self.lut_rgb[r0, g1, b1].astype(np.float32)
        v100 = self.lut_rgb[r1, g0, b0].astype(np.float32)
        v101 = self.lut_rgb[r1, g0, b1].astype(np.float32)
        v110 = self.lut_rgb[r1, g1, b0].astype(np.float32)
        v111 = self.lut_rgb[r1, g1, b1].astype(np.float32)
        
        # 加权求和得到插值结果
        color_prior = (
            w000[..., None] * v000 +
            w001[..., None] * v001 +
            w010[..., None] * v010 +
            w011[..., None] * v011 +
            w100[..., None] * v100 +
            w101[..., None] * v101 +
            w110[..., None] * v110 +
            w111[..., None] * v111
        )  # [H, W, 3]
        
        # ============================================================
        # Step 5: 采样并插值置信度
        # ============================================================
        c000 = self.lut_conf[r0, g0, b0]  # [H, W]
        c001 = self.lut_conf[r0, g0, b1]
        c010 = self.lut_conf[r0, g1, b0]
        c011 = self.lut_conf[r0, g1, b1]
        c100 = self.lut_conf[r1, g0, b0]
        c101 = self.lut_conf[r1, g0, b1]
        c110 = self.lut_conf[r1, g1, b0]
        c111 = self.lut_conf[r1, g1, b1]
        
        confidence = (
            w000 * c000 +
            w001 * c001 +
            w010 * c010 +
            w011 * c011 +
            w100 * c100 +
            w101 * c101 +
            w110 * c110 +
            w111 * c111
        )  # [H, W]
        
        # 确保输出范围正确
        color_prior = np.clip(color_prior, 0, 255)
        confidence = np.clip(confidence, 0, 1)
        
        return color_prior, confidence
    
    def apply_to_tensor(
        self, 
        rgb_tensor,
        device: Optional[str] = None
    ):
        """
        对PyTorch张量应用LUT映射（支持GPU加速）
        
        Args:
            rgb_tensor: [B, 3, H, W] torch.Tensor，范围 [0, 1]
            device: 输出设备，默认与输入相同
            
        Returns:
            color_prior: [B, 3, H, W] torch.Tensor，范围 [0, 1]
            confidence: [B, 1, H, W] torch.Tensor，范围 [0, 1]
        """
        import torch
        
        # 保存原始设备
        if device is None:
            device = rgb_tensor.device
        
        B, C, H, W = rgb_tensor.shape
        assert C == 3, f"输入通道数应为3，实际为 {C}"
        
        # 转换到numpy处理（批量）
        rgb_np = rgb_tensor.detach().cpu().numpy()  # [B, 3, H, W]
        rgb_np = np.transpose(rgb_np, (0, 2, 3, 1))  # [B, H, W, 3]
        rgb_np = (rgb_np * 255).astype(np.uint8)
        
        # 批量处理
        color_priors = []
        confidences = []
        for i in range(B):
            cp, conf = self.trilinear_interpolate(rgb_np[i])
            color_priors.append(cp)
            confidences.append(conf)
        
        # 转回tensor
        color_prior = np.stack(color_priors, axis=0)  # [B, H, W, 3]
        color_prior = np.transpose(color_prior, (0, 3, 1, 2)) / 255.0  # [B, 3, H, W]
        
        confidence = np.stack(confidences, axis=0)  # [B, H, W]
        confidence = confidence[:, np.newaxis, :, :]  # [B, 1, H, W]
        
        color_prior = torch.from_numpy(color_prior.astype(np.float32)).to(device)
        confidence = torch.from_numpy(confidence.astype(np.float32)).to(device)
        
        return color_prior, confidence


# =============================================================================
# 单元测试
# =============================================================================
if __name__ == "__main__":
    import sys
    
    # 测试用占位LUT路径
    LUT_PATH = "./example_lut.npz"
    
    print("=" * 60)
    print("LUT Processor 单元测试")
    print("=" * 60)
    
    if not os.path.exists(LUT_PATH):
        print(f"\n[警告] LUT文件未找到: {LUT_PATH}")
        print("正在创建测试用模拟LUT...")
        
        # 创建模拟LUT用于测试
        N = 33
        grid = np.linspace(0, 255, N, dtype=np.float32)
        lut_rgb = np.zeros((N, N, N, 3), dtype=np.uint8)
        lut_conf = np.ones((N, N, N), dtype=np.float32) * 0.8
        
        # 简单的恒等映射（用于测试）
        for i, r in enumerate(grid):
            for j, g in enumerate(grid):
                for k, b in enumerate(grid):
                    lut_rgb[i, j, k] = [int(r), int(g), int(b)]
        
        np.savez_compressed(
            LUT_PATH,
            grid=grid,
            lut_rgb=lut_rgb,
            lut_conf=lut_conf
        )
        print(f"已创建模拟LUT: {LUT_PATH}")
    
    # 测试加载
    print("\n[测试1] 加载LUT...")
    lut = LUTProcessor(LUT_PATH)
    
    # 测试三线性插值
    print("\n[测试2] 三线性插值...")
    test_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    color_prior, confidence = lut.trilinear_interpolate(test_img)
    
    print(f"  输入形状: {test_img.shape}")
    print(f"  颜色先验形状: {color_prior.shape}")
    print(f"  置信度形状: {confidence.shape}")
    print(f"  颜色先验范围: [{color_prior.min():.2f}, {color_prior.max():.2f}]")
    print(f"  置信度范围: [{confidence.min():.4f}, {confidence.max():.4f}]")
    
    assert color_prior.shape == (256, 256, 3), "颜色先验形状错误"
    assert confidence.shape == (256, 256), "置信度形状错误"
    assert 0 <= color_prior.min() and color_prior.max() <= 255, "颜色先验范围错误"
    assert 0 <= confidence.min() and confidence.max() <= 1, "置信度范围错误"
    
    print("\n✓ 所有测试通过!")
