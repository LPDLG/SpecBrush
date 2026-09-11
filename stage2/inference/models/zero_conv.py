# -*- coding: utf-8 -*-
"""
zero_conv.py - 零初始化卷积层

功能说明：
---------
Zero-Convolution 是 ControlNet/BrushNet 中用于特征注入的关键组件。
其核心思想是：将卷积层的权重和偏置初始化为0，使得训练初期
辅助网络对主网络没有任何影响，随着训练进行逐渐学习有效的特征映射。

原理：
------
1. 初始化权重和偏置为0
2. 训练初期输出全为0，不影响主网络
3. 通过梯度下降逐渐学习有意义的特征映射
4. 保证训练稳定性

使用场景：
----------
- BrushNet 特征注入到主 UNet
- ControlNet 条件控制
- 任何需要渐进式特征融合的场景

Author: Auto-generated for BrushNet Integration
"""

import torch
import torch.nn as nn
from typing import Optional


class ZeroConv2d(nn.Module):
    """
    零初始化2D卷积层
    
    该层在初始化时将权重和偏置设为0，确保训练初期不影响主网络。
    随着训练进行，通过反向传播逐渐学习有效的特征映射。
    
    Attributes:
        conv (nn.Conv2d): 标准2D卷积层
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        """
        初始化零卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小 (默认1)
            stride: 步长 (默认1)
            padding: 填充 (默认0)
            bias: 是否使用偏置 (默认True)
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        # 核心：将权重和偏置初始化为0
        nn.init.zeros_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, C_in, H, W] 输入特征
            
        Returns:
            out: [B, C_out, H, W] 输出特征
        """
        return self.conv(x)


class ZeroConv1d(nn.Module):
    """
    零初始化1D卷积层（用于时间嵌入等1D特征）
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 1,
        bias: bool = True
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            bias=bias
        )
        
        nn.init.zeros_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ZeroLinear(nn.Module):
    """
    零初始化线性层
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        nn.init.zeros_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_zero_conv(
    in_channels: int, 
    out_channels: int,
    conv_type: str = '2d',
    kernel_size: int = 1,
    **kwargs
) -> nn.Module:
    """
    工厂函数：创建零初始化卷积层
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        conv_type: 卷积类型 '2d', '1d', 或 'linear'
        kernel_size: 卷积核大小
        **kwargs: 传递给卷积层的其他参数
        
    Returns:
        零初始化的卷积层
    """
    if conv_type == '2d':
        return ZeroConv2d(in_channels, out_channels, kernel_size, **kwargs)
    elif conv_type == '1d':
        return ZeroConv1d(in_channels, out_channels, kernel_size, **kwargs)
    elif conv_type == 'linear':
        return ZeroLinear(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"不支持的卷积类型: {conv_type}")


# =============================================================================
# 单元测试
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Zero Convolution 单元测试")
    print("=" * 60)
    
    # 测试 ZeroConv2d
    print("\n[测试1] ZeroConv2d...")
    zero_conv = ZeroConv2d(64, 128, kernel_size=1)
    x = torch.randn(2, 64, 32, 32)
    y = zero_conv(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {y.shape}")
    print(f"  输出均值: {y.mean().item():.6f} (应接近0)")
    print(f"  输出标准差: {y.std().item():.6f} (应接近0)")
    
    assert y.shape == (2, 128, 32, 32), "输出形状错误"
    assert abs(y.mean().item()) < 1e-6, "初始输出应接近0"
    
    # 测试 ZeroLinear
    print("\n[测试2] ZeroLinear...")
    zero_linear = ZeroLinear(256, 512)
    x = torch.randn(2, 256)
    y = zero_linear(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {y.shape}")
    print(f"  输出均值: {y.mean().item():.6f} (应接近0)")
    
    assert y.shape == (2, 512), "输出形状错误"
    assert abs(y.mean().item()) < 1e-6, "初始输出应接近0"
    
    # 测试工厂函数
    print("\n[测试3] make_zero_conv 工厂函数...")
    conv_2d = make_zero_conv(32, 64, conv_type='2d')
    linear = make_zero_conv(128, 256, conv_type='linear')
    
    print(f"  2D卷积类型: {type(conv_2d).__name__}")
    print(f"  线性层类型: {type(linear).__name__}")
    
    print("\n✓ 所有测试通过!")
