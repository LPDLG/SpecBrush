"""
Pureformer MDTA + GDFN Blocks (CVPRW 2025 NTIRE)

Extracted as plug-and-play modules for the Self-Supervised Mu-Denoiser.
Based on: Multi-Dconv Head Transposed Attention (MDTA) + Gated-Dconv Feed-Forward Network (GDFN)

Reference: User-provided implementation following Pureformer paper architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dim for NCHW (per spatial location)."""
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class MultiDilatedDWConv(nn.Module):
    """
    Approximate Pureformer 'Multi-Dconv head' by summing several depthwise convs with different dilation.
    Keeps it lightweight & plug-and-play.
    """
    def __init__(self, channels: int, dilations=(1, 2), bias=True):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.Conv2d(
                    channels, channels,
                    kernel_size=3, stride=1,
                    padding=d, dilation=d,
                    groups=channels, bias=bias
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.convs) == 1:
            return self.convs[0](x)
        out = 0
        for c in self.convs:
            out = out + c(x)
        return out / len(self.convs)


class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (channel-wise attention).
    - Q,K,V are built with 1x1 conv then multi-dilated depthwise conv (multi-dconv head).
    - Attention is computed in channel dimension, consistent with Pureformer description.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        bias: bool = True,
        dilations=(1, 2),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"
        self.dim = dim
        self.num_heads = num_heads

        # per-head learnable temperature (stabilizes channel-attention)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_mdw = MultiDilatedDWConv(dim * 3, dilations=dilations, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_mdw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # [B, heads, C/head, HW]
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        # normalize on token axis for stability
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # channel attention: [B, heads, C/head, C/head]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B, heads, C/head, HW]
        out = out.view(b, c, h, w)

        out = self.project_out(out)
        out = self.proj_drop(out)
        return out


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network.
    Pureformer: 一路 ReLU，一路 Sigmoid，逐元素相乘门控，再 residual add.
    """
    def __init__(
        self,
        dim: int,
        ffn_expansion_factor: float = 2.0,
        bias: bool = True,
        gate: str = "relu_sigmoid",  # align with Pureformer description
        dilations=(1,),
    ):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = MultiDilatedDWConv(hidden * 2, dilations=dilations, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)
        self.gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)

        if self.gate == "relu_sigmoid":
            x = F.relu(x1) * torch.sigmoid(x2)
        else:
            # fallback (optional)
            x = F.gelu(x1) * x2

        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    """
    PreNorm + (MDTA + GDFN) + residuals.
    This is the plug-and-play core from Pureformer.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_expansion_factor: float = 2.0,
        bias: bool = True,
        ln_eps: float = 1e-6,
        attn_dilations=(1, 2),
        ffn_dilations=(1,),
        gate: str = "relu_sigmoid",
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, eps=ln_eps)
        self.attn = MDTA(dim, num_heads=num_heads, bias=bias, dilations=attn_dilations)
        self.norm2 = LayerNorm2d(dim, eps=ln_eps)
        self.ffn = GDFN(
            dim, ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, gate=gate, dilations=ffn_dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
