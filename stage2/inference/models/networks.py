# -*- coding: utf-8 -*-
"""Network factory for official-compatible inference.

Usage:
    ``network_G`` keeps the original entry point and now supports
    ``ConditionalUNetWithPriorControl``.
    ``network_Gs`` and ``Dis`` stay aligned with the official test pipeline.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

from models import modules as M

logger = logging.getLogger("base")


def define_G(opt):
    """Create texture generator, structure generator, and discriminator."""
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]
    setting = opt_net["setting"]

    if which_model == "ConditionalUNetWithPriorControl":
        from models.prior_control_wrapper import ConditionalUNetWithPriorControl
        prior_opt = opt.get("prior_controlnet", {}) or {}
        mglc_opt = opt.get("mglc", {}) or {}
        net_g = ConditionalUNetWithPriorControl(
            in_nc=setting.get("in_nc", 3),
            out_nc=setting.get("out_nc", 3),
            nf=setting.get("nf", 64),
            depth=setting.get("depth", 4),
            prior_control_enabled=prior_opt.get("enabled", True),
            gate_hidden=prior_opt.get("gate_hidden", 16),
            mglc_opt=mglc_opt,
            restore_S_guidance=opt.get("restore_S_guidance", True),
            restore_S_guidance_scale=opt.get("restore_S_guidance_scale", 1.0),
        )
    else:
        net_g = getattr(M, which_model)(**setting)

    opt_net_s = opt["network_Gs"]
    which_model_s = opt_net_s["which_model_G"]
    setting_s = opt_net_s["setting"]
    net_gs = getattr(M, which_model_s)(**setting_s)
    dis = Discriminator(nc=4, ngf=64, t_emb_dim=256, act=nn.LeakyReLU(0.2))
    return net_g, net_gs, dis

from . import dense_layer
from . import layers
from . import up_or_down_sampling

dense = dense_layer.dense
conv2d = dense_layer.conv2d
get_sinusoidal_positional_embedding = layers.get_timestep_embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        return self.main(temb)


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim=128,
        downsample=False,
        act=nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1),
    ):
        super().__init__()
        self.fir_kernel = fir_kernel
        self.downsample = downsample
        self.conv1 = nn.Sequential(conv2d(in_channel, out_channel, kernel_size, padding=padding))
        self.conv2 = nn.Sequential(
            conv2d(out_channel, out_channel, kernel_size, padding=padding, init_scale=0.0)
        )
        self.dense_t1 = dense(t_emb_dim, out_channel)
        self.act = act
        self.skip = nn.Sequential(
            conv2d(in_channel, out_channel, 1, padding=0, bias=False)
        )

    def forward(self, input, t_emb):
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]
        out = self.act(out)
        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)
        skip = self.skip(input)
        return (out + skip) / np.sqrt(2)


class Discriminator(nn.Module):
    """Time-dependent discriminator kept for official inference compatibility."""

    def __init__(self, nc=3, ngf=64, t_emb_dim=128, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.act = act
        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )
        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlock(ngf * 2, ngf * 2, t_emb_dim=t_emb_dim, act=act)
        self.conv2 = DownConvBlock(
            ngf * 2, ngf * 4, t_emb_dim=t_emb_dim, downsample=True, act=act
        )
        self.conv3 = DownConvBlock(
            ngf * 4, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act
        )
        self.conv4 = DownConvBlock(
            ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act
        )
        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1, init_scale=0.0)
        self.end_linear = dense(ngf * 8, 1)
        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, t, x_t, s_optimum):
        t = t.to(x_t.device)
        t_embed = self.act(self.t_embed(t))
        input_x = torch.cat([x_t, s_optimum], 1)
        h0 = self.start_conv(input_x)
        h1 = self.conv1(h0, t_embed)
        h2 = self.conv2(h1, t_embed)
        h3 = self.conv3(h2, t_embed)
        out = self.conv4(h3, t_embed)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.final_conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        return self.end_linear(out).view(-1)



