"""Frozen StrDiffusion texture UNet with released SpecBrush controls."""
from __future__ import annotations

import functools
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.DenoisingUNet_arch import SPADEBlock
from .modules.mglc_block import MGLCBlock
from .modules.module_util import (
    Downsample, LinearAttention, NonLinearity, PreNorm, ResBlock, Residual,
    SinusoidalPosEmb, Upsample, default_conv,
)
from .prior_controlnet import PriorControlNet


class ConfidenceCalibrationGate(nn.Module):
    """Eq. (10): sigmoid(G([D_s(Q), D_s(M), channel_mean(H_s)]))."""
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 1, 1)
        )
    def forward(self, feature, confidence, missing_mask):
        size = feature.shape[-2:]
        q = F.interpolate(confidence.float(), size=size, mode="bilinear", align_corners=False)
        m = F.interpolate(missing_mask.float(), size=size, mode="bilinear", align_corners=False)
        hbar = feature.mean(dim=1, keepdim=True)
        return torch.sigmoid(self.net(torch.cat([q, m, hbar], dim=1)))


class ConditionalUNetWithPriorControl(nn.Module):
    """SpecBrush Stage-II generator.

    PriorControlNet injects at all encoder/mid residual scales. MGLC is applied
    at two spatial scales in S_inj (bottleneck and final decoder), which is a
    multi-scale subset of the prior-conditioned features used by Eq. (14).
    """
    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        depth: int = 4,
        prior_control_enabled: bool = True,
        gate_hidden: int = 16,
        mglc_opt: Optional[dict] = None,
        restore_S_guidance: bool = True,
        restore_S_guidance_scale: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.nf = int(nf)
        self.prior_control_enabled = bool(prior_control_enabled)
        self.restore_S_guidance = bool(restore_S_guidance)
        self.restore_S_guidance_scale = float(restore_S_guidance_scale)
        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        time_dim = nf * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(nf), nn.Linear(nf,time_dim), nn.GELU(), nn.Linear(time_dim,time_dim)
        )
        self.init_conv = default_conv(in_nc * 2, nf, 7)

        self.downs = nn.ModuleList()
        skip_channels = []
        for i in range(depth):
            dim_in = nf * int(math.pow(2,i))
            dim_out = nf * int(math.pow(2,i+1))
            blocks = [
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in,dim_out) if i != depth-1 else default_conv(dim_in,dim_out),
            ]
            if self.restore_S_guidance:
                blocks.append(SPADEBlock(dim_out, dim_out, 1))
            self.downs.append(nn.ModuleList(blocks))
            skip_channels.extend([dim_in,dim_in])

        self.ups = nn.ModuleList()
        for i in range(depth):
            dim_in = nf * int(math.pow(2,i))
            dim_out = nf * int(math.pow(2,i+1))
            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out+dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out+dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out,dim_in) if i != 0 else default_conv(dim_out,dim_in),
            ]))

        mid_dim = nf * int(math.pow(2,depth))
        self.mid_block1 = block_class(dim_in=mid_dim,dim_out=mid_dim,time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim,LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim,dim_out=mid_dim,time_emb_dim=time_dim)
        self.final_res_block = block_class(dim_in=nf*2,dim_out=nf,time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf,out_nc,3,1,1)

        self.prior_controlnet = PriorControlNet(nf=nf,depth=depth) if self.prior_control_enabled else None
        self.prior_gates = nn.ModuleList([ConfidenceCalibrationGate(gate_hidden) for _ in skip_channels])
        self.prior_gate_mid = ConfidenceCalibrationGate(gate_hidden)

        mopt = dict(mglc_opt or {})
        self.mglc_enabled = bool(mopt.get("enabled",True))
        kwargs = dict(
            gate_hidden=int(mopt.get("gate_hidden",16)),
            boundary_width=int(mopt.get("boundary_width",3)),
            lambda_b=float(mopt.get("lambda_b",1.0)),
            branch_mode=str(mopt.get("branch_mode","both")),
        )
        self.mglc_mid = MGLCBlock(mid_dim,**kwargs) if self.mglc_enabled else None
        self.mglc_dec = MGLCBlock(nf,**kwargs) if self.mglc_enabled else None

    def check_image_size(self,x,h,w):
        scale = 2 ** self.depth
        ph=(scale-h%scale)%scale; pw=(scale-w%scale)%scale
        return F.pad(x,(0,pw,0,ph),mode="reflect")

    def _inject(self,feature,residual,gate_module,mask,confidence):
        return feature + gate_module(feature,confidence,mask) * residual

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor,
        time: Union[int,float,torch.Tensor],
        S: Optional[torch.Tensor]=None,
        mask: Optional[torch.Tensor]=None,
        color_prior: Optional[torch.Tensor]=None,
        confidence: Optional[torch.Tensor]=None,
        observed_degraded: Optional[torch.Tensor]=None,
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        if isinstance(time,(int,float)):
            time=torch.tensor([time],device=xt.device)
        if time.dim()==0: time=time.unsqueeze(0)
        if self.prior_controlnet is not None and any(v is None for v in [mask,color_prior,confidence,observed_degraded]):
            raise ValueError("PriorControlNet requires mask, color_prior, confidence and observed_degraded")

        x=torch.cat([xt-cond,cond],dim=1)
        h,w=x.shape[-2:]
        x=self.check_image_size(x,h,w)
        t=self.time_mlp(time)
        mask_p=self.check_image_size(mask,h,w) if mask is not None else None

        prior_out=None
        if self.prior_controlnet is not None:
            prior_out=self.prior_controlnet(xt,mask,color_prior,confidence,observed_degraded,time)
        residuals=prior_out["down_residuals"] if prior_out is not None else None

        x=self.init_conv(x); x_res=x.clone(); skips=[]; ridx=0
        for blocks in self.downs:
            b1,b2,attn,downsample=blocks[:4]
            x=b1(x,t)
            if residuals is not None:
                x=self._inject(x,residuals[ridx],self.prior_gates[ridx],mask,confidence)
            ridx+=1; skips.append(x)
            x=b2(x,t); x=attn(x)
            if residuals is not None:
                x=self._inject(x,residuals[ridx],self.prior_gates[ridx],mask,confidence)
            ridx+=1; skips.append(x)
            x=downsample(x)
            if self.restore_S_guidance and S is not None and len(blocks)>4:
                guided=blocks[4](x,S)
                x=x+self.restore_S_guidance_scale*(guided-x)

        x=self.mid_block1(x,t); x=self.mid_attn(x); x=self.mid_block2(x,t)
        if prior_out is not None:
            x=self._inject(x,prior_out["mid_residual"],self.prior_gate_mid,mask,confidence)
        if self.mglc_mid is not None:
            x=self.mglc_mid(x,mask,confidence)

        for blocks in self.ups:
            b1,b2,attn,upsample=blocks
            x=b1(torch.cat([x,skips.pop()],dim=1),t)
            x=b2(torch.cat([x,skips.pop()],dim=1),t)
            x=attn(x); x=upsample(x)

        if self.mglc_dec is not None:
            x=self.mglc_dec(x,mask,confidence)
        x=self.final_res_block(torch.cat([x,x_res],dim=1),t)
        x=self.final_conv(x)[...,:h,:w]
        return x,x



