"""
Color utility functions for the pigment fading task.

- sRGB <-> CIE Lab (D65, 2°)
- simple Lab normalization helpers
- DeltaE2000 metric
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_XN: float = 0.95047
_YN: float = 1.0
_ZN: float = 1.08883


def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    u = u / 255.0
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(u: np.ndarray) -> np.ndarray:
    u = np.where(u <= 0.0031308, 12.92 * u, 1.055 * np.power(np.clip(u, 0.0, None), 1 / 2.4) - 0.055)
    return np.clip(u, 0.0, 1.0)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    r_lin = _srgb_to_linear(rgb[..., 0])
    g_lin = _srgb_to_linear(rgb[..., 1])
    b_lin = _srgb_to_linear(rgb[..., 2])

    x = (0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin) / _XN
    y = (0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin) / _YN
    z = (0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin) / _ZN

    delta = 6 / 29

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([l, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    delta = 6 / 29

    def finv(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))

    x = finv(fx) * _XN
    y = finv(fy) * _YN
    z = finv(fz) * _ZN

    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    rgb = _linear_to_srgb(np.stack([r_lin, g_lin, b_lin], axis=-1))
    return np.round(rgb * 255).astype(np.uint8)


@dataclass(frozen=True)
class LabNorm:
    L_scale: float = 100.0
    ab_scale: float = 128.0

    def normalize(self, lab: np.ndarray) -> np.ndarray:
        lab = np.asarray(lab, dtype=np.float32).copy()
        lab[..., 0] = lab[..., 0] / self.L_scale
        lab[..., 1] = lab[..., 1] / self.ab_scale
        lab[..., 2] = lab[..., 2] / self.ab_scale
        return lab

    def denormalize(self, lab_n: np.ndarray) -> np.ndarray:
        lab_n = np.asarray(lab_n, dtype=np.float32).copy()
        lab_n[..., 0] = lab_n[..., 0] * self.L_scale
        lab_n[..., 1] = lab_n[..., 1] * self.ab_scale
        lab_n[..., 2] = lab_n[..., 2] * self.ab_scale
        return lab_n


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c1 = np.sqrt(a1**2 + b1**2)
    c2 = np.sqrt(a2**2 + b2**2)
    c_bar = (c1 + c2) / 2.0

    g = 0.5 * (1 - np.sqrt((c_bar**7) / (c_bar**7 + 25**7)))
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = np.sqrt(a1p**2 + b1**2)
    c2p = np.sqrt(a2p**2 + b2**2)
    c_bar_p = (c1p + c2p) / 2.0

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dlp = l2 - l1
    dcp = c2p - c1p
    dhp = h2p - h1p
    dhp = np.where((c1p * c2p) == 0, 0.0, dhp)
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp / 2.0))

    l_bar_p = (l1 + l2) / 2.0
    h_bar_p = (h1p + h2p) / 2.0
    h_bar_p = np.where(np.abs(h1p - h2p) > 180, h_bar_p + 180, h_bar_p)
    h_bar_p = np.where((c1p * c2p) == 0, h1p + h2p, h_bar_p)
    h_bar_p = h_bar_p % 360.0

    t = (
        1
        - 0.17 * np.cos(np.radians(h_bar_p - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar_p))
        + 0.32 * np.cos(np.radians(3 * h_bar_p + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar_p - 63))
    )
    dtheta = 30 * np.exp(-((h_bar_p - 275) / 25) ** 2)
    rc = 2 * np.sqrt((c_bar_p**7) / (c_bar_p**7 + 25**7))
    sl = 1 + (0.015 * (l_bar_p - 50) ** 2) / np.sqrt(20 + (l_bar_p - 50) ** 2)
    sc = 1 + 0.045 * c_bar_p
    sh = 1 + 0.015 * c_bar_p * t
    rt = -np.sin(np.radians(2 * dtheta)) * rc

    return np.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp / sh) ** 2 + rt * (dcp / sc) * (dhp / sh))
