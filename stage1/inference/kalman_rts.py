"""Kalman-RTS posterior smoothing for ordered Stage-I aging predictions.

The smoother is applied across adjacent aging time points after
K-run diffusion inversion. This module intentionally operates on an ordered
sequence, not on a single RGB query or a two-point [prediction, observation]
pair.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def kalman_rts_smooth_lab(
    mean_lab: np.ndarray,
    var_lab: Optional[np.ndarray] = None,
    process_std_lab: float = 2.0,
    measurement_floor_lab: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth an ordered [T,3] Lab sequence with a random-walk state model.

    Args:
        mean_lab: K-sample posterior means at adjacent aging time points [T,3].
        var_lab: K-sample posterior variances [T,3]. If omitted, a constant
            measurement variance is used.
        process_std_lab: random-walk process standard deviation.
        measurement_floor_lab: lower bound for measurement standard deviation.

    Returns:
        smoothed_mean: [T,3]
        smoothed_var: [T,3]
    """
    y = np.asarray(mean_lab, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError(f"mean_lab must be [T,3], got {y.shape}")
    length = y.shape[0]
    if length == 0:
        return y.astype(np.float32), y.astype(np.float32)

    floor_var = float(measurement_floor_lab) ** 2
    if var_lab is None:
        r = np.full_like(y, floor_var, dtype=np.float64)
    else:
        r = np.asarray(var_lab, dtype=np.float64)
        if r.shape != y.shape:
            raise ValueError(f"var_lab shape {r.shape} != mean_lab shape {y.shape}")
        r = np.maximum(r, floor_var)

    q = np.full((3,), float(process_std_lab) ** 2, dtype=np.float64)
    x_f = np.empty_like(y)
    p_f = np.empty_like(y)
    x_pred = np.empty_like(y)
    p_pred = np.empty_like(y)

    x = y[0].copy()
    p = r[0].copy()
    for t in range(length):
        if t == 0:
            xp, pp = x.copy(), p.copy()
        else:
            xp, pp = x.copy(), p + q
        gain = pp / np.maximum(pp + r[t], 1e-12)
        x = xp + gain * (y[t] - xp)
        p = (1.0 - gain) * pp
        x_pred[t], p_pred[t] = xp, pp
        x_f[t], p_f[t] = x, p

    x_s = x_f.copy()
    p_s = p_f.copy()
    for t in range(length - 2, -1, -1):
        denom = np.maximum(p_pred[t + 1], 1e-12)
        gain = p_f[t] / denom
        x_s[t] = x_f[t] + gain * (x_s[t + 1] - x_pred[t + 1])
        p_s[t] = p_f[t] + gain * (p_s[t + 1] - p_pred[t + 1]) * gain

    return x_s.astype(np.float32), np.maximum(p_s, 0.0).astype(np.float32)
