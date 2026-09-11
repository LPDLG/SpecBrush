"""
IO and preprocessing helpers for the pigment restoration task.

- parse RGB log TXT files
- load Raman/XRD spectra from legacy multi-sheet workbooks
- load Raman/XRD spectra from single-sheet wide workbooks
- peak feature extraction helpers
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import re

from scipy import sparse
from scipy.signal import find_peaks, savgol_filter


_RGB_LINE_RE = re.compile(r"NO\.(\d+)\s+R:(\d+)\s+G:(\d+)\s+B:(\d+)")
_TEMP_LINE_RE = re.compile(r"(\d+)?\s*Temperature:\s*([0-9.]+)\s*C\s*Humidity:\s*([0-9.]+)")
_SHEET_PREFIX_RE = re.compile(r"^\s*(\d{2})\s+")


@dataclass
class RGBLog:
    rgb: np.ndarray
    block_idx: List[int]
    temperature: List[float]
    humidity: List[float]


@dataclass
class WideSpectrum:
    header: str
    exp_tag: str
    base_name: str
    x: np.ndarray
    y: np.ndarray
    count: int


_EXCEL_CACHE: Dict[str, Dict[str, object]] = {}
_WIDE_CACHE: Dict[Tuple[str, int, bool, str], Dict[str, WideSpectrum]] = {}


def parse_rgb_log_txt(path: str) -> RGBLog:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    blocks: List[Tuple[int, float, float, List[Tuple[int, int, int, int]]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Temperature" not in line or "Humidity" not in line:
            i += 1
            continue
        match = _TEMP_LINE_RE.search(line)
        if match is None:
            i += 1
            continue
        block_idx = int(match.group(1)) if match.group(1) else -1
        temp = float(match.group(2))
        hum = float(match.group(3))
        i += 1
        colors: List[Tuple[int, int, int, int]] = []
        while i < len(lines):
            line2 = lines[i]
            match2 = _RGB_LINE_RE.match(line2)
            if match2 is None:
                break
            colors.append((int(match2.group(1)), int(match2.group(2)), int(match2.group(3)), int(match2.group(4))))
            i += 1
        if colors:
            blocks.append((block_idx, temp, hum, colors))

    if not blocks:
        raise ValueError(f"No valid RGB blocks found in {path}")

    patch_count = max(no for _, _, _, colors in blocks for no, _, _, _ in colors)
    time_steps = len(blocks)
    rgb = np.zeros((time_steps, patch_count, 3), dtype=np.float64)
    block_idx: List[int] = []
    temperature: List[float] = []
    humidity: List[float] = []
    for t, (bidx, temp, hum, colors) in enumerate(blocks):
        block_idx.append(bidx)
        temperature.append(temp)
        humidity.append(hum)
        for no, r, g, b in colors:
            rgb[t, no - 1, :] = [r, g, b]
    return RGBLog(rgb=rgb, block_idx=block_idx, temperature=temperature, humidity=humidity)


def filter_rgb_log_by_humidity(log: RGBLog, tol: float = 15.0) -> RGBLog:
    hum = np.asarray(log.humidity, dtype=np.float32)
    if hum.size == 0:
        return log
    median = float(np.median(hum))
    keep = np.abs(hum - median) <= float(tol)
    if np.all(keep):
        return log
    return RGBLog(
        rgb=log.rgb[keep],
        block_idx=[b for b, k in zip(log.block_idx, keep) if k],
        temperature=[t for t, k in zip(log.temperature, keep) if k],
        humidity=[h for h, k in zip(log.humidity, keep) if k],
    )


def guess_experiment_tag(median_humidity: float) -> str:
    return "66" if float(median_humidity) < 70.0 else "76"


def infer_side_label(path: str) -> str:
    lower = path.lower()
    if "right" in lower:
        return "right"
    if "left" in lower:
        return "left"
    return "unknown"


def resample_1d(
    x: np.ndarray,
    y: np.ndarray,
    new_len: int,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x_min is None:
        x_min = float(np.nanmin(x))
    if x_max is None:
        x_max = float(np.nanmax(x))
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2:
        raise ValueError("Not enough valid points to resample.")
    new_x = np.linspace(x_min, x_max, int(new_len))
    return np.interp(new_x, x, y).astype(np.float32)


def _standardize_intensity(intensity: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    intensity = intensity.astype(np.float32)
    return (intensity - float(np.mean(intensity))) / (float(np.std(intensity)) + eps)


def adapt_sheet_name(sheet_name: str, exp_tag: str) -> str:
    if not sheet_name:
        return sheet_name
    return _SHEET_PREFIX_RE.sub(f"{exp_tag} ", sheet_name.strip())


def split_prefix_base(sheet_name: str) -> Tuple[str, str]:
    text = (sheet_name or "").strip()
    match = _SHEET_PREFIX_RE.match(text)
    if not match:
        return "", text
    return match.group(1), text[match.end():].strip()


def canonical_base_name(base: str) -> str:
    text = (base or "").strip().replace(" ", "")
    if not text:
        return ""
    parts = [part.strip() for part in text.split("+") if part.strip()]
    return "+".join(sorted(parts))


def _sheet_cache(excel_path: str) -> Dict[str, object]:
    if excel_path in _EXCEL_CACHE:
        return _EXCEL_CACHE[excel_path]
    xls = pd.ExcelFile(excel_path)
    sheet_names = list(xls.sheet_names)
    canon_index: Dict[str, Dict[str, str]] = {}
    for raw in sheet_names:
        prefix, base = split_prefix_base(str(raw))
        canon_index.setdefault(prefix, {})
        canon = canonical_base_name(base)
        if canon and canon not in canon_index[prefix]:
            canon_index[prefix][canon] = str(raw)
    cache: Dict[str, object] = {
        "sheet_names": sheet_names,
        "sheet_set": set(sheet_names),
        "nospace_index": {str(name).replace(" ", ""): str(name) for name in sheet_names},
        "canon_index": canon_index,
    }
    _EXCEL_CACHE[excel_path] = cache
    return cache


def resolve_sheet_name(excel_path: str, wanted_sheet: str, exp_tag: str) -> str:
    if not excel_path or not wanted_sheet:
        return ""
    cache = _sheet_cache(excel_path)
    wanted = adapt_sheet_name(wanted_sheet, exp_tag).strip()
    if wanted in cache["sheet_set"]:
        return wanted
    wanted_ns = wanted.replace(" ", "")
    nospace_index = cache["nospace_index"]
    if wanted_ns in nospace_index:
        return str(nospace_index[wanted_ns])
    prefix, base = split_prefix_base(wanted)
    canon = canonical_base_name(base)
    canon_index = cache["canon_index"]
    if prefix in canon_index and canon in canon_index[prefix]:
        return str(canon_index[prefix][canon])
    if not prefix:
        for mapping in canon_index.values():
            if canon in mapping:
                return str(mapping[canon])
    return ""


def load_raman_excel_sheet(excel_path: str, sheet_name: str, new_len: int = 1024, standardize: bool = True) -> np.ndarray:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if df.shape[1] < 2:
        raise ValueError(f"Raman sheet {sheet_name} has <2 columns.")
    out = resample_1d(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy(), new_len=new_len)
    return _standardize_intensity(out) if standardize else out


def load_xrd_excel_sheet(excel_path: str, sheet_name: str, new_len: int = 2048, standardize: bool = True) -> np.ndarray:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if df.shape[1] < 2:
        raise ValueError(f"XRD sheet {sheet_name} has <2 columns.")
    out = resample_1d(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy(), new_len=new_len)
    return _standardize_intensity(out) if standardize else out


def load_rruff_raman_sheet(excel_path: str, sheet_name: str, new_len: int = 1024, standardize: bool = True) -> np.ndarray:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"RRUFF Raman sheet {sheet_name} has <2 columns.")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        raise ValueError(f"RRUFF Raman sheet {sheet_name} has too few numeric points.")
    out = resample_1d(x[ok], y[ok], new_len=new_len)
    return _standardize_intensity(out) if standardize else out


def _iter_pairs(columns: Sequence[int], subheaders: Sequence[str]) -> Iterable[Tuple[int, int]]:
    idx = 0
    while idx < len(columns) - 1:
        cur = (subheaders[idx] or "").strip().lower()
        nxt = (subheaders[idx + 1] or "").strip().lower()
        if "x" in cur and "y" in nxt:
            yield columns[idx], columns[idx + 1]
            idx += 2
            continue
        idx += 1


def _canonical_wide_key(header: str) -> Tuple[str, str]:
    text = (header or "").strip()
    exp_tag = ""
    match = re.search(r"(66|76)\s*$", text)
    if match:
        exp_tag = match.group(1)
        text = text[:match.start()].strip()
    text = re.sub(r"^[0-9A-Za-z]+", "", text).strip()
    return exp_tag, canonical_base_name(text)


def parse_wide_sheet_workbook(
    excel_path: str,
    new_len: int,
    kind: str,
    standardize: bool = True,
) -> Dict[str, WideSpectrum]:
    cache_key = (excel_path, int(new_len), bool(standardize), str(kind))
    if cache_key in _WIDE_CACHE:
        return _WIDE_CACHE[cache_key]

    xls = pd.ExcelFile(excel_path)
    if len(xls.sheet_names) != 1:
        _WIDE_CACHE[cache_key] = {}
        return _WIDE_CACHE[cache_key]

    df = pd.read_excel(excel_path, sheet_name=xls.sheet_names[0], header=None)
    if df.shape[0] < 3:
        raise ValueError(f"Wide workbook {excel_path} has too few rows.")

    headers = pd.Series(df.iloc[0]).ffill().astype(str).tolist()
    subheaders = ["" if pd.isna(v) else str(v) for v in df.iloc[1].tolist()]
    values = df.iloc[2:].copy()

    grouped: Dict[str, List[int]] = {}
    ordered_headers: List[str] = []
    for idx, header in enumerate(headers):
        header = (header or "").strip()
        if not header or header.lower() == "nan":
            continue
        if header not in grouped:
            grouped[header] = []
            ordered_headers.append(header)
        grouped[header].append(idx)

    out: Dict[str, WideSpectrum] = {}
    for header in ordered_headers:
        columns = grouped[header]
        pairs = list(_iter_pairs(columns, [subheaders[i] for i in columns]))
        if not pairs:
            continue
        exp_tag, base_name = _canonical_wide_key(header)
        if not exp_tag or not base_name:
            continue
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for x_col, y_col in pairs:
            x = pd.to_numeric(values.iloc[:, x_col], errors="coerce").to_numpy()
            y = pd.to_numeric(values.iloc[:, y_col], errors="coerce").to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                continue
            x = x[ok].astype(np.float32)
            y = y[ok].astype(np.float32)
            x_grid = np.linspace(float(np.min(x)), float(np.max(x)), int(new_len), dtype=np.float32)
            y_grid = resample_1d(x, y, new_len=new_len, x_min=float(x_grid[0]), x_max=float(x_grid[-1]))
            xs.append(x_grid)
            ys.append(y_grid)
        if not ys:
            continue
        mean_x = np.mean(np.stack(xs, axis=0), axis=0).astype(np.float32)
        mean_y = np.mean(np.stack(ys, axis=0), axis=0).astype(np.float32)
        if standardize:
            mean_y = _standardize_intensity(mean_y)
        out[f"{exp_tag} {base_name}"] = WideSpectrum(
            header=header,
            exp_tag=exp_tag,
            base_name=base_name,
            x=mean_x,
            y=mean_y,
            count=len(ys),
        )

    _WIDE_CACHE[cache_key] = out
    return out


def load_xy_from_excel(
    excel_path: str,
    sheet_name: str,
    x_col: int = 0,
    y_col: int = 1,
    header: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header)
    if df.shape[1] <= max(x_col, y_col):
        raise ValueError(f"Sheet {sheet_name} has <={max(x_col, y_col)} columns")
    x = pd.to_numeric(df.iloc[:, x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, y_col], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok].astype(np.float32)
    y = y[ok].astype(np.float32)
    if x.size < 10:
        raise ValueError(f"Sheet {sheet_name} has too few numeric points")
    return x, y


def load_spectrum_from_workbook(
    excel_path: str,
    wanted_sheet: str,
    exp_tag: str,
    new_len: int,
    kind: str,
    standardize: bool = True,
) -> np.ndarray:
    sheet_name = resolve_sheet_name(excel_path, wanted_sheet, exp_tag)
    if sheet_name:
        if kind == "raman":
            return load_raman_excel_sheet(excel_path, sheet_name, new_len=new_len, standardize=standardize)
        return load_xrd_excel_sheet(excel_path, sheet_name, new_len=new_len, standardize=standardize)

    prefix, base = split_prefix_base(adapt_sheet_name(wanted_sheet, exp_tag))
    wide = parse_wide_sheet_workbook(excel_path, new_len=new_len, kind=kind, standardize=standardize)
    key = f"{prefix} {canonical_base_name(base)}".strip()
    if key in wide:
        return wide[key].y.astype(np.float32)
    raise ValueError(f"Unable to resolve spectrum '{wanted_sheet}' in {excel_path}")


def load_xy_from_workbook(
    excel_path: str,
    wanted_sheet: str,
    exp_tag: str,
    new_len: int,
    kind: str,
) -> Tuple[np.ndarray, np.ndarray]:
    sheet_name = resolve_sheet_name(excel_path, wanted_sheet, exp_tag)
    if sheet_name:
        return load_xy_from_excel(excel_path, sheet_name, x_col=0, y_col=1, header=0)

    prefix, base = split_prefix_base(adapt_sheet_name(wanted_sheet, exp_tag))
    wide = parse_wide_sheet_workbook(excel_path, new_len=new_len, kind=kind, standardize=False)
    key = f"{prefix} {canonical_base_name(base)}".strip()
    if key in wide:
        return wide[key].x.astype(np.float32), wide[key].y.astype(np.float32)
    raise ValueError(f"Unable to resolve XY spectrum '{wanted_sheet}' in {excel_path}")


def baseline_als(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    y = y.astype(np.float64)
    length = y.size
    d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2))
    w = np.ones(length)
    for _ in range(int(niter)):
        w_mat = sparse.spdiags(w, 0, length, length)
        z = sparse.linalg.spsolve(w_mat + lam * (d @ d.T), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z.astype(np.float32)


def preprocess_spectrum_for_peaks(
    x: np.ndarray,
    y: np.ndarray,
    do_baseline: bool = True,
    do_smooth: bool = True,
    smooth_window: int = 11,
    smooth_poly: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    y2 = y.astype(np.float32)
    if do_baseline:
        y2 = y2 - baseline_als(y2, lam=1e6, p=0.01, niter=10)
    if do_smooth and y2.size >= smooth_window:
        if smooth_window % 2 == 0:
            smooth_window += 1
        y2 = savgol_filter(y2, window_length=smooth_window, polyorder=smooth_poly).astype(np.float32)
    y2 = y2 - float(np.min(y2))
    y2 = y2 / (float(np.max(y2)) + 1e-8)
    return x.astype(np.float32), y2.astype(np.float32)


def extract_peak_features_xy(
    x: np.ndarray,
    y: np.ndarray,
    top_k: int = 32,
    prominence: float = 0.05,
    distance: Optional[int] = None,
) -> np.ndarray:
    x2, y2 = preprocess_spectrum_for_peaks(x, y, do_baseline=True, do_smooth=True)
    peaks, _ = find_peaks(y2, prominence=prominence, distance=distance)
    if peaks.size == 0:
        return np.zeros((2 * int(top_k),), dtype=np.float32)
    heights = y2[peaks]
    order = np.argsort(-heights)
    peaks = peaks[order][: int(top_k)]
    heights = heights[order][: int(top_k)]
    x_min = float(np.min(x2))
    x_max = float(np.max(x2))
    if x_max <= x_min:
        pos = np.zeros_like(heights, dtype=np.float32)
    else:
        pos = ((x2[peaks] - x_min) / (x_max - x_min)).astype(np.float32)
    pos_pad = np.zeros((int(top_k),), dtype=np.float32)
    height_pad = np.zeros((int(top_k),), dtype=np.float32)
    pos_pad[: peaks.size] = pos
    height_pad[: peaks.size] = heights.astype(np.float32)
    return np.concatenate([pos_pad, height_pad], axis=0).astype(np.float32)
