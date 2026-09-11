#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess RGB logs and spectral workbooks into pair-based NPZ files plus sidecar indexes."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from data.index.sample_index import write_sample_index
from data.splits.grouping import group_split_indices, random_split_indices
from utils.color_utils import LabNorm, rgb_to_lab
from utils.io_utils import (
    adapt_sheet_name,
    canonical_base_name,
    extract_peak_features_xy,
    filter_rgb_log_by_humidity,
    guess_experiment_tag,
    infer_side_label,
    load_spectrum_from_workbook,
    load_xy_from_workbook,
    parse_rgb_log_txt,
    parse_wide_sheet_workbook,
    split_prefix_base,
)


def _parse_int_list(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    if '-' in text and ',' not in text:
        start, end = text.split('-')
        return list(range(int(start), int(end) + 1))
    return [int(part.strip()) for part in text.split(',') if part.strip()]


def _load_meta_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_patch_sheet_maps(meta: Dict, exp_tag: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    if 'experiments' in meta and isinstance(meta['experiments'], dict) and exp_tag in meta['experiments']:
        entry = meta['experiments'][exp_tag]
        p2r = {str(k): str(v) for k, v in entry.get('patch_to_raman_sheet', {}).items()}
        p2x = {str(k): str(v) for k, v in entry.get('patch_to_xrd_sheet', {}).items()}
    else:
        p2r = {str(k): str(v) for k, v in meta.get('patch_to_raman_sheet', {}).items()}
        p2x = {str(k): str(v) for k, v in meta.get('patch_to_xrd_sheet', {}).items()}
    return {k: adapt_sheet_name(v, exp_tag) for k, v in p2r.items()}, {k: adapt_sheet_name(v, exp_tag) for k, v in p2x.items()}


def _infer_patch_sheet_map_from_wide(
    excel_path: str,
    exp_tag: str,
    patch_ids: Sequence[int],
    new_len: int,
    kind: str,
) -> Dict[str, str]:
    if not excel_path:
        return {}
    try:
        wide = parse_wide_sheet_workbook(excel_path, new_len=new_len, kind=kind, standardize=False)
    except Exception:
        return {}
    ordered_bases: List[str] = []
    seen = set()
    for spec in wide.values():
        if str(spec.exp_tag) != str(exp_tag):
            continue
        base_name = canonical_base_name(spec.base_name)
        if not base_name or base_name in seen:
            continue
        seen.add(base_name)
        ordered_bases.append(base_name)
    out: Dict[str, str] = {}
    for pid, base_name in zip(patch_ids, ordered_bases):
        out[str(pid)] = f'{exp_tag} {base_name}'
    return out


def _candidate_sheet_names(primary_map: Dict[str, str], fallback_map: Dict[str, str], patch_id: int) -> List[str]:
    candidates: List[str] = []
    for mapping in (primary_map, fallback_map):
        raw = str(mapping.get(str(patch_id), '') or '').strip()
        if raw and raw not in candidates:
            candidates.append(raw)
    return candidates


def _sample_map(mapping: Dict[str, str], max_items: int = 4) -> Dict[str, str]:
    keys = list(mapping.keys())[: int(max_items)]
    return {key: mapping[key] for key in keys}


def _wide_keys_sample(excel_path: str, new_len: int, kind: str, max_items: int = 6) -> List[str]:
    if not excel_path:
        return []
    try:
        wide = parse_wide_sheet_workbook(excel_path, new_len=new_len, kind=kind, standardize=False)
    except Exception as exc:
        return [f'<parse_error: {exc}>']
    return list(wide.keys())[: int(max_items)]


def _save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def _slice_dict(arrays: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {key: value[idx] for key, value in arrays.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb_logs', type=str, required=True)
    ap.add_argument('--exp_tags', type=str, default='')
    ap.add_argument('--hum_tol', type=float, default=15.0)
    ap.add_argument('--use_patches', type=str, default='1-9')
    ap.add_argument('--meta_json', type=str, default='')
    ap.add_argument('--raman_excel', type=str, default='')
    ap.add_argument('--xrd_excel', type=str, default='')
    ap.add_argument('--peak_features', action='store_true')
    ap.add_argument('--peak_top_k', type=int, default=32)
    ap.add_argument('--peak_prominence', type=float, default=0.05)
    ap.add_argument('--allow_empty_spectra', action='store_true')
    ap.add_argument('--print_sheet_map', action='store_true')
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--split_mode', type=str, default='group_exp_patch', choices=['random', 'group_exp_patch', 'group_patch', 'group_exp'])
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    args = ap.parse_args()

    rgb_logs = [p.strip() for p in args.rgb_logs.split(',') if p.strip()]
    exp_tags_input = [t.strip() for t in args.exp_tags.split(',') if t.strip()]
    if exp_tags_input and len(exp_tags_input) != len(rgb_logs):
        raise ValueError('--exp_tags length must match --rgb_logs length')
    patch_ids = _parse_int_list(args.use_patches)
    if not patch_ids:
        raise ValueError('No patches selected')

    meta: Dict = _load_meta_json(args.meta_json) if args.meta_json else {}
    raman_len = int(meta.get('raman_len', 1024)) if meta else 1024
    xrd_len = int(meta.get('xrd_len', 2048)) if meta else 2048
    lab_norm = LabNorm()

    raman_by_exp_patch: Dict[Tuple[int, int], np.ndarray] = {}
    xrd_by_exp_patch: Dict[Tuple[int, int], np.ndarray] = {}
    has_raman_by_exp_patch: Dict[Tuple[int, int], int] = {}
    has_xrd_by_exp_patch: Dict[Tuple[int, int], int] = {}
    raman_peaks_by_exp_patch: Dict[Tuple[int, int], np.ndarray] = {}
    xrd_peaks_by_exp_patch: Dict[Tuple[int, int], np.ndarray] = {}
    exp_meta: List[Dict[str, object]] = []
    loaded_raman = 0
    loaded_xrd = 0
    raman_debug_rows: List[Dict[str, object]] = []
    xrd_debug_rows: List[Dict[str, object]] = []

    for exp_id, log_path in enumerate(rgb_logs):
        log = filter_rgb_log_by_humidity(parse_rgb_log_txt(log_path), tol=float(args.hum_tol))
        hum_med = float(np.median(np.asarray(log.humidity, dtype=np.float32)))
        exp_tag = exp_tags_input[exp_id] if exp_tags_input else guess_experiment_tag(hum_med)
        side = infer_side_label(log_path)
        exp_meta.append({'exp_id': exp_id, 'log_path': log_path, 'exp_tag': exp_tag, 'side': side, 'humidity_median': hum_med, 'T_blocks': int(log.rgb.shape[0])})
        p2r, p2x = _get_patch_sheet_maps(meta, exp_tag) if meta else ({}, {})
        auto_p2r = _infer_patch_sheet_map_from_wide(args.raman_excel, exp_tag, patch_ids, raman_len, 'raman') if args.raman_excel else {}
        auto_p2x = _infer_patch_sheet_map_from_wide(args.xrd_excel, exp_tag, patch_ids, xrd_len, 'xrd') if args.xrd_excel else {}

        if args.raman_excel:
            resolved = {}
            last_errors: Dict[str, str] = {}
            for pid in patch_ids:
                candidates = _candidate_sheet_names(p2r, auto_p2r, pid)
                resolved[str(pid)] = candidates[0] if candidates else ''
                if not candidates:
                    has_raman_by_exp_patch[(exp_id, pid)] = 0
                    last_errors[str(pid)] = 'no meta candidate and no wide-order fallback candidate'
                    continue
                loaded = False
                for raw in candidates:
                    try:
                        spec = load_spectrum_from_workbook(args.raman_excel, raw, exp_tag, new_len=raman_len, kind='raman', standardize=True)
                        raman_by_exp_patch[(exp_id, pid)] = spec.astype(np.float32)
                        has_raman_by_exp_patch[(exp_id, pid)] = int(np.any(np.abs(spec) > 1e-8))
                        loaded_raman += int(has_raman_by_exp_patch[(exp_id, pid)])
                        resolved[str(pid)] = raw
                        if args.peak_features:
                            x, y = load_xy_from_workbook(args.raman_excel, raw, exp_tag, new_len=raman_len, kind='raman')
                            raman_peaks_by_exp_patch[(exp_id, pid)] = extract_peak_features_xy(x, y, top_k=int(args.peak_top_k), prominence=float(args.peak_prominence))
                        loaded = True
                        break
                    except Exception as exc:
                        last_errors[str(pid)] = f'{raw}: {exc}'
                        continue
                if not loaded:
                    has_raman_by_exp_patch[(exp_id, pid)] = 0
            raman_debug_rows.append({
                'exp_id': int(exp_id),
                'exp_tag': exp_tag,
                'meta_map': _sample_map({str(pid): p2r.get(str(pid), '') for pid in patch_ids}),
                'auto_map': _sample_map(auto_p2r),
                'resolved': _sample_map(resolved),
                'wide_keys': _wide_keys_sample(args.raman_excel, raman_len, 'raman'),
                'last_errors': _sample_map(last_errors),
            })
            if args.print_sheet_map:
                print(f'[INFO] Raman mapping exp={exp_tag}: {resolved}')

        if args.xrd_excel:
            resolved = {}
            last_errors: Dict[str, str] = {}
            for pid in patch_ids:
                candidates = _candidate_sheet_names(p2x, auto_p2x, pid)
                resolved[str(pid)] = candidates[0] if candidates else ''
                if not candidates:
                    has_xrd_by_exp_patch[(exp_id, pid)] = 0
                    last_errors[str(pid)] = 'no meta candidate and no wide-order fallback candidate'
                    continue
                loaded = False
                for raw in candidates:
                    try:
                        spec = load_spectrum_from_workbook(args.xrd_excel, raw, exp_tag, new_len=xrd_len, kind='xrd', standardize=True)
                        xrd_by_exp_patch[(exp_id, pid)] = spec.astype(np.float32)
                        has_xrd_by_exp_patch[(exp_id, pid)] = int(np.any(np.abs(spec) > 1e-8))
                        loaded_xrd += int(has_xrd_by_exp_patch[(exp_id, pid)])
                        resolved[str(pid)] = raw
                        if args.peak_features:
                            x, y = load_xy_from_workbook(args.xrd_excel, raw, exp_tag, new_len=xrd_len, kind='xrd')
                            xrd_peaks_by_exp_patch[(exp_id, pid)] = extract_peak_features_xy(x, y, top_k=int(args.peak_top_k), prominence=float(args.peak_prominence))
                        loaded = True
                        break
                    except Exception as exc:
                        last_errors[str(pid)] = f'{raw}: {exc}'
                        continue
                if not loaded:
                    has_xrd_by_exp_patch[(exp_id, pid)] = 0
            xrd_debug_rows.append({
                'exp_id': int(exp_id),
                'exp_tag': exp_tag,
                'meta_map': _sample_map({str(pid): p2x.get(str(pid), '') for pid in patch_ids}),
                'auto_map': _sample_map(auto_p2x),
                'resolved': _sample_map(resolved),
                'wide_keys': _wide_keys_sample(args.xrd_excel, xrd_len, 'xrd'),
                'last_errors': _sample_map(last_errors),
            })
            if args.print_sheet_map:
                print(f'[INFO] XRD mapping exp={exp_tag}: {resolved}')

    if args.raman_excel and not args.allow_empty_spectra and loaded_raman == 0:
        raise ValueError(
            'Raman excel provided but no spectra were loaded. '
            'Check meta_json, workbook layout, or wide-sheet header order. '
            f'Debug: {json.dumps(raman_debug_rows, ensure_ascii=False)}'
        )
    if args.xrd_excel and not args.allow_empty_spectra and loaded_xrd == 0:
        raise ValueError(
            'XRD excel provided but no spectra were loaded. '
            'Check meta_json, workbook layout, or wide-sheet header order. '
            f'Debug: {json.dumps(xrd_debug_rows, ensure_ascii=False)}'
        )

    x0_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    raman_list: List[np.ndarray] = []
    xrd_list: List[np.ndarray] = []
    has_raman_list: List[int] = []
    has_xrd_list: List[int] = []
    raman_peaks_list: List[np.ndarray] = []
    xrd_peaks_list: List[np.ndarray] = []
    meta_patch: List[int] = []
    meta_t: List[int] = []
    meta_exp_id: List[int] = []
    meta_exp_tag: List[int] = []
    meta_exp_hum: List[float] = []
    sample_rows: List[Dict[str, object]] = []

    for exp_id, log_path in enumerate(rgb_logs):
        log = filter_rgb_log_by_humidity(parse_rgb_log_txt(log_path), tol=float(args.hum_tol))
        rgb = log.rgb
        hum_med = float(np.median(np.asarray(log.humidity, dtype=np.float32)))
        exp_tag = str(exp_meta[exp_id]['exp_tag'])
        side = str(exp_meta[exp_id]['side'])
        patch_idx = [pid - 1 for pid in patch_ids]
        lab_series = rgb_to_lab(rgb[:, patch_idx, :])
        lab0 = lab_series[0]
        for patch_offset, pid in enumerate(patch_ids):
            for t in range(1, rgb.shape[0]):
                seq = np.stack([lab0[patch_offset], lab_series[t, patch_offset]], axis=0)
                seq_n = lab_norm.normalize(seq).astype(np.float32)
                mask = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
                x0_list.append(seq_n)
                mask_list.append(mask)
                if args.raman_excel:
                    r = raman_by_exp_patch.get((exp_id, pid), np.zeros((raman_len,), dtype=np.float32))
                    hr = int(has_raman_by_exp_patch.get((exp_id, pid), 0))
                    raman_list.append(r.astype(np.float32))
                    has_raman_list.append(hr)
                    if args.peak_features:
                        raman_peaks_list.append(raman_peaks_by_exp_patch.get((exp_id, pid), np.zeros((2 * int(args.peak_top_k),), dtype=np.float32)).astype(np.float32))
                if args.xrd_excel:
                    x = xrd_by_exp_patch.get((exp_id, pid), np.zeros((xrd_len,), dtype=np.float32))
                    hx = int(has_xrd_by_exp_patch.get((exp_id, pid), 0))
                    xrd_list.append(x.astype(np.float32))
                    has_xrd_list.append(hx)
                    if args.peak_features:
                        xrd_peaks_list.append(xrd_peaks_by_exp_patch.get((exp_id, pid), np.zeros((2 * int(args.peak_top_k),), dtype=np.float32)).astype(np.float32))
                meta_patch.append(int(pid))
                meta_t.append(int(t))
                meta_exp_id.append(int(exp_id))
                meta_exp_tag.append(int(exp_tag))
                meta_exp_hum.append(float(hum_med))
                sample_idx = len(sample_rows)
                sequence_parent_id = f'{side}:{exp_id}:{pid}'
                spectral_parent_id = f'{exp_tag}:{pid}'
                sample_rows.append({
                    'sample_id': f'sample-{sample_idx:06d}',
                    'npz_row': sample_idx,
                    'source_log': log_path,
                    'exp_id': exp_id,
                    'exp_tag': exp_tag,
                    'side': side,
                    'patch_id': pid,
                    'time_index': t,
                    'sequence_parent_id': sequence_parent_id,
                    'spectral_parent_id': spectral_parent_id,
                    'augmentation_parent_id': sequence_parent_id,
                    'is_augmented': 0,
                    'split_group_id': spectral_parent_id,
                    'has_raman': int(has_raman_by_exp_patch.get((exp_id, pid), 0)),
                    'has_xrd': int(has_xrd_by_exp_patch.get((exp_id, pid), 0)),
                })

    arrays_all: Dict[str, np.ndarray] = {
        'x0': np.stack(x0_list, axis=0),
        'mask': np.stack(mask_list, axis=0),
        'meta_patch_id': np.asarray(meta_patch, dtype=np.int64),
        'meta_t': np.asarray(meta_t, dtype=np.int64),
        'meta_exp_id': np.asarray(meta_exp_id, dtype=np.int64),
        'meta_exp_tag': np.asarray(meta_exp_tag, dtype=np.int64),
        'meta_exp_humidity_median': np.asarray(meta_exp_hum, dtype=np.float32),
    }
    if args.raman_excel:
        arrays_all['raman'] = np.stack(raman_list, axis=0).astype(np.float32)
        arrays_all['meta_has_raman'] = np.asarray(has_raman_list, dtype=np.int64)
        arrays_all['has_raman'] = np.asarray(has_raman_list, dtype=np.int64)
        if args.peak_features:
            arrays_all['raman_peaks'] = np.stack(raman_peaks_list, axis=0).astype(np.float32)
    if args.xrd_excel:
        arrays_all['xrd'] = np.stack(xrd_list, axis=0).astype(np.float32)
        arrays_all['meta_has_xrd'] = np.asarray(has_xrd_list, dtype=np.int64)
        arrays_all['has_xrd'] = np.asarray(has_xrd_list, dtype=np.int64)
        if args.peak_features:
            arrays_all['xrd_peaks'] = np.stack(xrd_peaks_list, axis=0).astype(np.float32)

    n_total = int(arrays_all['x0'].shape[0])
    if args.split_mode == 'random':
        train_idx, val_idx, test_idx = random_split_indices(n_total, args.seed, args.val_ratio, args.test_ratio)
        split_detail = {'mode': 'random'}
    else:
        if args.split_mode == 'group_exp_patch':
            groups = [(int(e), int(p)) for e, p in zip(meta_exp_id, meta_patch)]
        elif args.split_mode == 'group_patch':
            groups = [(int(p),) for p in meta_patch]
        else:
            groups = [(int(e),) for e in meta_exp_id]
        train_idx, val_idx, test_idx = group_split_indices(groups, args.seed, args.val_ratio, args.test_ratio)
        split_detail = {'mode': args.split_mode, 'num_groups': len(set(groups))}

    os.makedirs(args.output_dir, exist_ok=True)
    _save_npz(os.path.join(args.output_dir, 'train.npz'), _slice_dict(arrays_all, train_idx))
    _save_npz(os.path.join(args.output_dir, 'val.npz'), _slice_dict(arrays_all, val_idx))
    _save_npz(os.path.join(args.output_dir, 'test.npz'), _slice_dict(arrays_all, test_idx))
    _save_npz(os.path.join(args.output_dir, 'all.npz'), arrays_all)

    def _select_rows(indices: Sequence[int]) -> List[Dict[str, object]]:
        return [sample_rows[int(i)] for i in indices]

    write_sample_index(os.path.join(args.output_dir, 'sample_index.csv'), sample_rows)
    write_sample_index(os.path.join(args.output_dir, 'train_index.csv'), _select_rows(train_idx))
    write_sample_index(os.path.join(args.output_dir, 'val_index.csv'), _select_rows(val_idx))
    write_sample_index(os.path.join(args.output_dir, 'test_index.csv'), _select_rows(test_idx))

    meta_out = {
        'experiments': exp_meta,
        'patch_ids': patch_ids,
        'raman_len': raman_len,
        'xrd_len': xrd_len,
        'loaded_raman_cnt': int(loaded_raman),
        'loaded_xrd_cnt': int(loaded_xrd),
        'peak_features': bool(args.peak_features),
        'peak_top_k': int(args.peak_top_k),
        'pair_only': True,
        'sidecar_index': True,
        'split': {'seed': int(args.seed), 'val_ratio': float(args.val_ratio), 'test_ratio': float(args.test_ratio), **split_detail},
        'n_total': n_total,
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'n_test': int(len(test_idx)),
    }
    with open(os.path.join(args.output_dir, 'preprocess_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta_out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
