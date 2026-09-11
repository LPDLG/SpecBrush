"""Configuration loading for the released Stage-I release."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict


def _resolve_path(base_dir: str, value: str) -> str:
    if not value or os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return normalize_config(json.load(handle), config_path=path)


def normalize_config(cfg: Dict[str, Any], config_path: str = "") -> Dict[str, Any]:
    cfg = deepcopy(cfg)
    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    project_root = os.path.abspath(os.path.join(base_dir, "..")) if os.path.basename(base_dir) == "configs" else os.getcwd()

    data = cfg.setdefault("data", {})
    for key in ("train_npz", "val_npz", "test_npz", "train_index", "val_index", "test_index"):
        if isinstance(data.get(key), str):
            data[key] = _resolve_path(project_root, data[key])

    train = cfg.setdefault("train", {})
    if isinstance(train.get("save_dir"), str):
        train["save_dir"] = _resolve_path(project_root, train["save_dir"])

    diffusion = cfg.setdefault("diffusion", {})
    if "beta_start" in diffusion and "beta_0" not in diffusion:
        diffusion["beta_0"] = diffusion["beta_start"]
    if "beta_end" in diffusion and "beta_T" not in diffusion:
        diffusion["beta_T"] = diffusion["beta_end"]

    missing = cfg.setdefault("missing_modality", {})
    missing.setdefault("enable", True)
    missing.setdefault("p_full", 0.7)
    missing.setdefault("lambda_align", 0.1)

    inference = cfg.setdefault("inference", {})
    inference.setdefault("condition", "rgb_only")
    inference.setdefault("num_samples", 20)
    inference.setdefault("use_kalman_rts_for_ordered_aging_sequences", True)
    return cfg
