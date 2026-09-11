# -*- coding: utf-8 -*-
"""Option parsing helpers for official-compatible inference.

Usage:
    python test.py -opt options/test/ir-sde-brushnet.yml
    python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.enabled=false

Mask semantics:
    This file only parses configuration. Runtime mask semantics are handled in
    the dataset/model code with explicit ``mask_hole`` and ``mask_known`` names.
"""

import logging
import os
import os.path as osp
from typing import Iterable, Optional

import yaml

from file_utils import OrderedYaml

Loader, Dumper = OrderedYaml()


def _apply_override(opt: dict, override: str) -> None:
    """Apply one dotted ``key=value`` override in-place."""
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected key=value.")

    key, raw_value = override.split("=", 1)
    value = yaml.safe_load(raw_value)
    cursor = opt
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse(opt_path: str, is_train: bool = True, overrides: Optional[Iterable[str]] = None):
    """Parse a YAML option file and apply optional CLI overrides."""
    with open(opt_path, mode="r", encoding="utf-8") as handle:
        opt = yaml.load(handle, Loader=Loader)

    if overrides:
        for override in overrides:
            _apply_override(opt, override)

    gpu_ids = opt.get("gpu_ids", [])
    gpu_list = ",".join(str(x) for x in gpu_ids)
    print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    opt["is_train"] = is_train

    scale = 1
    if opt["distortion"] == "sr":
        scale = opt["degradation"]["scale"]
        opt["network_G"]["setting"]["upscale"] = scale

    for phase, dataset in opt["datasets"].items():
        phase_name = phase.split("_")[0]
        dataset["phase"] = phase_name
        dataset["scale"] = scale

        is_lmdb = False
        for key in (
            "dataroot_GT",
            "dataroot_LQ",
            "dataroot_degraded",
            "dataroot_mask",
            "dataroot_color_prior",
            "dataroot_confidence",
        ):
            if dataset.get(key):
                dataset[key] = osp.expanduser(dataset[key])
                if str(dataset[key]).endswith("lmdb"):
                    is_lmdb = True

        dataset["data_type"] = "lmdb" if is_lmdb else "img"
        if dataset["mode"].endswith("mc"):
            dataset["data_type"] = "mc"
            dataset["mode"] = dataset["mode"].replace("_mc", "")

    for key, path in opt["path"].items():
        if path and key != "strict_load":
            opt["path"][key] = osp.expanduser(path)

    opt["path"]["root"] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir)
    )
    config_dir = osp.basename(osp.dirname(osp.abspath(__file__)))
    if is_train:
        experiments_root = osp.join(
            opt["path"]["root"], "experiments", config_dir, opt["name"]
        )
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["training_state"] = osp.join(experiments_root, "training_state")
        opt["path"]["log"] = experiments_root
        opt["path"]["val_images"] = osp.join(experiments_root, "val_images")
        if "debug" in opt["name"]:
            opt["train"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:
        results_root = osp.join(opt["path"]["root"], "results", config_dir)
        opt["path"]["results_root"] = osp.join(results_root, opt["name"])
        opt["path"]["log"] = osp.join(results_root, opt["name"])

    return opt


def dict2str(opt, indent_l=1):
    """Render a nested option dictionary for logging."""
    msg = ""
    for key, value in opt.items():
        if isinstance(value, dict):
            msg += " " * (indent_l * 2) + key + ":[\n"
            msg += dict2str(value, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + key + ": " + str(value) + "\n"
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    """Convert nested dict/list structures into ``NoneDict`` recursively."""
    if isinstance(opt, dict):
        new_opt = {}
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    if isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    return opt


def check_resume(opt, resume_iter):
    """Keep resume handling compatible with the original codebase."""
    logger = logging.getLogger("base")
    if opt["path"]["resume_state"]:
        if (
            opt["path"].get("pretrain_model_G", None) is not None
            or opt["path"].get("pretrain_model_D", None) is not None
        ):
            logger.warning(
                "pretrain_model path will be ignored when resuming training."
            )

        opt["path"]["pretrain_model_G"] = osp.join(
            opt["path"]["models"], f"{resume_iter}_G.pth"
        )
        logger.info("Set [pretrain_model_G] to " + opt["path"]["pretrain_model_G"])
        if "gan" in opt["model"]:
            opt["path"]["pretrain_model_D"] = osp.join(
                opt["path"]["models"], f"{resume_iter}_D.pth"
            )
            logger.info("Set [pretrain_model_D] to " + opt["path"]["pretrain_model_D"])
