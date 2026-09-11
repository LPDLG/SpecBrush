"""SpecBrush, GT-free SpecBrush Stage-II inference entry point."""
from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
from PIL import Image
import torch

import options as option
import str_utils as str_util
import utils as util
from data import create_dataloader, create_dataset
from models import create_model


def _save_rgb(tensor: torch.Tensor, path: str) -> None:
    x = tensor.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = x.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB").save(path)


def _save_gray(tensor: torch.Tensor, path: str) -> None:
    x = tensor.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    if x.dim() == 3:
        x = x[0]
    arr = (x.clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        default="options/test/specbrush_test.yml",
        type=str,
        help="SpecBrush SpecBrush inference config.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Optional dotted key=value override; may be repeated.",
    )
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False, overrides=args.overrides)
    opt = option.dict_to_nonedict(opt)
    os.makedirs(opt["path"]["results_root"], exist_ok=True)
    util.setup_logger(
        "base", opt["path"]["log"], "test_" + opt["name"],
        level=logging.INFO, screen=True, tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    if int(opt["sde"]["T"]) != 200:
        raise ValueError("Stage II requires 200 training/inference steps.")
    if int(opt.get("inference", {}).get("sampling_steps", 200)) != 200:
        raise ValueError("Stage-II inference uses 200 sampling steps.")

    model = create_model(opt)
    device = model.device

    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device,
    )
    sde.set_model(model.model)

    structure_sde = str_util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device,
    )
    structure_sde.set_model(model.models)

    save_intermediates = bool(opt.get("inference", {}).get("save_intermediates", False))
    for _, dataset_opt in opt["datasets"].items():
        dataset = create_dataset(dataset_opt)
        loader = create_dataloader(dataset, dataset_opt, opt)
        result_dir = os.path.join(opt["path"]["results_root"], dataset_opt["name"])
        os.makedirs(result_dir, exist_ok=True)

        logger.info("Testing %s samples from %s", len(dataset), dataset_opt["name"])
        started = time.time()
        for index, sample in enumerate(loader, start=1):
            stem = sample["stem"][0]
            degraded = sample["degraded"]
            mask_known = sample["mask_known"]
            mask_hole = sample["mask_hole"]
            prior = sample.get("color_prior", None)
            confidence = sample.get("confidence", None)

            # GT, when optionally present for a separate metric run, is never
            # passed into the model and cannot affect the generated output.
            model.feed_data(
                degraded=degraded,
                mask_known=mask_known,
                mask_hole=mask_hole,
                color_prior=prior,
                confidence=confidence,
                sample_name=stem,
            )
            output = model.test(sde, structure_sde)
            _save_rgb(output, os.path.join(result_dir, f"{stem}.png"))

            if save_intermediates:
                sample_dir = os.path.join(result_dir, f"{stem}_intermediates")
                os.makedirs(sample_dir, exist_ok=True)
                visuals = model.get_current_visuals()
                for key in ("degraded", "color_prior", "mu_c", "output"):
                    if key in visuals:
                        _save_rgb(visuals[key], os.path.join(sample_dir, f"{key}.png"))
                for key in ("mask_hole", "confidence"):
                    if key in visuals:
                        _save_gray(visuals[key], os.path.join(sample_dir, f"{key}.png"))

            logger.info("[%d/%d] %s", index, len(dataset), stem)

        logger.info(
            "Finished %s in %.2f s", dataset_opt["name"], time.time() - started
        )


if __name__ == "__main__":
    main()
