"""SpecBrush training entry point for SpecBrush Stage II.

Default training protocol:
- frozen pretrained StrDiffusion backbone;
- train PriorControlNet, MuCleaner and MGLC only;
- pixel-space IR-SDE noise-prediction objective only;
- batch size 8, 200,000 iterations, T=200, sigma_max=30, cosine schedule;
- Adam (beta1=0.9, beta2=0.99) with MultiStepLR.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random

import numpy as np
import torch

import options as option
import str_utils as str_util
import utils as util
from data import create_dataloader, create_dataset
from models import create_model


def _setup_logger(opt):
    os.makedirs(opt["path"]["experiments_root"], exist_ok=True)
    os.makedirs(opt["path"]["models"], exist_ok=True)
    os.makedirs(opt["path"]["training_state"], exist_ok=True)
    util.setup_logger(
        "base",
        opt["path"]["log"],
        "train_" + opt["name"],
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))
    return logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_train_loader(opt):
    if "train" not in opt["datasets"]:
        raise ValueError("datasets.train is required")
    dataset_opt = opt["datasets"]["train"]
    dataset = create_dataset(dataset_opt)
    loader = create_dataloader(dataset, dataset_opt, opt, sampler=None)
    if len(loader) == 0:
        raise RuntimeError("training dataloader is empty")
    return dataset, loader


def _move(batch, key, device, required=True):
    value = batch.get(key, None)
    if value is None:
        if required:
            raise KeyError(f"training batch is missing required key: {key}")
        return None
    return value.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        default="options/train/specbrush_train.yml",
        type=str,
        help="Path to the released Stage-II YAML config.",
    )
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)
    opt["dist"] = False

    seed = int(opt["train"].get("manual_seed", 0) or 0)
    _set_seed(seed)
    torch.backends.cudnn.benchmark = True
    logger = _setup_logger(opt)

    dataset, train_loader = _get_train_loader(opt)
    total_iters = int(opt["train"]["niter"])
    total_epochs = int(math.ceil(total_iters / max(len(train_loader), 1)))
    logger.info(
        "SpecBrush Stage II: images=%d, batches/epoch=%d, total_iters=%d, approx_epochs=%d",
        len(dataset), len(train_loader), total_iters, total_epochs,
    )

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

    current_step = 0
    start_epoch = 0
    resume_path = opt["path"].get("resume_state", None)
    if resume_path:
        resume = torch.load(resume_path, map_location=device)
        current_step = int(resume["iter"])
        start_epoch = int(resume["epoch"])
        model.resume_training(resume)
        logger.info("Resumed from epoch=%d iter=%d", start_epoch, current_step)

    print_freq = int(opt["logger"].get("print_freq", 100))
    save_freq = int(opt["logger"].get("save_checkpoint_freq", 10000))
    warmup_iter = int(opt["train"].get("warmup_iter", -1) or -1)

    for epoch in range(start_epoch, total_epochs + 1):
        for batch in train_loader:
            if current_step >= total_iters:
                break
            current_step += 1

            degraded = _move(batch, "degraded", device)
            target = _move(batch, "GT", device)
            mask_hole = _move(batch, "mask", device).clamp(0.0, 1.0)
            mask_known = 1.0 - mask_hole
            color_prior = _move(batch, "color_prior", device)
            confidence = _move(batch, "confidence", device)
            structure_gt = _move(batch, "GT_gray", device)
            structure_lq = _move(batch, "GT_edge", device)

            # Eq. (11): MuCleaner replaces the original masked degraded mean.
            # Keep this computation in the graph so MuCleaner is optimized by
            # the same Stage-II diffusion objective as the other control modules.
            raw_mu = degraded * mask_known
            mu_clean = model.compute_mu_clean(raw_mu, mask_hole)

            # IR-SDE forward state, using the purified conditional mean.
            timesteps, states = sde.generate_random_states(x0=target, mu=mu_clean)

            model.feed_data(
                states,
                mu_clean,
                target,
                mask_known,
                structure_sde,
                structure_gt,
                structure_lq,
                color_prior=color_prior,
                confidence=confidence,
                original_degraded=degraded,
            )
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(current_step, warmup_iter=warmup_iter)

            if current_step % print_freq == 0 or current_step == 1:
                logs = model.get_current_log()
                message = (
                    f"<epoch:{epoch:4d}, iter:{current_step:8d}, "
                    f"lr:{model.get_current_learning_rate():.3e}> "
                    + " ".join(f"{k}:{float(v):.6e}" for k, v in logs.items())
                )
                logger.info(message)

            if save_freq > 0 and current_step % save_freq == 0:
                logger.info("Saving checkpoint at iter=%d", current_step)
                model.save(str(current_step))
                model.save_training_state(epoch, current_step, label=str(current_step))

        if current_step >= total_iters:
            break

    logger.info("Saving final Stage-II checkpoint at iter=%d", current_step)
    model.save("latest")
    model.save_training_state(total_epochs, current_step, label="latest")
    logger.info("Stage-II training completed.")


if __name__ == "__main__":
    main()
