"""Dataset and dataloader factories for official-compatible inference."""

import logging

import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    """Create a dataloader for train or test phases."""
    phase = dataset_opt["phase"]
    if phase == "train":
        if opt["dist"]:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt["n_workers"]
            assert dataset_opt["batch_size"] % world_size == 0
            batch_size = dataset_opt["batch_size"] // world_size
            shuffle = True
        else:
            num_workers = dataset_opt["n_workers"] * len(opt["gpu_ids"])
            batch_size = dataset_opt["batch_size"]
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(phase == "val"),
    )


def create_dataset(dataset_opt):
    """Instantiate one dataset by ``mode``."""
    mode = dataset_opt["mode"]
    if mode == "LQ":
        from data.LQ_dataset import LQDataset as dataset_cls
    elif mode == "LQGT":
        from data.LQGT_dataset import LQGTDataset as dataset_cls
    elif mode == "GT":
        from data.GT_dataset import GTDataset as dataset_cls
    elif mode == "SteLQGT":
        from data.StereoLQGT_dataset import StereoLQGTDataset as dataset_cls
    elif mode == "SteLQ":
        from data.StereoLQ_dataset import StereoLQDataset as dataset_cls
    elif mode == "BokehLQGT":
        from data.BokehLQGT_dataset import BokehLQGTDataset as dataset_cls
    elif mode == "BokehLQ":
        from data.BokehLQ_dataset import BokehLQDataset as dataset_cls
    elif mode == "mural_inference":
        from data.mural_inference_dataset import MuralInferenceDataset as dataset_cls
    else:
        raise NotImplementedError(f"Dataset [{mode}] is not recognized.")

    dataset = dataset_cls(dataset_opt)
    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_opt["name"]
        )
    )
    return dataset
