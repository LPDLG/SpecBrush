"""create dataset and dataloader"""
import logging

import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt["phase"]
    if phase == "train":
        if opt["dist"]:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt["n_workers"]
            assert dataset_opt["batch_size"] % world_size == 0
            batch_size = dataset_opt["batch_size"] // world_size
            shuffle = False
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
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=(phase=="val")
        )


def create_dataset(dataset_opt):
    mode = dataset_opt["mode"]
    if mode == "LQ":  # Predictor
        from data.LQ_dataset import LQDataset as D
        dataset = D(dataset_opt)
    elif mode == "LQGT":  # SFTMD
        from data.LQGT_dataset import LQGTDataset as D
        dataset = D(dataset_opt)
    elif mode == "GT":  # Corrector
        from data.GT_dataset import GTDataset as D
        dataset = D(dataset_opt)
    elif mode == 'SteLQGT':
        from data.StereoLQGT_dataset import StereoLQGTDataset as D
        dataset = D(dataset_opt)
    elif mode == 'SteLQ':
        from data.StereoLQ_dataset import StereoLQDataset as D
        dataset = D(dataset_opt)
    elif mode == 'BokehLQGT':
        from data.BokehLQGT_dataset import BokehLQGTDataset as D
        dataset = D(dataset_opt)
    elif mode == 'BokehLQ':
        from data.BokehLQ_dataset import BokehLQDataset as D
        dataset = D(dataset_opt)
    # ============================================================
    # 壁画修复数据集（新增，不影响原有代码）
    # ============================================================
    elif mode == 'mural_paired_inpainting':
        from data.mural_paired_inpainting_dataset import MuralPairedInpaintingDataset as D
        lut_path = dataset_opt.get('lut_path')
        if not lut_path:
            raise ValueError(
                "mural_paired_inpainting requires datasets.*.lut_path to be set explicitly."
            )
        gt_mode = dataset_opt.get('gt_mode', 'paired_true')
        prior_method = dataset_opt.get('prior_method', 'fast')
        debug_mode = dataset_opt.get('debug_mode', False)
        lut_alpha = dataset_opt.get('lut_alpha', 0.7)
        lut_beta = dataset_opt.get('lut_beta', 0.3)
        lut_inpaint_method = dataset_opt.get('lut_inpaint_method', 'telea')
        lut_delta_gain = dataset_opt.get('lut_delta_gain', 1.0)

        dataset = D(
            opt=dataset_opt,
            lut_path=lut_path,
            gt_mode=gt_mode,
            prior_method=prior_method,
            debug_mode=debug_mode,
            lut_alpha=lut_alpha,
            lut_beta=lut_beta,
            lut_inpaint_method=lut_inpaint_method,
            lut_delta_gain=lut_delta_gain,
        )
    elif mode == 'mural_inpainting':
        from data.mural_inpainting_dataset import MuralInpaintingDataset as D
        # 从 dataset_opt 中读取配置（配置现在在 datasets.train 下）
        lut_path = dataset_opt.get('lut_path')
        if not lut_path:
            raise ValueError(
                "mural_inpainting 数据集必须显式提供 datasets.*.lut_path，"
                "不再回退到旧的 pigment_lut33.npz 默认路径。"
            )
        gt_mode = dataset_opt.get('gt_mode', 'mixed')
        prior_method = dataset_opt.get('prior_method', 'fast')  # 新增：prior方法选择
        debug_mode = dataset_opt.get('debug_mode', False)
        lut_alpha = dataset_opt.get('lut_alpha', 0.7)
        lut_beta = dataset_opt.get('lut_beta', 0.3)
        lut_inpaint_method = dataset_opt.get('lut_inpaint_method', 'telea')
        lut_delta_gain = dataset_opt.get('lut_delta_gain', 1.0)
        
        dataset = D(
            opt=dataset_opt,
            lut_path=lut_path,
            gt_mode=gt_mode,
            prior_method=prior_method,  # 新增参数
            debug_mode=debug_mode,
            lut_alpha=lut_alpha,
            lut_beta=lut_beta,
            lut_inpaint_method=lut_inpaint_method,
            lut_delta_gain=lut_delta_gain
        )
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))

    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_opt["name"]
        )
    )
    return dataset
