# SpecBrush code release

This package contains the core implementation of **SpecBrush: A Two-Stage Diffusion Mural Restoration Method with Spectral Material Color Prior and Confidence Gating**.

Historical debugging branches, prototype-retrieval experiments, physics-cycle experiments, and GT-assisted output-selection paths are intentionally excluded from this release.

## Repository structure

```text
code/
  stage1/
    models/                  Material Color Inversion Network
    training/                masked diffusion + missing-modality training
    inference/               RGB-only inference + posterior uncertainty + Kalman-RTS
    pigment_task/            offline 33^3 LUT construction
    configs/lab_raman_xrd.json
  stage2/
    train/                    Stage-II training
    inference/                GT-free Stage-II deployment inference
```

## Code structure

- **Material Color Inversion Network**: `code/stage1/models/`
  - `ColorEncoder`: degraded-color encoder `E_c`
  - `MultimodalConditioner`: `Phi_m([E_c, E_r, E_x])`
  - `RGBConditionPredictor`: missing-modality branch `Psi_eta(E_c)`
  - `MambaDenoiser`: masked diffusion denoiser
- **Stage-I objective**: `code/stage1/training/trainer.py`
  - only `L_inv + lambda_align * L_align`
- **K-sample posterior + Kalman-RTS**: `code/stage1/inference/`
- **Offline LUT**: `code/stage1/pigment_task/build_pigment_lut33.py`
- **PriorControlNet / MP-Encoder / confidence-calibrated zero-residual injection**:
  `code/stage2/*/models/prior_controlnet.py` and `prior_control_wrapper.py`
- **MuCleaner**: `code/stage2/*/models/mu_denoiser.py`
- **MGLC**: `code/stage2/*/models/modules/mglc_block.py`

## Default protocol

Stage I uses Adam with learning rate `1e-4`, batch size `64`, and `300` epochs. The default RGB-only posterior uses `K=20` diffusion samples. Kalman-RTS smoothing is applied across adjacent aging time points during offline LUT construction. The released LUT builder uses a `33 x 33 x 33` RGB grid.

The image-space prior uses `alpha=0.85`, `beta=0.15`, `q_inp=0.3`, a distance-transform spatial confidence clipped to `[0.1, 1]`, and one-level Gaussian-pyramid smoothing (`sigma=2.0`) followed by bilateral filtering (`d=-1`, `sigmaColor=10`, `sigmaSpace=16`).

Stage II freezes the pretrained StrDiffusion texture backbone and optimizes only PriorControlNet, MuCleaner, and MGLC. It uses batch size `8`, `200,000` iterations, `T=200`, `sigma_max=30`, a cosine IR-SDE schedule, Adam with `beta1=0.9` and `beta2=0.99`, and MultiStepLR. The sole Stage-II optimization target is the pixel-space IR-SDE diffusion/noise-prediction objective; no perceptual, edge, structural, self-supervised denoising, TV, or high-frequency auxiliary loss is enabled.

The Stage-II trainable parameter count of the released control modules is **66.220797 M** (approximately **66.22 M**).

## Additional implementation defaults

The default implementation uses the following values: `p_full=0.7` and `lambda_align=0.1` for Stage-I missing-modality learning; Stage-II control-module learning rate `1e-6`; MultiStepLR milestones at `100k` and `150k`; MuCleaner hidden width `24` with `2` Transformer blocks; and MGLC `lambda_b=1.0`.

These values are implementation defaults.

## Commands

### Stage I training

```bash
cd code/stage1
python train.py --config configs/lab_raman_xrd.json
```

### Stage I RGB-only inference

```bash
python infer.py --ckpt <STAGE1_CHECKPOINT> --rgb "120,80,60"
```

Raman/XRD are not required by this deployment route.

### Stage I evaluation

```bash
python evaluate.py --ckpt <STAGE1_CHECKPOINT> --test_npz <TEST_NPZ> --condition rgb_only --num_samples 20
```

### Build the offline 33^3 LUT

```bash
python pigment_task/build_pigment_lut33.py \
  --ckpt <STAGE1_CHECKPOINT> \
  --aging_npz <ORDERED_LEADAGING_NPZ> \
  --num_samples 20 \
  --grid_size 33 \
  --device cuda \
  --out_npz pigment_lut33.npz
```

This command performs K-sample inversion and Kalman-RTS smoothing offline. They are not part of per-image Stage-II mural inference latency.

### Stage II training

```bash
cd code/stage2/train
python train.py -opt options/train/specbrush_train.yml
```

Before training, replace the dataset, LUT, and pretrained StrDiffusion placeholders in `options/train/specbrush_train.yml`.

### Stage II GT-free inference

```bash
cd code/stage2/inference
python test.py -opt options/test/specbrush_test.yml
```

The default inference configuration sets `dataroot_GT: null`. Ground truth is never used to construct the condition, choose an output, or alter the generated restoration.

## Important checkpoint compatibility note

The released Stage-II architecture may differ from earlier experimental branches. Checkpoints from incompatible branches should not be used with this release. Please use checkpoints produced by the same architecture and configuration.


