# SpecBrush dataset

## Project Profile

This repository is used for presentations on the SpecBrush dataset. Here, we provide representative mural restoration images, material color degradation data descriptions, and experimental examples for the SpecBrush project.

## SpecBrush

With the continuous breakthroughs in generative artificial intelligence and diffusion-based image restoration technologies, mural virtual restoration has become an important research topic in cultural heritage preservation. However, most existing image restoration methods mainly depend on degraded RGB appearances or generic generative priors, and they often ignore the material degradation mechanism behind mural color changes. For murals containing lead-based pigments, similar RGB appearances may correspond to different pigment compositions, crystalline phase structures, and degradation pathways. This makes target color recovery ambiguous and may result in tone shifts, boundary artifacts, fine-line discontinuities, and unstable extension of repeated mural patterns.

In this paper, we propose **SpecBrush**, a two-stage diffusion mural restoration method with spectral material color prior and confidence gating. The proposed method first constructs a Material Color Inversion Network using RGB color evolution sequences, Raman spectra, and XRD patterns to learn material-conditioned color degradation relationships. The learned color recovery information is further transformed into an image-space color prior map and a confidence map. To support practical mural restoration, where region-wise Raman and XRD measurements are usually unavailable, a missing-modality conditional learning strategy is introduced to enable RGB-only inference while preserving the benefits of spectral-material supervision.

Based on the color prior generated in the first stage, the second stage performs confidence-gated diffusion restoration. PriorControlNet encodes the color prior map, confidence map, missing mask, and confidence-weighted color residual into a unified material-prior feature space, and injects multi-scale control residuals into a frozen diffusion backbone. MuCleaner purifies the conditional mean in the reverse sampling process to reduce the attraction of degraded colors and boundary contamination. MGLC enhances mask-gated local continuity to improve damaged-boundary transitions, fine-line preservation, and repeated-pattern continuity. Through these designs, SpecBrush aims to restore mural images with better color consistency, perceptual quality, and structural continuity.

The experimental data used in this project include MuralVerse-S, CanvasCLP, and Leadaging. MuralVerse-S contains publicly available mural images from different regions and is used for mural restoration evaluation. CanvasCLP contains Chinese painting images and is used to test the applicability of the proposed method in traditional painting restoration scenarios. Leadaging records RGB color sequences, Raman spectra, and XRD patterns of lead-pigment samples during the aging process, and is used to train the material color inversion stage. Extensive experiments demonstrate that SpecBrush achieves favorable restoration performance compared with representative CNN-based, Transformer-based, and diffusion-based restoration methods.

## code

We will upload the training code, testing code, pretrained models, and full dataset at an appropriate time.
