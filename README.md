# RCAN: Residual Channel Attention Network for Semiconductor Image Restoration

[![Kaggle Hackathon](https://img.shields.io/badge/Kaggle-AI%20Hackathon-blue.svg)](https://kaggle.com)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning pipeline built for **Single-Channel (Grayscale) Image Super-Resolution and Denoising** targeting semiconductor microscopy patterns for the KLA AI Hackathon (March 14–18, 2026).

---

## 🎯 Challenge Overview & Objective

The objective of this challenge is to restore single-channel images degraded by speckled noise, blur, and low spatial resolution, reconstructing them back to near-original quality at **256×256** spatial resolution.

Solutions are benchmarked on both restoration quality and inference optimization:
* **Primary Metrics:** Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Metric (SSIM).
* **Key Challenges:** Preserving fine-grain semiconductor edge features, handling varying degradation severity, and running optimized inference within Kaggle resource limits.

---

## 🏗️ Model Architecture & Specs

* **Model:** Residual Channel Attention Network (RCAN) right-sized for small/medium scientific datasets.
* **Parameters:** ~3.86M–4.09M trainable parameters.
* **Configuration:** 5 Residual Groups $\times$ 10 Residual Channel Attention Blocks (RCABs) = **50 total RCABs**.
* **Feature Dimension (`n_feats`):** 64 channels.
* **Reduction Ratio (`reduction`):** 16 (Squeeze-and-Excitation style Channel Attention).
* **Upscaling Scale Factor:** **2$\times$** (PixelShuffle from 128×128 patches to 256×256 outputs).

---

## 🚀 Key Pipeline Features & Innovations

* **Hybrid Reconstruction Loss:** $0.7 \times \text{Charbonnier Loss} + 0.3 \times (1 - \text{SSIM})$ balancing pixel-level fidelity and structural edge preservation.
* **Data Pipeline:** Direct float32 $[0, 1]$ range tensor processing with random sub-patch extraction (64×64 LR to 128×128 HR) and spatial augmentations (horizontal/vertical flips, $90^\circ$ rotations).
* **Optimization Strategy:** Adam optimizer with linear warmup (5 epochs) and Cosine Annealing learning rate decay down to $1\text{e-}6$.
* **Validation & Early Stopping:** PSNR evaluation on a 10% validation split after every epoch, saving best checkpoints with a patience threshold of 25 epochs.
* **Inference Strategy:** **8$\times$ Test-Time Augmentation (x8 TTA)** combining spatial flips and rotational symmetry transformations with inverse mapping.
* **Submission Serialization:** Converts output arrays into `.npy` files and serializes them into a base64-encoded `submission.csv` compliant with competition submission rules.

---

## 📂 Directory Structure & Submission Format

### Submission Output Specifications
* Reads noisy LR input images from `/kaggle/input/competition-data/test/NoisyLR/`.
* Output shape per image: 2D NumPy array (**256×256**), float32 in $[0, 1]$ range.
* Individual predictions saved to `/kaggle/working/submission/*.npy`.
* Serialized base64 submission generated at `/kaggle/working/submission.csv`.

```text
├── input/
│   └── competition-data/
│       └── test/
│           └── NoisyLR/          # Input noisy images
├── working/
│   ├── submission/
│   │   └── *.npy                 # Individual 256x256 float32 arrays
│   └── submission.csv            # Final base64-encoded submission file
├── src/
│   ├── model.py                  # RCAN architecture definition
│   ├── dataset.py                # Data loading & patch extraction
│   ├── loss.py                   # Hybrid Charbonnier + SSIM loss
│   └── utils.py                  # TTA & serialization utilities
└── train.py                      # Training loop & validation script
