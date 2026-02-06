### Underwater Image Enhancement — AI Summit 2026
## Overview

This project implements a deep learning pipeline for underwater image enhancement as part of the AI Summit 2026 Hackathon.

Underwater images suffer from:

Color distortion (blue/green dominance)

Light absorption and scattering

Low contrast

Reduced perceptual quality

Our goal is to restore visual clarity while preserving structural details.

## Objective

Develop a robust enhancement model to:

Correct color distortion

Reduce haze and scattering

Improve contrast and sharpness

Preserve structural information

## Model Architecture

We use a Residual U-Net architecture with:

Encoder–decoder structure

Skip connections

Instance normalization

Sigmoid output for normalized pixel range

Loss Function:

Total Loss = 0.7 * L1 + 0.3 * SSIM


# This balances:

Pixel-level accuracy (PSNR)

Structural similarity (SSIM)

# Evaluation Metrics

The model is evaluated using:

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

LPIPS (Perceptual similarity)

UCIQE (Underwater Image Quality Evaluation)

## Project Structure
```text
underwater-enhancement/
│
├── datasets/
│   ├── inputs/
│   ├── targets/
│
├── models/
│   └── unet.py
│
├── dataset.py
├── train.py
├── utils.py
├── requirements.txt
└── README.md


## Setup Instructions
1️) Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate     # Windows

2️) Install Dependencies
pip install -r requirements.txt


Or manually:

pip install torch torchvision torchaudio
pip install tqdm pillow pytorch-msssim lpips numpy matplotlib

# Training

Run:

python train.py


Training includes:

Automatic train/validation split (90/10)

Best model checkpoint saving

Validation SSIM monitoring

Best model is saved as:

best_model.pth

# GPU Support

Automatically detects GPU:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Optimized for:

RTX 40 series

NVIDIA A100 (DGX environment)

# Inference (Test Set)

After training:

Load best_model.pth

Run inference on test images

Save outputs with original filenames

Zip results for submission

## Author

Team VORTEX
AI Summit 2026 Participant 