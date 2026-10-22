# ELEC 475: Lab 2 — Completed Implementation
## From LeNet-5 to AlexNet: An Architectural & Empirical Evolution

This repository contains the completed in-lab and post-lab implementation, reproducible experiment harness, measured artifacts, and analytical report for ELEC 475 Lab 2.

### Directory Structure

```text
lab2_template/
├── configs/
│   └── config.py          # Centralized configuration dataclass (ExperimentConfig)
├── models/
│   ├── lenet.py           # Base LeNet-5 for 64×64 RGB input
│   ├── lenet_modern.py    # ReLU + overlapping-max-pool LeNet
│   ├── alexnet.py         # Five-stage AlexNet-64 with None/LRN/BatchNorm
│   └── zoo.py             # PyTorch Model Zoo AlexNet adapter
├── utils/
│   ├── dataset.py         # Automated CIFAR-10 and Tiny ImageNet loaders
│   ├── eda.py             # Exploratory Data Analysis tool (class distributions, RGB stats, imbalance ratio)
│   ├── transforms.py      # Deterministic validation and stochastic training pipelines
│   ├── metrics.py         # Loss, Top-1 accuracy, Macro/Micro F1, ConfusionMatrixTracker & EarlyStopping
│   ├── seed.py            # Global deterministic seeding across Python, NumPy, PyTorch (CPU/CUDA)
│   ├── visualization.py   # Training curve and ablation plotters
│   └── profiler.py        # Timer (CUDA sync), ModelProfiler (FLOPs/MACs/Params)
├── compare_models.py      # Multi-model complexity & FLOPs benchmarking CLI script
├── demo_transforms.py     # Pre-Lab benchmark: PyTorch GPU Tensors vs OpenCV vs NumPy
├── train.py               # Complete CLI training engine with Timer, ModelProfiler, Macro F1 & Early Stopping
├── evaluate.py            # Checkpoint evaluation with full classification report
├── run_experiments.py     # Seed-aware required and full experiment ladders
├── summarize_results.py   # Generates the master tables from saved JSON artifacts
├── commands.txt           # Exact commands for every submitted experiment
└── requirements.txt       # Python package dependencies
```

---

### Reproduction Guide

#### 1. Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell (or source .venv/bin/activate on Linux/WSL/macOS)
pip install -r requirements.txt
```

#### 2. Run Pre-Lab Exploratory Data Analysis & Verification
```powershell
# 1. Run Comprehensive Dual-Dataset Exploratory Data Analysis (EDA):
python utils/eda.py --dataset both

# 2. Run Automated Environment & Dataset Verification:
python prelab_verify.py
```

#### 3. In-Lab Models
- Verify the architectures with `python -m models.lenet` and `python -m models.lenet_modern`.
- Train Base LeNet (real-time epoch timers and FLOPs are logged):
  ```powershell
  python train.py --model lenet --dataset cifar10 --epochs 5 --save-model checkpoints/lenet_baseline.pth --plot-file plots/loss_lenet.png
  ```
- Train Modernized LeNet:
  ```powershell
  python train.py --model lenet_modern --dataset cifar10 --epochs 5 --save-model checkpoints/lenet_modern.pth --plot-file plots/loss_lenet_modern.png
  ```

The oral-checkoff spatial math, activation analysis, and measured convergence comparison are included in `ELEC475_Lab2_Report.pdf`.

#### 4. Post-Lab Experiment Matrix

Run the required experiments in Colab or another CUDA environment:

```powershell
python run_experiments.py --phase full --device cuda --seeds 42
```

The equivalent individual commands, including the 15-epoch augmentation comparison and supplemental BatchNorm seeds 123 and 2026, are recorded in `commands.txt`. Example normalization commands are:

```powershell
python train.py --model alexnet --norm none --dataset cifar10 --epochs 10 --seed 42 --no-augment --save-model checkpoints/alexnet_none.pth --plot-file plots/loss_alexnet_none.png
python train.py --model alexnet --norm lrn --dataset cifar10 --epochs 10 --seed 42 --no-augment --save-model checkpoints/alexnet_lrn.pth --plot-file plots/loss_alexnet_lrn.png
python train.py --model alexnet --norm bn --dataset cifar10 --epochs 10 --seed 42 --no-augment --save-model checkpoints/alexnet_bn.pth --plot-file plots/loss_alexnet_bn.png
```

Generate or refresh the tables and comparative plot after training:

```powershell
python summarize_results.py
python -m utils.visualization --plot-ablation
python compare_models.py --dataset cifar10 --device cuda
```

Datasets are downloaded into `data/` when absent. That directory, local environments, caches, and rendered report QA images are intentionally excluded from version control and the submission archive.
