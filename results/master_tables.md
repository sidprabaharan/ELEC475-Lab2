# ELEC 475 Lab 2 - Generated Results Tables

Values marked `Pending` require the corresponding experiment to be run.

## Table 7.3a - Computational Complexity

| Model | Conv depth | Normalization / regularization | Augmentation | Params (M) | FLOPs (M) | Mean epoch (s) | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Base LeNet-5 | 2 | None | None | Pending | Pending | Pending | Pending |
| 2. Modernized LeNet | 2 | ReLU + overlapping MaxPool | None | Pending | Pending | Pending | Pending |
| 3. AlexNet (No Norm) | 5 | Dropout (p=0.5) | None | Pending | Pending | Pending | Pending |
| 4. AlexNet + LRN | 5 | LRN + Dropout (p=0.5) | None | Pending | Pending | Pending | Pending |
| 5. AlexNet + BatchNorm | 5 | BatchNorm2d + Dropout (p=0.5) | None | Pending | Pending | Pending | Pending |
| 6. AlexNet + BN + Aug | 5 | BatchNorm2d + Dropout (p=0.5) | Crop + flip + jitter | Pending | Pending | Pending | Pending |
| 7. PyTorch Zoo AlexNet | 5 | Zoo model + Dropout | None | Pending | Pending | Pending | Pending |

## Table 7.3b - Predictive Performance

| Model | CIFAR-10 accuracy | CIFAR-10 macro F1 | Tiny ImageNet accuracy | Tiny ImageNet macro F1 |
| --- | --- | --- | --- | --- |
| 1. Base LeNet-5 | Pending | Pending | Pending | Pending |
| 2. Modernized LeNet | Pending | Pending | Pending | Pending |
| 3. AlexNet (No Norm) | Pending | Pending | Pending | Pending |
| 4. AlexNet + LRN | Pending | Pending | Pending | Pending |
| 5. AlexNet + BatchNorm | Pending | Pending | Pending | Pending |
| 6. AlexNet + BN + Aug | Pending | Pending | Pending | Pending |
| 7. PyTorch Zoo AlexNet | Pending | Pending | Pending | Pending |

## Augmentation and Generalization Gap

| Configuration | Final train accuracy | Final validation accuracy | Generalization gap |
| --- | --- | --- | --- |
| AlexNet BN, no augmentation | Pending | Pending | Pending |
| AlexNet BN, with augmentation | Pending | Pending | Pending |

## Multi-seed Results

Pending: run at least two seeds for an experiment.
