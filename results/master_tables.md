# ELEC 475 Lab 2 - Generated Results Tables

Values marked `Pending` require the corresponding experiment to be run.

## Table 7.3a - Computational Complexity

| Model | Conv depth | Normalization / regularization | Augmentation | Params (M) | FLOPs (M) | Mean epoch (s) | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Base LeNet-5 | 2 | None | None | 0.338 | 7.17 | 21.98 | 0.39 |
| 2. Modernized LeNet | 2 | ReLU + overlapping MaxPool | None | 0.446 | 9.19 | 22.33 | 0.66 |
| 3. AlexNet (No Norm) | 5 | Dropout (p=0.5) | None | 6.609 | 436.70 | 23.72 | 2.12 |
| 4. AlexNet + LRN | 5 | LRN + Dropout (p=0.5) | None | 6.609 | 436.70 | 25.94 | 1.94 |
| 5. AlexNet + BatchNorm | 5 | BatchNorm2d + Dropout (p=0.5) | None | 6.611 | 437.97 | 24.39 | 1.39 |
| 6. AlexNet + BN + Aug | 5 | BatchNorm2d + Dropout (p=0.5) | Crop + flip + jitter | 6.611 | 437.97 | 52.56 | 1.39 |
| 7. PyTorch Zoo AlexNet | 5 | Zoo model + Dropout | None | 57.045 | 188.20 | 24.93 | 2.15 |

## Table 7.3b - Predictive Performance

| Model | CIFAR-10 accuracy | CIFAR-10 macro F1 | Tiny ImageNet accuracy | Tiny ImageNet macro F1 |
| --- | --- | --- | --- | --- |
| 1. Base LeNet-5 | 54.69% | 54.28% | 11.78% | 9.72% |
| 2. Modernized LeNet | 66.36% | 66.11% | 20.01% | 17.94% |
| 3. AlexNet (No Norm) | 72.83% | 72.78% | 16.66% | 14.15% |
| 4. AlexNet + LRN | 73.46% | 73.46% | 0.50% | 0.00% |
| 5. AlexNet + BatchNorm | 78.23% | 78.20% | 30.63% | 29.11% |
| 6. AlexNet + BN + Aug | 81.26% | 81.17% | 28.91% | 26.94% |
| 7. PyTorch Zoo AlexNet | 66.74% | 66.72% | 0.53% | 0.01% |

## Augmentation and Generalization Gap

| Configuration | Final train accuracy | Final validation accuracy | Generalization gap |
| --- | --- | --- | --- |
| AlexNet BN, no augmentation | 90.34% | 82.14% | 8.20 percentage points |
| AlexNet BN, with augmentation | 84.38% | 79.73% | 4.65 percentage points |

## Multi-seed Results

| Experiment | Seeds | Validation accuracy | Macro F1 |
| --- | --- | --- | --- |
| alexnet_bn | 3 | 79.24 +/- 1.10% | 79.13 +/- 1.20% |
