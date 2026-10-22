"""
ELEC 475: Lab 2 - Model Complexity & Benchmark Comparison Tool
Compares FLOPs, MACs, Parameters, Latency, and Throughput across:
  1. LeNet-5 (Base LeCun et al. 1998)
  2. LeNet-Modern (ReLU + Overlapping MaxPool)
  3. AlexNet-64 (No Norm)
  4. AlexNet-64 (Local Response Normalization - LRN)
  5. AlexNet-64 (Batch Normalization - BatchNorm2d)
  6. Torchvision AlexNet (PyTorch Model Zoo Reference Baseline)

Academic Reference:
  He et al. (CVPR 2016), "Deep Residual Learning for Image Recognition";
  Howard et al. (2017), "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications".
"""
import argparse
import os
import torch

from models import LeNet5, LeNetModern, AlexNet64, get_torchvision_alexnet
from utils.profiler import ModelProfiler, benchmark_inference, compare_model_complexities

def main():
    parser = argparse.ArgumentParser(description="ELEC 475: Model Complexity Benchmark (FLOPs & Parameters)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "tiny_imagenet"],
                        help="Target dataset: 'cifar10' (10 classes) or 'tiny_imagenet' (200 classes)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for complexity profiling (default: 1)")
    parser.add_argument("--reps", type=int, default=50, help="Number of benchmark iterations for latency testing")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Profiling device. 'auto' selects CUDA when available.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Optional text file for the generated comparison table")
    args = parser.parse_args()

    num_classes = 10 if args.dataset == "cifar10" else 200
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    input_size = (args.batch_size, 3, 64, 64)

    print("=" * 75)
    print("ELEC 475: Lab 2 - Comprehensive Model Complexity & FLOPs Benchmark")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Dataset       : {args.dataset.upper()} ({num_classes} classes)")
    print(f"Input Shape   : {input_size}")
    print("=" * 75)

    print("\nInstantiating models for profiling...")
    models_dict = {
        "1. LeNet-5 (Base)": LeNet5(num_classes=num_classes),
        "2. LeNet-Modern (ReLU+Pool)": LeNetModern(num_classes=num_classes),
        "3. AlexNet-64 (None)": AlexNet64(num_classes=num_classes, norm_type="none"),
        "4. AlexNet-64 (LRN)": AlexNet64(num_classes=num_classes, norm_type="lrn"),
        "5. AlexNet-64 (BatchNorm)": AlexNet64(num_classes=num_classes, norm_type="bn"),
        "6. Torchvision Zoo AlexNet": get_torchvision_alexnet(num_classes=num_classes),
    }

    print("Profiling FLOPs, Multiply-Accumulates, and Latency...")
    table_md = compare_model_complexities(
        models_dict,
        input_size=input_size,
        device=device,
        reps=args.reps,
    )

    print("\n" + table_md + "\n")
    print("=" * 75)

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as handle:
            handle.write(table_md + "\n")
        print(f"Comparison table saved to: {args.output_file}")
    print("Academic Notes on Complexity Metrics:")
    print("1. FLOPs vs MACs: 1 Multiply-Accumulate (MAC) corresponds to ~2 Floating Point Operations (FLOPs).")
    print("2. Memory vs Compute: Dense Linear layers contribute the majority of parameters, but Conv layers dominate FLOPs.")
    print("3. Reference: He et al. (CVPR 2016), Howard et al. (MobileNets 2017).")
    print("=" * 75)

if __name__ == "__main__":
    main()
