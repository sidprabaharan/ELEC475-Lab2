"""Reproducible experiment runner for the ELEC 475 Lab 2 report.

The runner keeps the assignment's required experiment settings in one place and
creates a checkpoint, plot, training-history JSON, and evaluation JSON for each
run. Existing complete runs are skipped unless ``--force`` is supplied.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Experiment:
    key: str
    model: str
    dataset: str
    epochs: int
    norm: str = "bn"
    augment: bool = False


INLAB = (
    Experiment("lenet_baseline", "lenet", "cifar10", 5),
    Experiment("lenet_modern", "lenet_modern", "cifar10", 5),
)

CIFAR_POSTLAB = (
    Experiment("alexnet_none", "alexnet", "cifar10", 10, norm="none"),
    Experiment("alexnet_lrn", "alexnet", "cifar10", 10, norm="lrn"),
    Experiment("alexnet_bn", "alexnet", "cifar10", 10, norm="bn"),
    Experiment("alexnet_no_aug", "alexnet", "cifar10", 15, norm="bn"),
    Experiment("alexnet_with_aug", "alexnet", "cifar10", 15, norm="bn", augment=True),
    Experiment("zoo_alexnet", "zoo_alexnet", "cifar10", 15),
)

TINY_CORE = (
    Experiment("alexnet_tiny_imagenet", "alexnet", "tiny_imagenet", 10, norm="bn", augment=True),
)

# These additional runs are needed only if every cell in Table 7.3b is to be
# populated for Tiny ImageNet. They are separate because they are GPU-expensive.
TINY_FULL = (
    Experiment("lenet_tiny", "lenet", "tiny_imagenet", 10),
    Experiment("lenet_modern_tiny", "lenet_modern", "tiny_imagenet", 10),
    Experiment("alexnet_none_tiny", "alexnet", "tiny_imagenet", 10, norm="none"),
    Experiment("alexnet_lrn_tiny", "alexnet", "tiny_imagenet", 10, norm="lrn"),
    Experiment("alexnet_bn_no_aug_tiny", "alexnet", "tiny_imagenet", 10, norm="bn"),
    Experiment("zoo_alexnet_tiny", "zoo_alexnet", "tiny_imagenet", 10),
)


def run(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def output_paths(experiment: Experiment, seed: int):
    suffix = "" if seed == 42 else f"_seed{seed}"
    stem = f"{experiment.key}{suffix}"
    plot_key = "lenet" if experiment.key == "lenet_baseline" else experiment.key
    plot_stem = f"{plot_key}{suffix}"
    return (
        ROOT / "checkpoints" / f"{stem}.pth",
        ROOT / "plots" / f"loss_{plot_stem}.png",
        ROOT / "results" / f"history_{stem}.json",
        ROOT / "results" / f"evaluation_{stem}.json",
    )


def run_experiment(experiment: Experiment, seed: int, device: str, force: bool, dry_run: bool) -> None:
    checkpoint, plot, history, evaluation = output_paths(experiment, seed)
    if not force and all(path.exists() for path in (checkpoint, plot, history, evaluation)):
        print(f"Skipping complete run: {experiment.key} (seed {seed})")
        return

    train_command = [
        sys.executable,
        "train.py",
        "--model", experiment.model,
        "--dataset", experiment.dataset,
        "--epochs", str(experiment.epochs),
        "--seed", str(seed),
        "--device", device,
        "--save-model", str(checkpoint),
        "--plot-file", str(plot),
        "--history-file", str(history),
    ]
    if experiment.model == "alexnet":
        train_command.extend(["--norm", experiment.norm])
    if not experiment.augment:
        train_command.append("--no-augment")
    run(train_command, dry_run)

    evaluate_command = [
        sys.executable,
        "evaluate.py",
        "--checkpoint", str(checkpoint),
        "--model", experiment.model,
        "--dataset", experiment.dataset,
        "--seed", str(seed),
        "--device", device,
        "--output-file", str(evaluation),
    ]
    if experiment.model == "alexnet":
        evaluate_command.extend(["--norm", experiment.norm])
    run(evaluate_command, dry_run)


def select_experiments(phase: str) -> Iterable[Experiment]:
    if phase == "inlab":
        return INLAB
    if phase == "cifar":
        return CIFAR_POSTLAB
    if phase == "tiny":
        return TINY_CORE
    if phase == "core":
        return INLAB + CIFAR_POSTLAB + TINY_CORE
    if phase == "full":
        return INLAB + CIFAR_POSTLAB + TINY_CORE + TINY_FULL
    raise ValueError(phase)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible ELEC 475 Lab 2 experiments")
    parser.add_argument("--phase", choices=["inlab", "cifar", "tiny", "core", "full"], default="core")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force", action="store_true", help="Rerun experiments even when all outputs exist")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    for directory in ("checkpoints", "plots", "results"):
        (ROOT / directory).mkdir(exist_ok=True)

    for seed in args.seeds:
        for experiment in select_experiments(args.phase):
            run_experiment(experiment, seed, args.device, args.force, args.dry_run)

    if not args.skip_profile:
        run([
            sys.executable,
            "compare_models.py",
            "--dataset", "cifar10",
            "--device", args.device,
            "--output-file", str(ROOT / "results" / "complexity_cifar10.md"),
        ], args.dry_run)
        run([
            sys.executable,
            "-m", "utils.visualization",
            "--plot-ablation",
            "--results-dir", str(ROOT / "results"),
            "--save-path", str(ROOT / "plots" / "ablation_summary.png"),
        ], args.dry_run)
        run([sys.executable, "summarize_results.py"], args.dry_run)


if __name__ == "__main__":
    main()
