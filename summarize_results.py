"""Build report-ready Lab 2 tables from experiment JSON files."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

ROWS = (
    ("1. Base LeNet-5", "lenet_baseline", "lenet_tiny", "2", "None", "None", "1. LeNet-5 (Base)"),
    ("2. Modernized LeNet", "lenet_modern", "lenet_modern_tiny", "2", "ReLU + overlapping MaxPool", "None", "2. LeNet-Modern (ReLU+Pool)"),
    ("3. AlexNet (No Norm)", "alexnet_none", "alexnet_none_tiny", "5", "Dropout (p=0.5)", "None", "3. AlexNet-64 (None)"),
    ("4. AlexNet + LRN", "alexnet_lrn", "alexnet_lrn_tiny", "5", "LRN + Dropout (p=0.5)", "None", "4. AlexNet-64 (LRN)"),
    ("5. AlexNet + BatchNorm", "alexnet_bn", "alexnet_bn_no_aug_tiny", "5", "BatchNorm2d + Dropout (p=0.5)", "None", "5. AlexNet-64 (BatchNorm)"),
    ("6. AlexNet + BN + Aug", "alexnet_with_aug", "alexnet_tiny_imagenet", "5", "BatchNorm2d + Dropout (p=0.5)", "Crop + flip + jitter", "5. AlexNet-64 (BatchNorm)"),
    ("7. PyTorch Zoo AlexNet", "zoo_alexnet", "zoo_alexnet_tiny", "5", "Zoo model + Dropout", "None", "6. Torchvision Zoo AlexNet"),
)


def load_json(prefix: str, key: str):
    path = RESULTS / f"{prefix}_{key}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value, digits=2, suffix=""):
    return "Pending" if value is None else f"{value:.{digits}f}{suffix}"


def parse_latencies():
    path = RESULTS / "complexity_cifar10.md"
    if not path.exists():
        return {}
    latencies = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "Model Architecture" in line or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 5:
            try:
                latencies[cells[0]] = float(cells[4])
            except ValueError:
                continue
    return latencies


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def multi_seed_rows():
    groups = {}
    for path in RESULTS.glob("evaluation_*.json"):
        key = path.stem.removeprefix("evaluation_")
        if "_seed" in key and key.rsplit("_seed", 1)[1].isdigit():
            key = key.rsplit("_seed", 1)[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("dataset") != "cifar10":
            continue
        groups.setdefault(key, {"accuracy": [], "f1": []})
        groups[key]["accuracy"].append(data["top1_accuracy"])
        groups[key]["f1"].append(data["macro_f1"])

    rows = []
    for key, values in sorted(groups.items()):
        if len(values["accuracy"]) < 2:
            continue
        accuracy_std = stdev(values["accuracy"])
        f1_std = stdev(values["f1"])
        rows.append((
            key,
            len(values["accuracy"]),
            f"{mean(values['accuracy']):.2f} +/- {accuracy_std:.2f}%",
            f"{mean(values['f1']):.2f} +/- {f1_std:.2f}%",
        ))
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    latencies = parse_latencies()
    complexity_rows = []
    predictive_rows = []

    for label, cifar_key, tiny_key, depth, regularization, augmentation, profile_label in ROWS:
        history = load_json("history", cifar_key)
        cifar = load_json("evaluation", cifar_key)
        tiny = load_json("evaluation", tiny_key)
        complexity = history.get("complexity", {}) if history else {}
        summary = history.get("summary", {}) if history else {}

        complexity_rows.append((
            label,
            depth,
            regularization,
            augmentation,
            fmt(complexity.get("trainable_params", None) / 1e6 if complexity else None, 3),
            fmt(complexity.get("flops_m")),
            fmt(summary.get("mean_epoch_time_seconds")),
            fmt(latencies.get(profile_label)),
        ))
        predictive_rows.append((
            label,
            fmt(cifar.get("top1_accuracy") if cifar else None, suffix="%"),
            fmt(cifar.get("macro_f1") if cifar else None, suffix="%"),
            fmt(tiny.get("top1_accuracy") if tiny else None, suffix="%"),
            fmt(tiny.get("macro_f1") if tiny else None, suffix="%"),
        ))

    no_aug = load_json("history", "alexnet_no_aug")
    with_aug = load_json("history", "alexnet_with_aug")
    augmentation_rows = []
    for label, data in (("AlexNet BN, no augmentation", no_aug), ("AlexNet BN, with augmentation", with_aug)):
        summary = data.get("summary", {}) if data else {}
        augmentation_rows.append((
            label,
            fmt(summary.get("final_train_accuracy"), suffix="%"),
            fmt(summary.get("final_validation_accuracy"), suffix="%"),
            fmt(summary.get("generalization_gap"), suffix=" percentage points"),
        ))

    sections = [
        "# ELEC 475 Lab 2 - Generated Results Tables",
        "",
        "Values marked `Pending` require the corresponding experiment to be run.",
        "",
        "## Table 7.3a - Computational Complexity",
        "",
        markdown_table(
            ["Model", "Conv depth", "Normalization / regularization", "Augmentation", "Params (M)", "FLOPs (M)", "Mean epoch (s)", "Latency (ms)"],
            complexity_rows,
        ),
        "",
        "## Table 7.3b - Predictive Performance",
        "",
        markdown_table(
            ["Model", "CIFAR-10 accuracy", "CIFAR-10 macro F1", "Tiny ImageNet accuracy", "Tiny ImageNet macro F1"],
            predictive_rows,
        ),
        "",
        "## Augmentation and Generalization Gap",
        "",
        markdown_table(["Configuration", "Final train accuracy", "Final validation accuracy", "Generalization gap"], augmentation_rows),
        "",
        "## Multi-seed Results",
        "",
    ]
    seed_rows = multi_seed_rows()
    sections.append(markdown_table(["Experiment", "Seeds", "Validation accuracy", "Macro F1"], seed_rows) if seed_rows else "Pending: run at least two seeds for an experiment.")
    sections.append("")

    output = RESULTS / "master_tables.md"
    output.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report-ready tables saved to: {output}")


if __name__ == "__main__":
    main()
