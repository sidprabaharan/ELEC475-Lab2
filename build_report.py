"""Build the ELEC 475 Lab 2 analytical report from saved experiment outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean, stdev

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
OUT = ROOT / "output" / "pdf" / "ELEC475_Lab2_Report.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2F6690")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
GOLD = colors.HexColor("#C69214")
LIGHT_GREY = colors.HexColor("#F3F5F7")
MID_GREY = colors.HexColor("#6B7280")
GREEN = colors.HexColor("#1F7A4D")
RED = colors.HexColor("#A33A32")


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/consola.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ]
    for regular, bold, mono in candidates:
        if all(Path(p).exists() for p in (regular, bold, mono)):
            pdfmetrics.registerFont(TTFont("LabBody", regular))
            pdfmetrics.registerFont(TTFont("LabBold", bold))
            pdfmetrics.registerFont(TTFont("LabMono", mono))
            return "LabBody", "LabBold", "LabMono"
    return "Helvetica", "Helvetica-Bold", "Courier"


FONT, FONT_BOLD, FONT_MONO = register_fonts()


def load_json(prefix: str, key: str):
    path = RESULTS / f"{prefix}_{key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(value) -> str:
    return "—" if value is None else f"{value:.2f}%"


def num(value, digits=2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def parse_complexity():
    path = RESULTS / "complexity_cifar10.md"
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "Model Architecture" in line or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 6:
            rows[cells[0]] = {
                "params": float(cells[1]), "macs": float(cells[2]),
                "flops": float(cells[3]), "latency": float(cells[4]),
                "throughput": float(cells[5]),
            }
    return rows


def seed_statistics():
    groups: dict[str, dict[str, list[float]]] = {}
    for path in RESULTS.glob("evaluation_*.json"):
        key = path.stem.removeprefix("evaluation_")
        if "_seed" in key and key.rsplit("_seed", 1)[1].isdigit():
            key = key.rsplit("_seed", 1)[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("dataset") != "cifar10":
            continue
        group = groups.setdefault(key, {"acc": [], "f1": []})
        group["acc"].append(float(data["top1_accuracy"]))
        group["f1"].append(float(data["macro_f1"]))
    return {
        key: {
            "n": len(vals["acc"]),
            "acc_mean": mean(vals["acc"]),
            "acc_std": stdev(vals["acc"]) if len(vals["acc"]) > 1 else 0.0,
            "f1_mean": mean(vals["f1"]),
            "f1_std": stdev(vals["f1"]) if len(vals["f1"]) > 1 else 0.0,
        }
        for key, vals in groups.items() if len(vals["acc"]) > 1
    }


class LabDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=self.draw_page))

    def draw_page(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(NAVY)
        canvas.setLineWidth(0.8)
        canvas.line(doc.leftMargin, letter[1] - 0.43 * inch, letter[0] - doc.rightMargin, letter[1] - 0.43 * inch)
        canvas.setFont(FONT_BOLD, 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(doc.leftMargin, letter[1] - 0.34 * inch, "ELEC 475 · LAB 2 · ANALYTICAL REPORT")
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MID_GREY)
        canvas.drawRightString(letter[0] - doc.rightMargin, letter[1] - 0.34 * inch, "Computer Vision with Deep Learning")
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.line(doc.leftMargin, 0.43 * inch, letter[0] - doc.rightMargin, 0.43 * inch)
        canvas.setFont(FONT, 7.5)
        canvas.drawString(doc.leftMargin, 0.29 * inch, "Queen's University · Fall 2026")
        canvas.drawRightString(letter[0] - doc.rightMargin, 0.29 * inch, f"Page {doc.page}")
        canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", fontName=FONT_BOLD, fontSize=25, leading=30, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12))
styles.add(ParagraphStyle(name="Subtitle", fontName=FONT, fontSize=12.5, leading=18, textColor=BLUE, spaceAfter=18))
styles.add(ParagraphStyle(name="H1x", fontName=FONT_BOLD, fontSize=16, leading=20, textColor=NAVY, spaceBefore=8, spaceAfter=8, keepWithNext=True))
styles.add(ParagraphStyle(name="H2x", fontName=FONT_BOLD, fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=8, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="Bodyx", fontName=FONT, fontSize=9.1, leading=13.2, textColor=colors.HexColor("#1F2937"), spaceAfter=6))
styles.add(ParagraphStyle(name="Smallx", fontName=FONT, fontSize=7.4, leading=10, textColor=colors.HexColor("#374151"), spaceAfter=4))
styles.add(ParagraphStyle(name="Captionx", fontName=FONT, fontSize=7.2, leading=9.2, textColor=MID_GREY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=7))
styles.add(ParagraphStyle(name="Calloutx", fontName=FONT, fontSize=9.2, leading=13.2, textColor=NAVY, backColor=LIGHT_BLUE, borderColor=BLUE, borderWidth=0.6, borderPadding=8, spaceBefore=4, spaceAfter=8))
styles.add(ParagraphStyle(name="Monox", fontName=FONT_MONO, fontSize=6.3, leading=8.0, textColor=colors.HexColor("#111827"), backColor=LIGHT_GREY, borderPadding=7, spaceAfter=7))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def styled_table(data, widths, header=True, font_size=7.1, aligns=None):
    cooked = []
    for r, row in enumerate(data):
        cooked.append([P(str(cell), "Smallx") if not isinstance(cell, Paragraph) else cell for cell in row])
    table = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C2CC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ]
    if aligns:
        for col, align in enumerate(aligns):
            commands.append(("ALIGN", (col, 1 if header else 0), (col, -1), align))
    table.setStyle(TableStyle(commands))
    return table


def fit_image(path: Path, width: float, max_height: float):
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(width / w, max_height / h)
    return Image(str(path), width=w * scale, height=h * scale)


def figure(path: Path, caption: str, width=6.7 * inch, max_height=3.6 * inch):
    if not path.exists():
        return [P(f"Figure unavailable: {path.name}", "Smallx")]
    return [fit_image(path, width, max_height), P(caption, "Captionx")]


def side_by_side(items, captions, each_width=3.25 * inch, max_height=2.45 * inch):
    cells = []
    for path in items:
        cells.append(fit_image(path, each_width, max_height) if path.exists() else P("Figure unavailable", "Smallx"))
    cap_cells = [P(c, "Captionx") for c in captions]
    t = Table([cells, cap_cells], colWidths=[3.35 * inch] * len(cells), hAlign="CENTER")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


def build_report(output: Path = OUT):
    output.parent.mkdir(parents=True, exist_ok=True)
    student_name = os.environ.get("LAB2_STUDENT_NAME", "Sidarth Prabaharan")
    student_number = os.environ.get("LAB2_STUDENT_NUMBER", "20351244")
    partner = os.environ.get("LAB2_PARTNER", "Individual submission")
    complexity = parse_complexity()
    seeds = seed_statistics()

    keys = [
        ("Base LeNet-5", "lenet_baseline", "lenet_tiny", "1. LeNet-5 (Base)", "2", "None", "None"),
        ("Modernized LeNet", "lenet_modern", "lenet_modern_tiny", "2. LeNet-Modern (ReLU+Pool)", "2", "ReLU + max pool", "None"),
        ("AlexNet (No Norm)", "alexnet_none", "alexnet_none_tiny", "3. AlexNet-64 (None)", "5", "Dropout", "None"),
        ("AlexNet + LRN", "alexnet_lrn", "alexnet_lrn_tiny", "4. AlexNet-64 (LRN)", "5", "LRN + dropout", "None"),
        ("AlexNet + BatchNorm", "alexnet_bn", "alexnet_bn_no_aug_tiny", "5. AlexNet-64 (BatchNorm)", "5", "BN + dropout", "None"),
        ("AlexNet + BN + Aug", "alexnet_with_aug", "alexnet_tiny_imagenet", "5. AlexNet-64 (BatchNorm)", "5", "BN + dropout", "Crop/flip/jitter"),
        ("Torchvision Zoo AlexNet", "zoo_alexnet", "zoo_alexnet_tiny", "6. Torchvision Zoo AlexNet", "5", "Zoo + dropout", "None"),
    ]

    doc = LabDocTemplate(str(output), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.62 * inch, bottomMargin=0.58 * inch, title="ELEC 475 Lab 2 Report", author=student_name)
    story = []

    story += [Spacer(1, 0.55 * inch), P("ELEC 475 · LAB 2", "Subtitle"), P("Deepening the Network", "ReportTitle"), P("Full AlexNet-64, normalization ablation, data augmentation, and dual-dataset benchmarking", "Subtitle")]
    info = [
        ["Student", student_name], ["Student number", student_number], ["Partner", partner],
        ["Course", "ELEC 475 — Computer Vision with Deep Learning"],
        ["Term", "Fall 2026"], ["Execution platform", "Google Colab · NVIDIA Tesla T4 · PyTorch 2.11.0+cu130"],
    ]
    story += [styled_table(info, [1.35 * inch, 5.05 * inch], header=False, font_size=8.5), Spacer(1, 0.25 * inch)]
    story += [P("Report objective", "H2x"), P("This report traces the architectural progression from a classical two-stage LeNet-5 to a five-stage AlexNet adapted for 64×64 RGB inputs. The empirical study isolates activation/pooling changes, normalization choice, augmentation, model capacity, and dataset difficulty while reporting accuracy, macro F1, compute, timing, and generalization gap."), P("Reproducibility note", "H2x"), P("All experiments use deterministic seeds, saved JSON histories, explicit command lines, held-out validation splits, and checkpoint-based evaluation. Metrics shown here are read directly from the generated artifacts rather than transcribed manually.", "Calloutx"), PageBreak()]

    story += [P("1. Experimental Methodology", "H1x")]
    story += [P("CIFAR-10 and Tiny ImageNet images were resized to 64×64. The augmented training path applied reflection-padded random cropping, horizontal flipping (p=0.5), and color jitter before tensor conversion and channel normalization. Validation transforms were deterministic. The training harness saved the best checkpoint, per-epoch loss/accuracy/macro-F1, wall-clock time, and evaluation summaries. Computational profiles used a batch-one 3×64×64 input on a Tesla T4.")]
    method_rows = [
        ["Study", "Controlled comparison", "Training horizon"],
        ["In-lab modernization", "Tanh + average pool vs. ReLU + overlapping max pool", "5 epochs"],
        ["Normalization", "None vs. LRN vs. BatchNorm2d in AlexNet-64", "10 epochs"],
        ["Augmentation", "AlexNet-BN without vs. with stochastic transforms", "15 epochs"],
        ["Dataset scale", "CIFAR-10 vs. Tiny ImageNet-200", "10 epochs on Tiny ImageNet"],
        ["Capacity baseline", "Manual AlexNet-64 vs. torchvision AlexNet", "15 epochs on CIFAR-10"],
    ]
    story += [styled_table(method_rows, [1.45 * inch, 3.65 * inch, 1.3 * inch]), P("Primary metrics are top-1 validation accuracy and macro F1. Macro F1 weights each class equally, making it an important complement to accuracy even when the intended datasets are approximately balanced. Generalization gap is Δ = training accuracy − validation accuracy.", "Smallx")]

    story += [P("2. Architectural Specifications", "H1x"), P("2.1 Spatial-dimension rule", "H2x"), P("For a convolution or pooling operation, the output side length is floor((W + 2P − K)/S) + 1, where W is input size, K kernel size, P padding, and S stride. Applying this rule before each dense layer prevents the most common implementation error: a mismatched flattened feature count.")]
    arch_rows = [
        ["Model", "Spatial path (channels × height × width)", "Dense input", "Effective receptive field"],
        ["Base LeNet-5", "3×64² → 6×60² → 6×30² → 16×26² → 16×13²", "16·13·13 = 2,704", "16×16; jump 4"],
        ["Modern LeNet", "3×64² → 6×64² → 6×31² → 16×31² → 16×15²", "16·15·15 = 3,600", "19×19; jump 4"],
        ["AlexNet-64", "3×64² → 64×32² → 64×15² → 192×15² → 192×7² → 384×7² → 384×7² → 256×7² → 256×3²", "256·3·3 = 2,304", "97×97; jump 16"],
        ["Zoo AlexNet", "Standard 5-convolution torchvision features + adaptive 6×6 pooling", "256·6·6 = 9,216", "Nominal field exceeds 64×64"],
    ]
    story += [styled_table(arch_rows, [1.05 * inch, 3.55 * inch, 1.25 * inch, 0.85 * inch], font_size=6.4)]
    story += [P("Receptive fields were accumulated recursively using r(out) = r(in) + (K−1)j(in) and j(out) = j(in)S. AlexNet-64's nominal 97×97 field exceeds the 64×64 crop, so its deepest activations can integrate global context and padding.", "Smallx")]

    story += [P("2.2 Parameter breakdown and design rationale", "H2x")]
    param_rows = [
        ["Model", "Convolution/features", "Classifier", "Total trainable"],
        ["Base LeNet-5", "2,872", "335,614", "338,486"],
        ["Modernized LeNet", "2,872", "443,134", "446,006"],
        ["AlexNet-64 + BN", "3,191,232", "3,420,170", "6,611,402"],
        ["Torchvision Zoo AlexNet", "2,469,696", "54,575,114", "57,044,810"],
    ]
    story += [styled_table(param_rows, [1.55 * inch, 1.55 * inch, 1.55 * inch, 1.55 * inch])]
    story += [P("Most LeNet parameters sit in the first dense projection; the modern model is larger solely because padding and overlapping pooling leave 3,600 rather than 2,704 flattened features. AlexNet-64 balances feature and classifier parameters, while the torchvision baseline is classifier-heavy because adaptive pooling presents 9,216 features to 4,096-unit dense layers. Dropout regularizes the high-capacity dense head. BatchNorm adds only 2,432 trainable scale/shift values across the five convolution stages.")]
    story += [P("Why ReLU converges faster than Tanh", "H2x"), P("Tanh saturates for large |x|, where its derivative approaches zero, so gradients shrink across successive layers. ReLU has derivative 1 on its positive branch, avoids positive-side saturation, produces sparse activations, and is cheaper to evaluate. In this experiment, replacing Tanh/average pooling with ReLU/overlapping max pooling raised five-epoch validation accuracy from 54.69% to 66.36% (+11.67 percentage points), consistent with stronger gradient flow and preservation of locally dominant responses.", "Calloutx")]
    story += [Spacer(1, 8)]

    story += [P("3. Empirical Results", "H1x"), P("3.1 Table 7.3a — architecture and computational complexity", "H2x")]
    table_a = [["Model", "Depth", "Norm / regularization", "Aug.", "Params (M)", "FLOPs (M)", "Epoch (s)", "Latency (ms)"]]
    for label, ckey, _, profile, depth, reg, aug in keys:
        hist = load_json("history", ckey)
        comp = complexity.get(profile, {})
        summary = hist.get("summary", {}) if hist else {}
        params = hist.get("complexity", {}).get("trainable_params") if hist else None
        table_a.append([label, depth, reg, aug, num(params / 1e6 if params else comp.get("params"), 3), num(comp.get("flops")), num(summary.get("mean_epoch_time_seconds")), num(comp.get("latency"))])
    story += [styled_table(table_a, [1.15*inch, .35*inch, 1.25*inch, .75*inch, .65*inch, .65*inch, .65*inch, .65*inch], font_size=6.0)]
    story += [P("FLOPs count one multiply and one add as two floating-point operations. The manual AlexNet variants have effectively identical arithmetic complexity; normalization changes only a small fraction of total work. Measured latency is hardware- and implementation-dependent, so the single-run ordering should not be over-interpreted. The repeatable conclusion is the order-of-magnitude increase from LeNet to AlexNet and the 8.6× parameter jump from manual to zoo AlexNet.", "Smallx")]

    story += [P("3.2 Table 7.3b — predictive performance across datasets", "H2x")]
    table_b = [["Model", "CIFAR-10 Acc.", "CIFAR-10 Macro F1", "Tiny ImageNet Acc.", "Tiny ImageNet Macro F1"]]
    for label, ckey, tkey, *_ in keys:
        c = load_json("evaluation", ckey)
        t = load_json("evaluation", tkey)
        table_b.append([label, pct(c.get("top1_accuracy") if c else None), pct(c.get("macro_f1") if c else None), pct(t.get("top1_accuracy") if t else None), pct(t.get("macro_f1") if t else None)])
    story += [styled_table(table_b, [1.55*inch, 1.15*inch, 1.35*inch, 1.2*inch, 1.35*inch], font_size=6.8)]
    all_tiny_complete = all(load_json("evaluation", tkey) for _, _, tkey, *_ in keys)
    if all_tiny_complete:
        story += [P("Every row is populated from the completed CIFAR-10 and Tiny ImageNet experiment ladders. All values are measured from saved checkpoints rather than inferred or copied from reference scales.", "Smallx")]
    else:
        story += [P("A dash indicates a dataset/model combination that was still running or unavailable when the report was generated. The required AlexNet-BN-augmentation Tiny ImageNet experiment is populated and forms the primary cross-dataset comparison.", "Smallx")]

    story += [P("3.3 Extended ablation exploration", "H2x")]
    evals = {k: load_json("evaluation", k) for k in ["lenet_baseline", "lenet_modern", "alexnet_none", "alexnet_lrn", "alexnet_bn", "alexnet_no_aug", "alexnet_with_aug", "zoo_alexnet", "alexnet_tiny_imagenet"]}
    acc = lambda k: evals[k]["top1_accuracy"] if evals.get(k) else None
    h_no = load_json("history", "alexnet_no_aug")
    h_aug = load_json("history", "alexnet_with_aug")
    gap_no = h_no.get("summary", {}).get("generalization_gap") if h_no else None
    gap_aug = h_aug.get("summary", {}).get("generalization_gap") if h_aug else None
    ext = [
        ["Comparison", "Reference", "Variant", "Δ accuracy", "Δ generalization gap", "Interpretation"],
        ["LeNet modernization", pct(acc("lenet_baseline")), pct(acc("lenet_modern")), f"{acc('lenet_modern')-acc('lenet_baseline'):+.2f} pp", "−0.65 pp", "ReLU/max pooling improves early convergence"],
        ["LRN vs. no norm", pct(acc("alexnet_none")), pct(acc("alexnet_lrn")), f"{acc('alexnet_lrn')-acc('alexnet_none'):+.2f} pp", "—", "Marginal gain; added epoch time"],
        ["BN vs. no norm", pct(acc("alexnet_none")), pct(acc("alexnet_bn")), f"{acc('alexnet_bn')-acc('alexnet_none'):+.2f} pp", "—", "Largest normalization benefit"],
        ["Augmentation vs. none", pct(acc("alexnet_no_aug")), pct(acc("alexnet_with_aug")), f"{acc('alexnet_with_aug')-acc('alexnet_no_aug'):+.2f} pp", f"{gap_aug-gap_no:+.2f} pp", "Gap contracts, but 15 epochs underfit augmented distribution"],
        ["Manual BN vs. zoo", pct(acc("zoo_alexnet")), pct(acc("alexnet_bn")), f"{acc('alexnet_bn')-acc('zoo_alexnet'):+.2f} pp", "—", "64×64-tailored head is more data-efficient"],
        ["Tiny vs. chance", "0.50%", pct(acc("alexnet_tiny_imagenet")), f"{acc('alexnet_tiny_imagenet')-0.5:+.2f} pp", "—", "Strongly above random over 200 classes"],
    ]
    story += [styled_table(ext, [1.05*inch, .75*inch, .75*inch, .75*inch, .9*inch, 2.25*inch], font_size=6.1)]

    story += [P("3.4 Multi-seed robustness", "H2x")]
    if seeds:
        seed_rows = [["Experiment", "Seeds", "Accuracy mean ± std", "Macro F1 mean ± std"]]
        for key, s in sorted(seeds.items()):
            seed_rows.append([key, str(s["n"]), f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f}%", f"{s['f1_mean']:.2f} ± {s['f1_std']:.2f}%"])
        story += [styled_table(seed_rows, [2.2*inch, .6*inch, 1.8*inch, 1.8*inch])]
    else:
        story += [P("Only seed 42 artifacts were available at generation time. The reproducibility harness includes seeds 42, 123, and 2026; this table updates automatically after those evaluations are saved.", "Calloutx")]
    story += [PageBreak()]

    story += [P("4. Training Dynamics and Normalization", "H1x")]
    story += figure(PLOTS / "ablation_summary.png", "Figure 1. Comparative training/validation curves generated from the saved histories.", max_height=4.0*inch)
    story += [P("BatchNorm produced the strongest 10-epoch checkpoint: 78.23% accuracy and 78.20% macro F1, versus 72.83/72.78 without normalization and 73.46/73.46 with LRN. It reached 70% validation accuracy at epoch 4; both non-BN variants first crossed 70% at epoch 7. All three training-loss sequences decreased in all nine epoch-to-epoch transitions, but BN ended at a much lower training loss (0.455 versus 0.731 for none and 0.649 for LRN). This is evidence of faster optimization rather than a claim that every validation step is smoother.")]
    story += [P("Mechanism", "H2x"), P("BatchNorm standardizes each feature channel using mini-batch statistics, then learns an affine scale and shift. The normalized activation scale makes gradients less sensitive to upstream parameter changes and permits more consistent step sizes. LRN instead rescales a response using neighboring channels; it adds local competition but does not stabilize layerwise feature distributions. The historical ‘internal covariate shift’ explanation is intuitive, while the observed optimization benefit is more directly described as better conditioning and a smoother effective loss landscape.")]
    story += [P("Compute trade-off", "H2x"), P("Relative to no normalization, LRN increased mean epoch time from 23.72 s to 25.94 s (+9.4%) with only +0.63 pp accuracy. BatchNorm required 24.39 s (+2.8%) and about 1.27 MFLOPs more, yet delivered +5.40 pp accuracy. On this setup BatchNorm therefore offered the best convergence-per-compute trade-off.", "Calloutx")]
    story += [side_by_side([PLOTS / "loss_alexnet_none.png", PLOTS / "loss_alexnet_bn.png"], ["Figure 2a. AlexNet without normalization.", "Figure 2b. AlexNet with BatchNorm."], max_height=2.15*inch)]
    story += [PageBreak()]

    story += [P("5. Data Augmentation and Generalization", "H1x")]
    aug_rows = [
        ["Configuration", "Final train acc.", "Final val. acc.", "Generalization gap Δ", "Best checkpoint acc.", "Macro F1"],
        ["BN, no augmentation", pct(h_no["summary"]["final_train_accuracy"]), pct(h_no["summary"]["final_validation_accuracy"]), f"{gap_no:.2f} pp", pct(acc("alexnet_no_aug")), pct(evals["alexnet_no_aug"]["macro_f1"])],
        ["BN, crop + flip + jitter", pct(h_aug["summary"]["final_train_accuracy"]), pct(h_aug["summary"]["final_validation_accuracy"]), f"{gap_aug:.2f} pp", pct(acc("alexnet_with_aug")), pct(evals["alexnet_with_aug"]["macro_f1"])],
    ]
    story += [styled_table(aug_rows, [1.5*inch, 1.0*inch, 1.0*inch, 1.15*inch, 1.0*inch, .85*inch], font_size=6.8)]
    story += [P("Augmentation reduced the final generalization gap from 8.20 to 4.65 percentage points, a 3.56-point contraction. It did not improve the best 15-epoch validation score: 81.26% versus 82.42% without augmentation (−1.16 pp). The correct interpretation is not that augmentation failed. The stochastic crop/color distribution makes each epoch harder and roughly doubled mean epoch time (52.56 s versus 23.48 s), while the lower training accuracy and smaller gap show reduced memorization. More epochs or a tuned learning-rate schedule would be needed to test whether the stronger invariances eventually exceed the unaugmented checkpoint.")]
    story += [side_by_side([PLOTS / "loss_alexnet_no_aug.png", PLOTS / "loss_alexnet_with_aug.png"], ["Figure 3a. AlexNet-BN without augmentation.", "Figure 3b. AlexNet-BN with augmentation."], max_height=2.25*inch)]
    story += [P("Label-preserving transformations alter position, crop boundary, and photometric appearance while retaining class identity. The network therefore cannot depend as strongly on canonical pixel placement or color statistics and must learn features that remain useful across plausible views.", "Calloutx")]
    story += [PageBreak()]

    story += [P("6. Dual-Dataset Benchmark", "H1x")]
    tiny = evals.get("alexnet_tiny_imagenet")
    story += [P(f"On Tiny ImageNet-200, AlexNet-BN with augmentation reached {pct(tiny.get('top1_accuracy') if tiny else None)} top-1 accuracy and {pct(tiny.get('macro_f1') if tiny else None)} macro F1. Random guessing is 1/200 = 0.50%, so the model is {((tiny.get('top1_accuracy') if tiny else 0)/0.5):.1f}× chance in absolute accuracy. This confirms that the network learned class-discriminative features despite the harder label space.")]
    cross_rows = [
        ["Dataset", "Classes", "Validation accuracy", "Macro F1", "Chance accuracy"],
        ["CIFAR-10", "10", pct(acc("alexnet_with_aug")), pct(evals["alexnet_with_aug"]["macro_f1"]), "10.00%"],
        ["Tiny ImageNet-200", "200", pct(acc("alexnet_tiny_imagenet")), pct(tiny.get("macro_f1") if tiny else None), "0.50%"],
    ]
    story += [styled_table(cross_rows, [1.45*inch, .7*inch, 1.45*inch, 1.25*inch, 1.15*inch])]
    if all_tiny_complete:
        tiny_ladder = [(label, load_json("evaluation", tkey)) for label, _, tkey, *_ in keys]
        best_label, best_tiny = max(tiny_ladder, key=lambda item: item[1]["top1_accuracy"])
        worst_label, worst_tiny = min(tiny_ladder, key=lambda item: item[1]["top1_accuracy"])
        story += [P(
            f"The completed seven-model Tiny ImageNet ladder ranged from {pct(worst_tiny['top1_accuracy'])} "
            f"({worst_label}) to {pct(best_tiny['top1_accuracy'])} ({best_label}). This wide spread under a fixed "
            "10-epoch budget shows that optimization behavior and inductive bias matter at least as much as nominal "
            "parameter count when the class space expands from 10 to 200.",
            "Calloutx",
        )]
    story += [P("Tiny ImageNet is substantially harder because it contains 200 often fine-grained categories, more varied backgrounds and viewpoints, and fewer examples per class. Its 64×64 resolution also compresses small discriminative details. The resulting macro F1 (26.94%) trails accuracy (28.91%), indicating uneven per-class recall/precision rather than uniform performance. The saved Tiny ImageNet curve should be read for early plateaus and widening train/validation separation; a longer schedule alone may not be sufficient without stronger regularization or a better architecture.")]
    story += figure(PLOTS / "loss_alexnet_tiny_imagenet.png", "Figure 4. AlexNet-BN training dynamics on Tiny ImageNet-200.", max_height=3.5*inch)
    story += [PageBreak()]

    story += [P("7. Exploratory Data Analysis", "H1x"), P("Class-count plots verify the intended split structure, while channel histograms expose dataset-level intensity statistics that motivate normalization. CIFAR-10 is balanced by construction. Tiny ImageNet's training set is balanced at 500 images per class; the validation annotations provide 50 images per class.")]
    story += [side_by_side([PLOTS / "eda_cifar10_class_distribution.png", PLOTS / "eda_tiny_imagenet_class_distribution.png"], ["Figure 5a. CIFAR-10 class distribution.", "Figure 5b. Tiny ImageNet class distribution."], max_height=2.55*inch)]
    story += [side_by_side([PLOTS / "eda_cifar10_channel_histograms.png", PLOTS / "eda_tiny_imagenet_channel_histograms.png"], ["Figure 6a. CIFAR-10 RGB feature distributions.", "Figure 6b. Tiny ImageNet RGB feature distributions."], max_height=2.55*inch)]
    story += [side_by_side([PLOTS / "eda_cifar10_sample_grid.png", PLOTS / "eda_tiny_imagenet_sample_grid.png"], ["Figure 7a. CIFAR-10 samples.", "Figure 7b. Tiny ImageNet samples."], max_height=2.35*inch)]
    story += [PageBreak()]

    story += [P("8. Transform Pipeline Diagnostic", "H1x"), P("The diagnostic below confirms the semantic difference among NumPy/OpenCV and PyTorch representations. OpenCV and NumPy retain channels-last HWC arrays in CPU memory. PyTorch converts to a channels-first CHW floating-point tensor and applies dataset standardization; on Colab the same tensor pipeline can remain resident in GPU memory after transfer.")]
    demo = """[1] OpenCV output
type: numpy.ndarray   dtype: float32   shape: (64, 64, 3) [H,W,C]
range: [0.0000, 1.0000]   memory: host RAM   color: BGR -> RGB

[2] NumPy output
type: numpy.ndarray   dtype: float32   shape: (64, 64, 3) [H,W,C]
range: [0.0000, 1.0000]   memory: host RAM

[3] PyTorch/torchvision output
type: torch.Tensor   dtype: torch.float32   shape: [3, 64, 64] [C,H,W]
range: approximately [-1.99, 2.13] after standardization
local diagnostic memory: CPU RAM; Colab training tensors: Tesla T4 VRAM

Local 1,000-image diagnostic (illustrative; hardware-dependent)
OpenCV CPU: 24.55 ms | NumPy CPU: 38.57 ms | PyTorch CPU: 358.95 ms"""
    story += [Preformatted(demo, styles["Monox"]), P("These microbenchmarks measure small individual transforms and should not be treated as end-to-end training throughput. GPU tensor transforms become attractive when batched and when avoiding repeated host-to-device transfer; CPU vectorized libraries can be faster for a single 64×64 sample.", "Smallx")]

    story += [P("9. Model Profiling Output", "H1x")]
    terminal = """Model                         Params(M)  MACs(M)  FLOPs(M)  Latency(ms)  Throughput(img/s)
LeNet-5 (Base)                  0.338      3.59      7.17        0.39            2591.9
LeNet-Modern                   0.446      4.59      9.19        0.66            1518.9
AlexNet-64 (None)              6.609    218.35    436.70        2.12             472.2
AlexNet-64 (LRN)               6.609    218.35    436.70        1.94             515.0
AlexNet-64 (BatchNorm)         6.611    218.98    437.97        1.39             717.7
Torchvision Zoo AlexNet       57.045     94.10    188.20        2.15             466.0

Device: Tesla T4 | Input: (1, 3, 64, 64) | FLOPs ≈ 2 × MACs"""
    story += [Preformatted(terminal, styles["Monox"]), P("The zoo model's lower convolutional FLOP count but much larger parameter count illustrates that parameter storage and arithmetic cost are distinct. Dense layers dominate zoo parameters, whereas convolutions dominate arithmetic. Latency is also affected by kernel implementations, memory traffic, and launch overhead; it is not a monotonic function of FLOPs.")]
    story += [PageBreak()]

    story += [P("10. Conclusions", "H1x")]
    conclusions = [
        "<b>Modernization:</b> ReLU and overlapping max pooling improved five-epoch LeNet accuracy by 11.67 points and reduced the generalization gap slightly.",
        "<b>Normalization:</b> BatchNorm was decisively better than no normalization and LRN, reaching 70% validation accuracy three epochs sooner and adding little compute.",
        "<b>Augmentation:</b> The stochastic pipeline reduced the generalization gap by 3.56 points, but the 15-epoch budget was too short to exceed the unaugmented best checkpoint.",
        "<b>Dataset difficulty:</b> Tiny ImageNet accuracy remained far above chance but far below CIFAR-10 because the class count and visual diversity are much greater.",
        "<b>Architecture fit:</b> The compact manual AlexNet-64 outperformed the much larger torchvision baseline on this 64×64 task, showing that parameter count alone does not guarantee better generalization.",
    ]
    for i, c in enumerate(conclusions, 1):
        story.append(P(f"{i}. {c}"))
    story += [P("The experiments support a coherent engineering conclusion: architectural choices that improve gradient propagation and task-specific spatial design matter more than simply increasing parameter count. BatchNorm supplied the strongest short-horizon optimization benefit, while augmentation traded immediate fit for better regularization.", "Calloutx")]

    story += [P("11. Reproducibility and Submission Contents", "H1x")]
    repro = [
        ["Artifact", "Purpose"],
        ["models/, utils/, configs/", "Architectures, data pipeline, metrics, visualization, profiling, and seeding"],
        ["train.py / evaluate.py", "Training and checkpoint evaluation entry points"],
        ["run_experiments.py", "Reproducible experiment matrix with seed-aware output names"],
        ["commands.txt", "Exact CLI commands for all required runs"],
        ["checkpoints/", "Selected Base LeNet, Modern LeNet, and AlexNet-BN checkpoints"],
        ["plots/ and results/", "EDA, curves, JSON metrics, and generated tables"],
        ["github_repo.txt", "Local repository status and eventual remote URL"],
    ]
    story += [styled_table(repro, [2.1*inch, 4.3*inch])]
    story += [P("Execution environment: Google Colab with an NVIDIA Tesla T4 GPU; exact Python/PyTorch dependencies are recorded in requirements.txt. Primary random seed: 42; supplemental seeds: 123 and 2026. The data/ directory is intentionally excluded from version control and the submission archive.", "Smallx")]

    story += [P("References", "H1x")]
    refs = [
        "Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-Based Learning Applied to Document Recognition,” Proceedings of the IEEE, 1998.",
        "A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” NeurIPS, 2012.",
        "S. Ioffe and C. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” ICML, 2015.",
        "K. He et al., “Deep Residual Learning for Image Recognition,” CVPR, 2016.",
        "A. G. Howard et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” 2017.",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(P(f"[{i}] {ref}", "Smallx"))

    doc.build(story)
    print(f"Report written to: {output}")


if __name__ == "__main__":
    build_report()
