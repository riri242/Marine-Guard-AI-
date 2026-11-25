"""Regenerate seven MarineGuard report charts from recorded experiment metrics.

This is a reporting utility, not a model evaluation or training script.
For new experimental results, edit the source metrics only after verification.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "reports" / "figures"
ROOT.mkdir(parents=True, exist_ok=True)


def finish(filename: str, title: str, note: str, ylabel: str, ax):
    ax.set_title(title, pad=16, weight="bold", fontsize=13)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.figure.text(0.5, 0.025, note, ha="center", fontsize=8, color="dimgray")
    ax.figure.tight_layout(rect=(0, 0.07, 1, 1))
    ax.figure.savefig(ROOT / filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(ax.figure)


def grouped(filename, title, categories, series, ylabel, note, ylim=None, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(9.7, 5.8))
    x = np.arange(len(categories))
    width = 0.72 / len(series)
    for j, (label, values) in enumerate(series):
        pos = x - 0.36 + (j + 0.5) * width
        bars = ax.bar(pos, values, width=width, label=label)
        for bar, value in zip(bars, values):
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, categories)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(0, ylim)
    else:
        ax.set_ylim(0, max(v for _, vals in series for v in vals) * 1.21)
    finish(filename, title, note, ylabel, ax)


grouped(
    "01_original_yolo26n_class_metrics.png",
    "Original YOLO26n | Class-level validation metrics",
    ["Net", "Rope"],
    [("Precision", [45.01, 94.43]), ("Recall", [85.71, 47.78]),
     ("mAP50–95", [29.94, 35.47])],
    "Percent (%)",
    "178 validation images; only 7 Net instances. Results are not a reliable generalization estimate.",
    ylim=112,
)
grouped(
    "02_finalist_class_map.png",
    "Finalist comparison | Original validation split",
    ["Net", "Rope"],
    [("YOLO12n", [43.7, 6.3]), ("YOLO26n", [29.94, 35.47])],
    "mAP50–95 (%)",
    "Selected YOLO26n for a better balance across both target classes.",
    ylim=58,
)
grouped(
    "03_cleansea_ablation_class_map.png",
    "CleanSea ablation | Per-class validation mAP",
    ["Net", "Rope"],
    [("Original", [29.94, 35.47]), ("Without CleanSea", [52.19, 25.25])],
    "mAP50–95 (%)",
    "Same validation images; different training duration and composition prevent a causal conclusion.",
    ylim=66,
)
grouped(
    "04_cleansea_recall_tradeoff.png",
    "CleanSea ablation | Recall trade-off",
    ["Original", "Without CleanSea"],
    [("Net", [85.71, 85.71]), ("Rope", [47.78, 16.03])],
    "Recall (%)",
    "Net recall comes from only seven instances; Rope recall declined in the ablation.",
    ylim=110,
)
grouped(
    "05_internal_vs_external_generalization.png",
    "Original YOLO26n | Internal versus external evaluation",
    ["Original validation", "TrashCan external"],
    [("mAP50–95", [32.71, 0.175])],
    "mAP50–95 (%)",
    "Different image domains; TrashCan was consulted during development and is an external diagnostic.",
    ylim=43,
    fmt="{:.3f}",
)
grouped(
    "06_camera_holdout_split.png",
    "Source-camera split | Version 2 investigation",
    ["Training: 1,461 images", "Validation: 850 images"],
    [("Net", [589, 371]), ("Rope", [1405, 642])],
    "Annotated object instances",
    "Alternative experiment only; this split was not used to train the selected original checkpoint.",
    ylim=1700,
    fmt="{:.0f}",
)

thresholds = {}
with (ROOT / "07_threshold_sweep_yolo26n.csv").open(newline="", encoding="utf-8") as handle:
    for record in csv.DictReader(handle):
        thresholds.setdefault(record["class"], []).append(
            (float(record["threshold"]), float(record["f1"]))
        )

fig, ax = plt.subplots(figsize=(9.7, 5.8))
for cls in ["Net", "Rope"]:
    points = sorted(thresholds[cls])
    x, y = zip(*points)
    ax.plot(x, [v * 100 for v in y], marker="o", label=cls)
    top = max(points, key=lambda p: p[1])
    ax.annotate(f"{cls}: {top[0]:.2f} / {top[1] * 100:.1f}%",
                (top[0], top[1] * 100), xytext=(6, 9), textcoords="offset points",
                fontsize=9)
ax.legend()
ax.set_xlabel("Confidence threshold")
ax.set_ylim(0, 85)
finish("07_confidence_threshold_f1.png",
       "Original YOLO26n | Confidence threshold and F1",
       "Original validation, greedy class-matched IoU >= 0.50. Thresholds were selected on limited data.",
       "F1 (%)", ax)

print("Generated seven report PNG charts in:", ROOT)
