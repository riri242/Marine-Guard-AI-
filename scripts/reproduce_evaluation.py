"""Reproduce evaluation of the selected original MarineGuard YOLO26n.

Run from the repository root with the original MarineGuard_YOLO_Fixed dataset.
The script audits source files, evaluates the checkpoint and saves metrics.
It does not retrain models or modify datasets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED = {
    "train": {"images": 2626, "net": 1067, "rope": 2055},
    "val": {"images": 178, "net": 7, "rope": 71},
    "test": {"images": 214},
}
RECORDED = {
    "val": {"mAP50": 0.5334, "mAP50_95": 0.3271},
    "test": {"mAP50": 0.00513, "mAP50_95": 0.00175},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_names(config: dict) -> None:
    names = config.get("names")
    if isinstance(names, dict):
        values = [names.get(i, names.get(str(i))) for i in (0, 1)]
    elif isinstance(names, list):
        values = names
    else:
        raise ValueError("data.yaml needs Net/Rope class names")
    if [str(x).lower() for x in values] != ["net", "rope"]:
        raise ValueError(f"Expected [Net, Rope], found {values}")


def audit(root: Path, split: str, seen: dict) -> tuple[dict, list[dict]]:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing images or labels for {split}: {root}")
    images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    counts = Counter()
    rows = []
    for image in images:
        rel = image.relative_to(images_dir)
        label = labels_dir / rel.with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Missing label: {label}")
        fingerprint = sha256(image)
        earlier = seen.get(fingerprint)
        if earlier and earlier[0] != split:
            raise ValueError(f"Exact image duplicate across splits: {earlier[1]} and {image}")
        seen[fingerprint] = (split, str(image))
        per_image = Counter()
        for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5 or values[0] not in ("0", "1"):
                raise ValueError(f"Invalid YOLO label: {label}:{line_no}")
            x, y, w, h = [float(v) for v in values[1:]]
            if not (all(map(math.isfinite, (x, y, w, h)))
                    and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                    and x - w / 2 >= -1e-5 and x + w / 2 <= 1 + 1e-5
                    and y - h / 2 >= -1e-5 and y + h / 2 <= 1 + 1e-5):
                raise ValueError(f"Invalid bounding box: {label}:{line_no}")
            per_image["net" if values[0] == "0" else "rope"] += 1
        counts.update(per_image)
        if not per_image:
            counts["backgrounds"] += 1
        rows.append({"split": split, "image": str(image.relative_to(root)),
                     "sha256": fingerprint, "net": per_image["net"], "rope": per_image["rope"]})
    result = {"images": len(images), "net": counts["net"], "rope": counts["rope"],
              "backgrounds": counts["backgrounds"]}
    if any(result.get(k) != v for k, v in EXPECTED[split].items()):
        raise ValueError(f"Unexpected {split} composition: {result}; expected {EXPECTED[split]}")
    return result, rows


def validate(model, data_yaml: Path, split: str, output: Path, device: str) -> dict:
    metrics = model.val(
        data=str(data_yaml), split=split, imgsz=640, batch=4,
        device=device, workers=0, conf=0.001, iou=0.70,
        plots=True, project=str(output / "plots"), name=split,
        exist_ok=True, verbose=False,
    )
    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "classes": {},
    }
    for index, class_id in enumerate(metrics.box.ap_class_index):
        class_id = int(class_id)
        summary["classes"]["Net" if class_id == 0 else "Rope"] = {
            "precision": float(metrics.box.p[index]),
            "recall": float(metrics.box.r[index]),
            "mAP50": float(metrics.box.ap50[index]),
            "mAP50_95": float(metrics.box.maps[class_id]),
        }
    summary["recorded"] = RECORDED[split]
    summary["difference"] = {key: round(summary[key] - value, 7)
                             for key, value in RECORDED[split].items()}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to original MarineGuard_YOLO_Fixed directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("reproduction_outputs"))
    parser.add_argument("--device", default="cpu", help="CPU or CUDA device, e.g. 0")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    args = parser.parse_args()

    root = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / "data.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing data.yaml: {config_path}")
    check_names(yaml.safe_load(config_path.read_text(encoding="utf-8")))

    inventory = []
    summaries = {}
    seen = {}
    for split in ("train", "val", "test"):
        summaries[split], rows = audit(root, split, seen)
        inventory.extend(rows)
        print(f"{split}: {summaries[split]}")

    with (output / "dataset_inventory.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "image", "sha256", "net", "rope"])
        writer.writeheader()
        writer.writerows(inventory)

    # Resolve the original YAML against the current machine, without changing it.
    resolved = output / "resolved_data.yaml"
    resolved.write_text(yaml.safe_dump({
        "path": str(root), "train": "images/train", "val": "images/val",
        "test": "images/test", "nc": 2, "names": ["Net", "Rope"],
    }, sort_keys=False), encoding="utf-8")

    report = {
        "source_dataset": str(root),
        "source_yaml_sha256": sha256(config_path),
        "audited_splits": summaries,
        "checkpoint": None,
        "metrics": None,
        "python": platform.python_version(),
        "device": args.device,
    }
    if not args.audit_only:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing original model: {checkpoint}")
        checksum = sha256(checkpoint)
        if (args.expected_checkpoint_sha256
                and checksum.lower() != args.expected_checkpoint_sha256.lower()):
            raise ValueError("Checkpoint SHA256 does not match the expected file")
        import torch
        import ultralytics
        from ultralytics import YOLO

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = YOLO(str(checkpoint))
        if {int(k): str(v).lower() for k, v in model.names.items()} != {0: "net", 1: "rope"}:
            raise ValueError(f"Unexpected model classes: {model.names}")
        results = {}
        for split in ("val", "test"):
            results[split] = validate(model, resolved, split, output, args.device)
            print(f"{split}: mAP50={results[split]['mAP50']:.4f}, "
                  f"mAP50–95={results[split]['mAP50_95']:.4f}")
        report.update({"checkpoint": str(checkpoint), "checkpoint_sha256": checksum,
                       "metrics": results, "torch": torch.__version__,
                       "ultralytics": ultralytics.__version__})

    (output / "reproduction_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved audit and evaluation artifacts: {output}")


if __name__ == "__main__":
    main()
