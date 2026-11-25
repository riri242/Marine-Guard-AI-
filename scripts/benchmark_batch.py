"""Measure end-to-end local batch processing using the selected YOLO26n.

Includes image reading, inference, drawing predictions and PNG encoding.
Excludes human review and Streamlit network/browser overhead. No training.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import ultralytics
from PIL import Image

from app.inference import load_model, png_bytes, predict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="Directory of authorized representative images")
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--max-images", type=int, default=0,
                        help="Limit image count (0 means all)")
    parser.add_argument("--output", type=Path, default=Path("outputs/batch_benchmark"))
    args = parser.parse_args()

    if not args.images.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.images}")
    if args.max_images < 0:
        raise ValueError("--max-images must be 0 or greater")

    paths = sorted(p for p in args.images.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.max_images:
        paths = paths[:args.max_images]
    if not paths:
        raise ValueError("No supported images were found")

    load_start = time.perf_counter()
    model = load_model(args.model)
    load_seconds = time.perf_counter() - load_start

    # Warm up once with a representative image; this call is excluded from timing.
    with Image.open(paths[0]) as source:
        warmup = source.copy()
    predict(warmup, model)

    rows = []
    for image_path in paths:
        start = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.copy()
        annotated, detections = predict(image, model)
        png_bytes(annotated)  # Include encoding used by the download feature.
        elapsed = time.perf_counter() - start
        rows.append({
            "image": image_path.name,
            "latency_seconds": round(elapsed, 6),
            "detections": len(detections),
        })

    elapsed_values = [row["latency_seconds"] for row in rows]
    total = sum(elapsed_values)
    summary = {
        "images": len(rows),
        "model_load_seconds": round(load_seconds, 3),
        "total_processing_seconds": round(total, 3),
        "mean_seconds_per_image": round(statistics.mean(elapsed_values), 3),
        "median_seconds_per_image": round(statistics.median(elapsed_values), 3),
        "p95_seconds_per_image": round(float(np.percentile(elapsed_values, 95)), 3),
        "throughput_images_per_second": round(len(rows) / total, 3) if total > 0 else None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "scope": "local file read + YOLO inference + annotation + PNG encoding; warmup and model load excluded",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "image_times.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "latency_seconds", "detections"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
