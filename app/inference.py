"""Reusable inference for the original MarineGuard YOLO26n model.

No Colab, Google Drive, or web server required.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from ultralytics import YOLO

# Original model's provisional per-class thresholds, measured on limited validation.
CONFIDENCE_BY_CLASS = {0: 0.25, 1: 0.15}
LOWEST_CONFIDENCE = min(CONFIDENCE_BY_CLASS.values())


def load_model(checkpoint: str | Path) -> YOLO:
    """Load the original trained checkpoint (never starts training)."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model checkpoint: {path}. Copy the ORIGINAL YOLO26n best.pt "
            "to models/best.pt."
        )
    model = YOLO(str(path))
    expected = {0: "net", 1: "rope"}
    actual = {int(k): str(v).lower() for k, v in model.names.items()}
    if actual != expected:
        raise ValueError(f"Unexpected model classes: {model.names}; expected Net/Rope.")
    return model


def predict(image: Image.Image, model: YOLO) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Return (annotated PIL image, JSON-friendly detections).

    Uses Ultralytics PIL input so RGB/BGR conversion is handled correctly.
    XYXY boxes are in original image pixel coordinates.
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    result = model.predict(
        source=rgb,
        imgsz=640,
        conf=LOWEST_CONFIDENCE,
        iou=0.70,
        verbose=False,
        save=False,
    )[0]

    boxes = result.boxes
    keep: list[int] = []
    if boxes is not None:
        for i, (cls, confidence) in enumerate(zip(boxes.cls.tolist(), boxes.conf.tolist())):
            if float(confidence) >= CONFIDENCE_BY_CLASS.get(int(cls), 1.0):
                keep.append(i)
        result.boxes = boxes[keep]

    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls.item())
            detections.append({
                "class_id": cls,
                "class_name": str(model.names[cls]),
                "confidence": round(float(box.conf.item()), 4),
                "box_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            })

    # Ultralytics plot() outputs BGR ndarray; convert to RGB PIL.
    annotated = Image.fromarray(result.plot()[..., ::-1].copy())
    return annotated, detections


def png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
