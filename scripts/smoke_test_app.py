"""Smoke-test the selected checkpoint and shared Streamlit inference function.

This checks integration, not detection accuracy. It never retrains a model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from app.inference import load_model, png_bytes, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--image", type=Path, default=None,
                        help="Optional example underwater image; otherwise use a blank RGB image")
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke_test.png"))
    args = parser.parse_args()

    model = load_model(args.model)
    if args.image is not None:
        with Image.open(args.image) as source:
            image = source.copy()
    else:
        image = Image.new("RGB", (640, 480), color=(22, 54, 67))

    annotated, detections = predict(image, model)
    assert annotated.size == image.size, "Prediction image size changed unexpectedly"
    assert isinstance(detections, list)
    for item in detections:
        assert item["class_id"] in (0, 1), item
        assert 0 <= item["confidence"] <= 1, item
        assert len(item["box_xyxy"]) == 4, item

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(png_bytes(annotated))
    print(f"PASS: loaded model, completed inference, returned {len(detections)} candidate detections.")
    print(f"Annotated image: {args.output}")
    if args.image is None:
        print("Blank-image inference only checks the integration; it does not measure accuracy.")


if __name__ == "__main__":
    main()
