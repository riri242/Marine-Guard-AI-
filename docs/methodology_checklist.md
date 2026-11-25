# Methodology checklist

This records the **implemented** methods, their evidence, and remaining work for MarineGuard AI Version 1. It is not a retrospective claim that every initial requirement was met.

| Original requirement | Status | Evidence and action |
|---|---|---|
| Grouped split; separate external test | Partial | Original `MarineGuard_YOLO_Fixed` used 2,626 train / 178 validation images with only seven validation Nets. A later source-camera holdout is in notebooks 05–06 (1,461 train / 850 validation). It is **cross-camera, not cross-location** for Nets. TrashCan was withheld from model training but consulted during development; report it as an external-domain diagnostic rather than a final untouched test. |
| Underwater augmentation and Net sampling | Partial | Notebook 01 trains with HSV (h=0.01, s=0.30, v=0.20), translation=0.05, scale=0.20, horizontal flip=0.5 and mosaic=0.30. Rotation, vertical flip and MixUp are disabled. Augmentations are applied online to training batches. No explicit class-based Net oversampling was implemented; document this rather than imply otherwise. |
| Model, hardware, training, checkpoint selection | Covered, with qualification | Notebook 01 documents candidate models, 640px, batch=8, SGD learning rate 0.01, seed=42, and candidate/finalist stages. Selected original YOLO26n finalist training recorded **21 epochs**, rather than the intended 30. Ultralytics `best.pt` uses validation fitness (mAP50–95-weighted) as checkpoint selection. Record hardware and installed versions alongside any rerun. |
| Class mAP/recall targets and validation thresholds | Partial | Per-class precision, recall, mAP50–95, and F1 versus confidence are in notebooks 01–02. Provisional thresholds: Net=0.25, Rope=0.15, selected on the small original validation set. **Numerical acceptance targets were not fixed in advance**; do not invent retrospective pass/fail targets. |
| Human review-time savings and batch latency | Partial | Notebook 01 records a CPU inference benchmark; app-level response and reviewer savings are not yet measured. See the procedure below. |
| Human-approved GenAI reporting | Optional / not implemented | A future extension, not part of Version 1. Do not claim that an LLM generates or approves reports. |

## Remaining verification, without retraining

**Reviewer task:** Select a fixed set of representative images that the reviewer has not memorized. Measure time to inspect and record verified Net/Rope decisions **manually** and **with MarineGuard** using the same annotation/decision criteria. Counterbalance the order or use matched sets to limit learning bias. Record image count, reviewer count, start/end times, confirmed findings, and corrections. Report both raw times and `(manual_time - assisted_time) / manual_time × 100%` as a *measured* result. If no timings are collected, mark reviewer savings as **not evaluated**.

**Batch processing:** Run a fixed image list through the deployed Streamlit inference path on the intended machine. Record total wall-clock time including image loading and postprocessing; report `total seconds / images` and `images / total seconds`, together with hardware and batch size. Keep model-only benchmark results separate from end-to-end application measurements.

**Data provenance:** The original training set contains 2,133 SeaClear images, 193 CleanSea images, and 300 extra backgrounds. The source and inclusion procedure for the CleanSea and extra background samples remain incompletely documented. Do not claim full from-raw-data training reproduction until this is resolved.

**External generalization:** Original YOLO26n performs poorly on TrashCan. The project scope is a Bistrina-focused research prototype, not a general underwater debris detector. Document additional locations and more verified Net/Rope examples as future work.

## Evidence map

- `notebooks/model_development/01_model_selection.ipynb`: augmentation, five-model screening, finalist training, threshold sweep, CPU benchmark
- `notebooks/model_development/02_model_evaluation.ipynb`: visualization, class metrics, resolution tests, TrashCan evaluation
- `notebooks/model_development/03_dataset_validation.ipynb`: provenance, COCO/YOLO consistency, Roboflow candidate dataset
- `notebooks/model_development/04_cleansea_ablation.ipynb`: CleanSea removal experiment
- `notebooks/model_development/05_cross_camera_split.ipynb`: deterministic source-camera split
- `notebooks/model_development/06_cross_camera_training_evaluation.ipynb`: alternate model training and cross-camera results
- `docs/reproducibility.md`: how to rerun evaluation with the original checkpoint and prepared dataset
