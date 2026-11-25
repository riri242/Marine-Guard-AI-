# Datasets

MarineGuard AI Version 1 detects **Net (class 0)** and **Rope (class 1)**. This page records dataset roles and preparation status for the **selected original YOLO26n** checkpoint. It does not describe the later camera-holdout model as the deployed model.

## Dataset sources

| Source | Role | Used in original training? |
|---|---|---|
| **SeaClear Marine Debris Dataset** | Main underwater Training and Validation images | **Yes** — 2,133 Training and 178 Validation images |
| **CleanSea-derived images** | Additional annotated Net/Rope training samples | **Yes** — 193 Training images |
| **Additional background images** | Negative examples, stored as `bg_*` in prepared Training | **Yes** — 300 Training images |
| **TrashCan 1.0** | Separate 214-image external-domain diagnostic | **No** |
| **Roboflow Fishing Net Detection v3** | Investigated as a fourth candidate dataset | **No** — audited but excluded |

**Important:** The selected model used **SeaClear + CleanSea-derived samples + additional backgrounds** for Training. TrashCan was never used to train the selected checkpoint. The fourth Roboflow dataset was inspected but not incorporated.

## Prepared original dataset

The selected checkpoint was trained against the team's `MarineGuard_YOLO_Fixed` dataset:

| Split | Images | Net boxes | Rope boxes |
|---|---:|---:|---:|
| Training | 2,626 | 1,067 | 2,055 |
| Validation | 178 | 7 | 71 |
| External Test (TrashCan) | 214 | See original test labels | See original test labels |

Training contains 2,133 SeaClear images + 193 CleanSea-derived images + 300 additional negative images. The original Validation and Test contain different image sources, and the validation split has only **seven Net annotations**, so its class-level performance estimates have high uncertainty.

### Class mapping

- **SeaClear COCO:** `net_plastic` (category 14) → **Net (0)**; `rope_fiber` (16) and `rope_plastic` (34) → **Rope (1)**.
- **TrashCan:** `trash_net` → **Net (0)**; `trash_rope` → **Rope (1)**.
- All other original categories are outside the two-class detection scope. Images with no target boxes can serve as backgrounds where the preparation procedure retains them.

Earlier source verification compared **2,311 prepared SeaClear images** with original COCO target annotations and found no count/coordinate mismatches under the recorded comparison tolerance. This verifies conversion consistency, **not** annotation quality or source-independent splitting.

## Directory layout

Datasets are **not stored in this GitHub repository**. With appropriate access to the dataset, the expected prepared directory is:

```text
MarineGuard_YOLO_Fixed/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image has a corresponding YOLO `.txt` label file (empty files for negative images). Each non-empty line contains `class_id x_center y_center width height`, using normalized coordinates. Refer to `docs/reproducibility.md` for dataset integrity checks and evaluation instructions.

## Data preparation and limitations

The team's initial data-preparation notebooks cover SeaClear exploration/conversion and TrashCan preparation. **They do not completely reconstruct** how the 193 CleanSea-derived images and 300 added backgrounds were incorporated into the *exact* original Training folder. This provenance must be completed before claiming fully reproducible training from raw data.

The earlier split was not strictly independent by underwater location: all SeaClear Net annotations came from the **Bistrina** site. Grouping by source camera/site folder also produced a validation set with only seven Net instances. Report the selected model as a **Bistrina-focused research prototype**, not a generalized underwater detector.

The subsequent cross-camera experiment (notebooks 05–06) created a separate SeaClear-only Training/Validation split: 1,461 / 850 images, with 589 / 371 Net annotations respectively. That experiment excluded the 193 CleanSea-derived images and 300 added backgrounds, and its checkpoint was **not selected** for the application. The split tests camera transfer within Bistrina, not transfer to unseen locations.

### Fourth dataset: Roboflow

The Roboflow Fishing Net Detection v3 dataset was audited in `03_dataset_validation.ipynb`. It contains 2,266 images across 13 source classes. The candidate required class mapping and cleaning; audits also identified inconsistent annotations and very limited labeled `rope` coverage. The team **excluded it from the final model** rather than incorporating an unverified source late in development. It may be reconsidered in future work after annotation review and source-aware splitting.

## Access and redistribution

- Obtain SeaClear, CleanSea, TrashCan, and Roboflow data **from their respective authorized sources** and check each provider's current license and attribution terms.
- Do **not** upload raw third-party images, exported dataset archives, or private Drive URLs to the public repository without explicit redistribution rights.
- Keep full training images and intermediate checkpoints in authorized storage. The GitHub repository contains preparation notebooks, final inference code, dataset documentation, and aggregate research figures.
- To reproduce evaluation, provide the exact prepared dataset and selected original `best.pt` through an authorized channel. Verify file hashes and recorded library versions.
- Before redistributing model weights, review applicable model/framework licensing and source-data conditions.
