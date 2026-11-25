# MarineGuard AI

**Underwater Fishing Gear Detection | Bistrina-focused Research Prototype**

MarineGuard AI is a computer vision capstone project that detects and localizes **fishing nets** and **ropes** in underwater images. The project explores whether a lightweight object detector can support the review of underwater imagery by highlighting potential fishing gear debris for human verification.

**Version 1** focuses on imagery similar to the **Bistrina environment** represented in the SeaClear dataset. It is a research prototype, **not a validated system for all marine locations**.

## What the application does

- Upload an underwater JPG or PNG image.
- Run the team's selected **original YOLO26n** detector for two classes: **Net (0)** and **Rope (1)**.
- Show predicted bounding boxes, class names and confidence scores alongside the input.
- Download the annotated image. Predictions are suggestions and require human review.

The Streamlit interface calls the model **directly** via `app/inference.py`; it does not need a separate FastAPI service or a training dataset at runtime.

## Project workflow

| Stage | Work completed |
|---|---|
| Data preparation | Converted selected SeaClear and TrashCan annotations to two-class YOLO format; the original selected training folder also contains CleanSea-derived samples and additional negative images. |
| Model selection | Screened **YOLOv8n, YOLOv10n, YOLO11n, YOLO12n and YOLO26n**; advanced two finalists for further evaluation. |
| Model evaluation | Compared class-specific precision, recall, mAP and F1; inspected predictions, image resolution, confidence thresholds and inference speed. |
| Dataset investigation | Verified SeaClear-to-YOLO annotation consistency, studied source-camera distributions, and audited a fourth Roboflow dataset that was **not used**. |
| Follow-up experiments | Evaluated removing CleanSea-derived samples and independently trained a SeaClear-only cross-camera variant. |
| Final model | Retained the **original YOLO26n checkpoint** as the Version 1 demonstration model. |

These are **historical experimental stages**, not a single unattended training command. The original YOLO26n finalist run recorded **21 epochs** before interruption, rather than completing the planned 30.

## Datasets

| Dataset/source | Role in selected model |
|---|---|
| SeaClear | Main training and validation source |
| CleanSea-derived samples | 193 additional training images |
| Extra negative/background images | 300 additional training images |
| TrashCan 1.0 | External-domain diagnostic **only**, not training |
| Roboflow Fishing Net Detection v3 | Investigated and excluded |

The **original training split** contains **2,626 images** (2,133 SeaClear + 193 CleanSea-derived + 300 additional backgrounds). The original validation has **178 images** and only **7 annotated Net instances**. The external TrashCan diagnostic has **214 images**.

Full source/class mapping, licensing cautions and preparation limitations are in [Dataset documentation](docs/datasets.md).

Raw image datasets are **not distributed** in this repository. The exact provenance and incorporation of the 193 CleanSea-derived images and 300 additional backgrounds remain incompletely reconstructed in the historical preprocessing notebooks.

## Recorded model results

Selected original YOLO26n, evaluated on its original 178-image validation split:

| Metric | Overall | Net | Rope |
|---|---:|---:|---:|
| Precision | 69.72% | 45.01% | 94.43% |
| Recall | 66.75% | 85.71% | 47.78% |
| mAP50 | 53.34% | 40.8% | 65.8% |
| mAP50–95 | **32.71%** | 29.94% | 35.47% |

The external TrashCan diagnostic achieved **0.175% mAP50–95**, showing a substantial generalization limitation. The original validation split's seven Net annotations are too few to support a reliable Net-performance claim.

The separate source-camera experiment evaluated a different training distribution and is retained for methodological analysis. It is **not** the checkpoint used in the application and does not establish cross-location generalization.

The original model's displayed prediction thresholds are **0.25 for Net** and **0.15 for Rope**, with 640px inference. Thresholds were selected on the limited original validation set and do not change the model's learned weights.

See [Methodology checklist](docs/methodology_checklist.md) and [Report figures](reports/figures/README.md) for supporting evidence and outstanding measurements.

## Repository structure

```text
Marine-Guard-AI-/
├── app/
│   ├── inference.py                   # Shared YOLO inference
│   └── streamlit_app.py               # User interface
├── models/
│   └── best.pt                        # Add selected original checkpoint
├── notebooks/
│   ├── data_preparation/              # Historical preprocessing work
│   └── model_development/             # 01–07: selection through reporting
├── scripts/
│   ├── smoke_test_app.py              # Check model integration
│   ├── reproduce_evaluation.py        # Dataset audit + model metrics
│   └── generate_report_figures.py     # Charts from recorded metrics
├── reports/figures/
├── docs/
│   ├── datasets.md
│   ├── methodology_checklist.md
│   └── reproducibility.md
├── requirements.txt
└── packages.txt
```

## Run the Streamlit application

**Requirements:** Python 3.11 or 3.12, the selected original `best.pt`, and the application dependencies.

1. Retrieve the selected checkpoint from the team's authorized Google Drive location:

   `MarineGuard_Model_Selection/budget20_30_v1/runs_budget20_30_v1/final/yolo26n/weights/best.pt`

2. Copy it to **`models/best.pt`**. Do **not** substitute the later Version 2 checkpoint.

3. From the repository root in **VS Code PowerShell**:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe scripts/smoke_test_app.py
   .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
   ```

   The smoke test verifies that the model loads, inference runs and a PNG is produced. A synthetic input **does not measure accuracy**. For a meaningful visual check, run `scripts/smoke_test_app.py --image "path/to/underwater_image.jpg"` with an authorized representative image.

4. Open the local Streamlit address and test upload, predicted labels, no-detection messaging and annotated-image download.

**Note:** The repository currently contains only a model placeholder. Streamlit cannot perform predictions until the correct checkpoint has been supplied. CPU-only startup and inference may be slower than Colab GPU inference.

## Deploy to Streamlit Community Cloud

1. Confirm the team can legally distribute the trained checkpoint, then make it available to the deployment at `models/best.pt`. Keep datasets and credentials out of the repository.
2. Merge the reviewed changes to the branch you plan to deploy.
3. In [Streamlit Community Cloud](https://share.streamlit.io), select the repository `riri242/Marine-Guard-AI-`, deployment branch and **entrypoint `app/streamlit_app.py`**.
4. Select a supported Python version and verify dependency installation, memory usage and image inference using the deployed URL.

The app needs the checkpoint and code, **not SeaClear, CleanSea or TrashCan datasets**. `requirements.txt` contains app dependencies, and `packages.txt` supplies Linux runtime libraries.

## Reproduce the evaluation

For **evaluation only** (no training), use the original prepared `MarineGuard_YOLO_Fixed` dataset and the selected checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe scripts/reproduce_evaluation.py --dataset "D:\path\to\MarineGuard_YOLO_Fixed" --checkpoint models/best.pt --device cpu
```

The script audits image and label counts, saves file hashes and computes validation and TrashCan metrics. See [Reproducibility instructions](docs/reproducibility.md) for full assumptions and limitations. [Notebook 07](notebooks/model_development/07_final_model_evaluation_reporting.ipynb) provides a Colab-based evaluation/reporting workflow.

**Historical notebook warning:** The data-preparation notebooks retain hardcoded local Windows paths and some unsafe operations, including destructive CleanSea removal. They are preserved as a record and **must not be run unchanged against original datasets**. Their outputs do not amount to a verified from-raw-data training reproduction.

## Limitations and remaining work

- **Dataset coverage:** Original validation contains seven Net instances; all SeaClear Net annotations originate from Bistrina. More independent sites and cameras are needed.
- **Generalization:** Performance is very weak on TrashCan. Detection results require human verification.
- **Sampling:** HSV color augmentation, translation, scaling, horizontal flips and mosaic were used. Explicit class-specific Net oversampling was **not** implemented.
- **Acceptance criteria:** Per-class evaluation metrics were measured, but numerical acceptance targets were not specified before training.
- **Operational validation:** Reviewer time savings and full Streamlit batch-processing latency have **not yet been measured**. The recorded CPU benchmark is model-only.
- **Optional GenAI reports:** Not implemented in Version 1; a human-approved reporting assistant remains future work.
- **Reproducibility:** Dataset provenance, exact selected checkpoint hash and a verified fresh-environment evaluation are still required before claiming full reproducibility.

Future development may include independently annotated Net/Rope imagery from additional underwater locations, better class coverage and sampling, stronger cross-domain evaluation, and a human-reviewed reporting workflow.

## Licensing and attribution

This project builds on Ultralytics YOLO and third-party research datasets. Before publishing or distributing model weights, review the applicable **Ultralytics AGPL-3.0 or commercial licensing terms** and the individual dataset licenses. Dataset links, license status and redistribution rights should be verified by the team. Do not upload unlicensed raw images, private data or credentials.

**Project status:** Selected checkpoint and inference code are prepared; repository integration, deployment checks and application-level evaluation remain to be completed.
