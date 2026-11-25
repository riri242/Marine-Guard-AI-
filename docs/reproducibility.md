# Reproducibility

The historical experiment notebooks are preserved in `notebooks/model_development/`. The reproducible **evaluation** path below uses the selected original YOLO26n checkpoint; it does not rerun model selection or training.

## Inputs

Obtain the team's prepared `MarineGuard_YOLO_Fixed` folder, including `data.yaml`, `images/{train,val,test}`, and `labels/{train,val,test}`, from the authorized dataset owner. The raw images are not distributed in this repository. Place the selected **original** checkpoint at `models/best.pt`.

Expected dataset inventory (from the recorded experiment):

| Split | Images | Net boxes | Rope boxes |
|---|---:|---:|---:|
| Train | 2,626 | 1,067 | 2,055 |
| Validation | 178 | 7 | 71 |
| TrashCan test | 214 | External diagnostic | External diagnostic |

The original training images consist of 2,133 SeaClear images, 193 CleanSea-derived images, and 300 additional background images. The origin and preparation of the 300 backgrounds and addition of the 193 CleanSea images are not fully reconstructed in the uploaded preparation notebooks. These inputs are required for a full training reproduction.

## Evaluate the selected checkpoint (no training)

Use Python 3.11 or 3.12 and a compatible PyTorch installation. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-research.txt
python scripts/reproduce_evaluation.py --dataset "D:\\path\\to\\MarineGuard_YOLO_Fixed" --checkpoint models/best.pt --device cpu
```

On Colab, use `--dataset /content/drive/MyDrive/MarineGuard_YOLO/MarineGuard_YOLO_Fixed --checkpoint /content/drive/MyDrive/MarineGuard_Model_Selection/budget20_30_v1/runs_budget20_30_v1/final/yolo26n/weights/best.pt --device 0`.

To audit the data without the checkpoint, append `--audit-only`.

Outputs are saved under `reproduction_outputs/`: image inventory with SHA-256 hashes, resolved YAML, mAP/precision/recall JSON, and Ultralytics plots. An optional `--expected-checkpoint-sha256 <hash>` rejects a different checkpoint.

### Recorded evaluation reference

| Split | mAP50 | mAP50–95 |
|---|---:|---:|
| Original validation | 0.5334 | 0.3271 |
| TrashCan external diagnostic | 0.00513 | 0.00175 |

The script records deviations from these values. Results should be close when the checkpoint, images, annotation files, inference settings, and library versions match; **bit-identical results across hardware, OS, and different dependency versions are not guaranteed**.

These are not unbiased deployment estimates: the original validation contains only seven Net instances and the TrashCan set was consulted during development.

## Reproducing training

Notebook 01 documents five-model screening followed by two finalist runs. The selected original YOLO26n was from an intended 30-epoch finalist stage but the recorded run ended after **21 epochs**, not 30. Reproducing its exact weights requires the same verified dataset bytes, pretrained checkpoint version, random seed, software/CUDA versions, training configuration, and run-interruption history. The notebook may reuse checkpoints found in Drive, and its paths refer to the original Colab environment.

The research notebooks are a **historical experimental record**, not a guaranteed single-command training pipeline. Interactive uploads and `TRAIN_MODE` decisions in later notebooks likewise prevent an unattended `Run all` on a fresh machine. To reproduce the reported evaluation, use the dedicated script above. The team should record the SHA-256 hash of the original `best.pt` and resolve the source of the 300 extra backgrounds before claiming full end-to-end training reproducibility.

## Notebook reading order

1. Model selection: five candidates, finalists, threshold sweep.
2. Model evaluation: qualitative review, 640/960 comparison, TrashCan diagnostic.
3. Dataset validation: provenance, original COCO labels, fourth dataset (Roboflow) inspection.
4. CleanSea ablation: exclude 193 cutout samples and retrain separately.
5. Camera-held-out manifest: deterministic SeaClear split.
6. Cross-camera training: follow-up experiment and model comparison.

The fourth Roboflow dataset was **inspected but not included** in the selected model. The later camera-held-out study excludes the CleanSea cutouts and additional backgrounds; because its dataset differs, direct score comparisons do not isolate the impact of camera separation.
