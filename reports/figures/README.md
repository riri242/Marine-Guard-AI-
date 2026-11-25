# MarineGuard AI — Report figures

These figures summarize **saved numerical results** from the model-development notebooks. They are not newly measured experimental results, and producing the images does not train or evaluate any model.

| Figure | Purpose |
|---|---|
| 01 | Original YOLO26n: precision, recall and mAP50–95 by class |
| 02 | YOLO12n versus YOLO26n finalist mAP on the original validation set |
| 03 | Per-class mAP for original versus CleanSea-excluded ablation |
| 04 | Recall differences in the CleanSea-excluded ablation |
| 05 | Selected checkpoint: original validation versus TrashCan external diagnostic |
| 06 | Source-camera train/validation split used only for Version 2 investigation |
| 07 | F1 at different confidence thresholds; CSV included |

## Regenerate the figures

From the repository root:

```bash
python -m pip install matplotlib numpy
python scripts/generate_report_figures.py
```

The script creates seven PNGs from recorded metrics and the threshold-sweep CSV. It does **not** run the neural network. For fresh measured plots, use `notebooks/model_development/07_final_model_evaluation_reporting.ipynb` with the actual `best.pt` and prepared dataset.

**Interpretation:** The original Validation set contains only seven Net instances; TrashCan external-domain performance is near zero; the CleanSea ablation used a different training history; Version 2 used a different split and was not deployed. Avoid portraying any figure as evidence of production readiness.

The source high-resolution charts were previously exported separately. The PNGs in this folder are regenerated from their recorded numeric inputs; their rendering may differ, while their reported measurements are preserved.
