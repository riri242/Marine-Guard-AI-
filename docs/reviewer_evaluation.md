# Reviewer and application evaluation

This protocol measures the practical benefit of MarineGuard AI Version 1. It **does not** alter the selected YOLO26n checkpoint or claim that time savings have already been demonstrated.

## 1. Prepare the evaluation set

- Use a fixed, representative set of authorized underwater images. Include examples with Nets, Ropes, both, and no relevant objects, plus difficult scenes.
- Use images not previously memorized by the reviewer where possible. Record how each image was chosen. Do **not** treat previously used model-development data as a new independent external test.
- Prepare a reference answer for each image: human-reviewed Net and Rope findings. A reviewer may correct existing annotations.
- Use the same image display size and consistent counting/decision criteria for both conditions.

## 2. Compare manual and assisted review

Two conditions:

- **Manual:** A reviewer inspects each image without model predictions, recording whether there are Nets/Ropes and the findings.
- **Assisted:** A reviewer inspects the original image plus MarineGuard annotations, then confirms or corrects the findings. Model results are **not** automatically accepted.

Balance image difficulty and order. If using one reviewer, avoid immediately repeating identical images across both conditions because familiarity can bias speed. Consider comparable randomized subsets and state this limitation. If possible, use multiple reviewers and counterbalanced ordering.

Record one row per image and condition in a private CSV with the following fields:

```text
reviewer_id,image_id,condition,order_index,start_time,end_time,duration_seconds,net_confirmed,rope_confirmed,false_positive_corrections,missed_object_corrections,notes
```

Report the number of reviewers and images, average and median seconds per image, missed objects/corrections, and whether either workflow produced systematically different decisions.

Time savings calculation:

```text
time_savings_percent = (mean_manual_seconds - mean_assisted_seconds) / mean_manual_seconds * 100
```

Only report an observed percentage after actual measurements. If manual review is faster or quality decreases with assistance, report that outcome.

## 3. Batch-processing latency

After installing dependencies and placing the original checkpoint at `models/best.pt`, run:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_batch --images "D:\path\to\authorized\sample_images" --max-images 30
```

The script writes `outputs/batch_benchmark/image_times.csv` and `summary.json` containing mean, median and p95 per-image time, images per second, hardware and software versions.

This is **local application-side processing**, including image loading, YOLO inference, annotation and PNG encoding. It excludes model load, warmup, network transfer, Streamlit rendering and human decisions. Do not describe it as complete Streamlit Cloud latency. For a deployed-app measurement, separately record browser-observed upload-to-result elapsed time on that deployment.

## 4. Application checks

Before final submission, verify with real authorized images:

- Correct original `best.pt` loads and reports class IDs `0 = Net`, `1 = Rope`.
- Uploaded images preserve orientation; boxes and confidence values match detections.
- Net and Rope thresholds are `0.25` and `0.15`, respectively.
- Images with no detections show a clear, non-definitive message.
- Annotated image downloads work; invalid/corrupt uploads are handled.
- CPU and hosted-app response time, RAM use, and startup behavior are acceptable for the demonstration.
- No third-party raw datasets, private keys or passwords are published.

## Reporting

This protocol is a **measurement plan**, not a completed study. Keep raw image-level review data private unless participants authorize sharing. Include only aggregate results and limitations in the final report.
