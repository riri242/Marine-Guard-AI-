# MarineGuard AI — Bistrina-focused Version 1

An underwater research prototype detecting **Net** and **Rope** objects in uploaded images. Uses the team's selected **original YOLO26n** checkpoint. **No API server and no retraining are required.**

## Setup

1. Download the original model file from Google Drive:
   `MarineGuard_Model_Selection/budget20_30_v1/runs_budget20_30_v1/final/yolo26n/weights/best.pt`
2. Place the file in `models/best.pt`. Do not substitute the v2 checkpoint.
3. Install Python (3.11 or 3.12 recommended), then:
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
4. Upload JPG/PNG files in the application.

## Repository layout

- `inference.py`: the reusable image prediction function, including provisional class thresholds and structured detection results.
- `streamlit_app.py`: self-contained upload UI. Calls `inference.py` in the same Python process.
- `models/best.pt`: trained original YOLO26n checkpoint (you supply this file).
- `requirements.txt`: Python dependencies.
- `packages.txt`: system OpenCV runtime support for Linux hosting.

## Deploy to Streamlit Community Cloud

1. Push these source files **and the authorized model checkpoint** to a GitHub repository.
2. Open https://share.streamlit.io and create an app from the repo.
3. Choose the branch and set **main file path** to `streamlit_app.py`.
4. Check the app logs if dependencies or memory prevent startup.

CPU inference works but can be slower than Colab GPU. Community Cloud resource limits depend on deployment conditions.

## About prediction settings

- Input resolution `imgsz=640`.
- Net minimum confidence: `0.25`.
- Rope minimum confidence: `0.15`.
- NMS IoU: `0.70`.

These thresholds were selected on a limited validation set. They are **not calibrated probabilities** and do not fix errors in the model weights.

## Scope and limitations

The model is a Bistrina-focused **research prototype**, not a reliable marine monitoring product. The old validation set contained only seven Net instances, and earlier external TrashCan evaluation showed very poor domain generalization. Human review of predictions is necessary. Additional independent Net/Rope images from other underwater locations are future work.

## Team responsibilities

- Model owner: provide the original `best.pt` and maintain `inference.py`.
- Application teammate: maintain `streamlit_app.py` and deployment.

Verify model/data weights distribution permissions and applicable Ultralytics licensing before making a public repository or commercial release.
