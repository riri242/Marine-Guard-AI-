"""Streamlit front end for MarineGuard AI (no API server required)."""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import load_model, png_bytes, predict

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "best.pt"

st.set_page_config(page_title="MarineGuard AI", page_icon="🌊", layout="wide")
st.title("🌊 MarineGuard AI")
st.caption("Bistrina-focused prototype · Underwater Net and Rope detection")
st.write("Upload an underwater image to visualize candidate fishing net and rope detections.")


@st.cache_resource(show_spinner="Loading the MarineGuard model...")
def model_resource():
    # Cached across Streamlit reruns; a lock prevents concurrent inference calls
    # on the shared Ultralytics object.
    return load_model(MODEL_FILE), Lock()


uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        with Image.open(uploaded) as im:
            original = im.copy()

        model, inference_lock = model_resource()
        with st.spinner("Detecting Nets and Ropes..."):
            with inference_lock:
                annotated, detections = predict(original, model)

        left, right = st.columns(2)
        with left:
            st.subheader("Input")
            st.image(original, use_container_width=True)
        with right:
            st.subheader("Prediction")
            st.image(annotated, use_container_width=True)

        if detections:
            st.success(f"{len(detections)} candidate detection(s)")
            st.dataframe(detections, use_container_width=True, hide_index=True)
        else:
            st.info("No Net or Rope was detected at the current thresholds. This does not prove the image is free of debris.")

        st.download_button(
            "Download annotated image",
            data=png_bytes(annotated),
            file_name="marineguard_detection.png",
            mime="image/png",
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"Unable to read image: {exc}")
    except Exception as exc:
        st.error(f"Inference failed: {exc}")

st.caption(
    "Research prototype for imagery similar to the Bistrina data. "
    "Predictions can be incorrect or incomplete and require human verification."
)
