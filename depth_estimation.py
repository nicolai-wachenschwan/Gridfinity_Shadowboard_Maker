import os
import requests
import numpy as np
import cv2
import onnxruntime as ort
import streamlit as st

# --- Constants ---
MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx?download=true"
MODEL_PATH = "depth_anything_v2_small.onnx"
INPUT_HEIGHT = 518
INPUT_WIDTH = 518

# --- Model Downloading ---
def download_model_if_needed():
    """Downloads the ONNX model from Hugging Face if it's not already present."""
    if not os.path.exists(MODEL_PATH):
        try:
            st.info(f"Downloading depth estimation model (~55 MB)... This may take a moment.")
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download the depth model: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH) # Clean up partial downloads
            return False
    return True

# --- Inference ---
@st.cache_data(show_spinner=False)
def get_depth_map(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Performs depth estimation on a single image using the Depth Anything V2 ONNX model.

    Args:
        image_bgr: The input image in BGR format (as loaded by OpenCV).

    Returns:
        A grayscale depth map (H, W) normalized to 0-255, or None if an error occurs.
    """
    if not download_model_if_needed():
        return None

    try:
        # --- 1. Preprocessing ---
        original_h, original_w = image_bgr.shape[:2]

        # Convert BGR to RGB and normalize to [0, 1]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0

        # Resize to model's expected input size
        image_resized = cv2.resize(image_rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)

        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_resized - mean) / std

        # Transpose from HWC to CHW format and add batch dimension
        image_tensor = image_normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # --- 2. ONNX Inference ---
        sess_options = ort.SessionOptions()
        session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"])

        # Note: Using IOBinding is more efficient but requires more setup.
        # For simplicity, we use the direct run method here.
        ort_inputs = {session.get_inputs()[0].name: image_tensor}
        ort_outs = session.run(None, ort_inputs)
        depth_prediction = ort_outs[0][0] # Get the first (and only) output, remove batch dim

        # --- 3. Postprocessing ---
        # Normalize the output depth map to be in the range [0, 1]
        min_val = np.min(depth_prediction)
        max_val = np.max(depth_prediction)
        if max_val > min_val:
            depth_normalized = (depth_prediction - min_val) / (max_val - min_val)
        else:
            depth_normalized = np.zeros(depth_prediction.shape, dtype=np.float32)

        # The model tends to output smaller values for closer objects.
        # We want the tool (closer) to be dark (value 0) and background (further) to be light (value 255).
        # The current normalization already achieves this. So, no inversion needed.

        # Resize back to the original image size
        depth_resized = cv2.resize(depth_normalized, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

        # Scale to 0-255 and convert to a single-channel uint8 image
        depth_map_uint8 = (depth_resized * 255.0).astype(np.uint8)

        return depth_map_uint8

    except Exception as e:
        st.error(f"An error occurred during depth estimation: {e}")
        return None
