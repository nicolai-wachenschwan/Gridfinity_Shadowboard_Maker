import os
import requests
import numpy as np
import cv2
import onnxruntime as ort
import streamlit as st

# --- Constants ---
MODELS = {
    "Small V2": {
        "url": "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx?download=true",
        "path": "depth_anything_v2_small.onnx",
        "size_mb": 55
    },
    "Base V2": {
        "url": "https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/main/onnx/model.onnx?download=true",
        "path": "depth_anything_v2_base.onnx",
        "size_mb": 371 # Approximate size
    }
}
INPUT_HEIGHT = 518
INPUT_WIDTH = 518

# --- Model Downloading ---
def download_model_if_needed(model_name: str):
    """Downloads the selected ONNX model from Hugging Face if it's not already present."""
    if model_name not in MODELS:
        st.error(f"Unknown model: {model_name}")
        return False

    model_info = MODELS[model_name]
    model_path = model_info["path"]
    model_url = model_info["url"]
    model_size_mb = model_info["size_mb"]

    if not os.path.exists(model_path):
        try:
            st.info(f"Downloading '{model_name}' model (~{model_size_mb} MB)... This may take a moment.")
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success(f"'{model_name}' model downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download the model '{model_name}': {e}")
            if os.path.exists(model_path):
                os.remove(model_path) # Clean up partial downloads
            return False
    return True

# --- Inference ---
@st.cache_data(show_spinner=False)
def get_depth_map(image_bgr: np.ndarray, model_name: str) -> np.ndarray | None:
    """
    Performs depth estimation on a single image using the specified Depth Anything V2 ONNX model.

    Args:
        image_bgr: The input image in BGR format (as loaded by OpenCV).
        model_name: The name of the model to use (e.g., "Small V2", "Base V2").

    Returns:
        A grayscale depth map (H, W) normalized to 0-255, or None if an error occurs.
    """
    if model_name not in MODELS:
        st.error(f"Unknown model for depth estimation: {model_name}")
        return None

    if not download_model_if_needed(model_name):
        return None

    model_path = MODELS[model_name]["path"]

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
        # Use st.cache_resource to avoid reloading the model on every run
        @st.cache_resource
        def get_ort_session(path):
            return ort.InferenceSession(path, sess_options=sess_options, providers=["CPUExecutionProvider"])

        session = get_ort_session(model_path)

        ort_inputs = {session.get_inputs()[0].name: image_tensor}
        ort_outs = session.run(None, ort_inputs)
        depth_prediction = ort_outs[0][0]

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
