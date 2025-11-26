
import sys
import numpy as np
import cv2
from PIL import Image

# Conditional import for Windows
if sys.platform == "win32":
    try:
        from transformers import pipeline
        import torch
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
else:
    TRANSFORMERS_AVAILABLE = False

# Global variable to hold the pipeline, to avoid re-initializing
SAM_PIPELINE = None

def get_paper_properties(paper_coords):
    """Calculates the bounding box and area of the paper."""
    if paper_coords is None or len(paper_coords) < 4:
        return (0, 0, 0, 0), 0
    rect = cv2.minAreaRect(paper_coords)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x, y, w, h = cv2.boundingRect(box)
    area = cv2.contourArea(box)
    return (x, y, w, h), area

def masks_overlap(box1, box2):
    """Checks if two bounding boxes (x, y, w, h) overlap."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def generate_sam2_masks(image: np.ndarray, paper_coords: np.ndarray) -> np.ndarray | None:
    """
    Generates, filters, and combines masks using SAM 2.1.
    This function is intended for Windows only.
    """
    global SAM_PIPELINE
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers library not found. Skipping SAM 2.1 processing.")
        return None

    if sys.platform != "win32":
        print("SAM 2.1 processing is only available on Windows.")
        return None

    try:
        # Initialize the pipeline only if it hasn't been already
        if SAM_PIPELINE is None:
            print("Initializing SAM 2.1 pipeline for the first time...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            SAM_PIPELINE = pipeline("mask-generation", model="facebook/sam2.1-hiera-base-plus", device=device)
            print(f"Pipeline initialized on '{device}'.")

        pil_image = Image.fromarray(image)

        print("Generating masks...")
        outputs = SAM_PIPELINE(pil_image, points_per_batch=64)
        print(f"{len(outputs['masks'])} masks generated.")

        paper_bbox, paper_area = get_paper_properties(paper_coords)
        if paper_area == 0:
            print("Warning: Paper area is 0, skipping paper-based filtering.")
            return None

        h_img, w_img = image.shape[:2]
        final_mask = np.full((h_img, w_img), 255, dtype=np.uint8)

        for mask_data in outputs['masks']:
            mask = np.array(mask_data) * 255

            coords = cv2.findNonZero(mask)
            if coords is None:
                continue

            tool_bbox = cv2.boundingRect(coords)
            x, y, w, h = tool_bbox
            mask_area = np.sum(mask > 0)

            # --- Filtering Criteria ---
            is_smaller_than_paper = mask_area < (paper_area * 0.95)
            is_overlapping_paper = masks_overlap(tool_bbox, paper_bbox)

            # New: Check if the mask touches the image border
            touches_border = (x == 0) or (y == 0) or ((x + w) == w_img) or ((y + h) == h_img)

            if is_smaller_than_paper and is_overlapping_paper and not touches_border:
                final_mask[mask > 0] = 0

        print("Masks filtered and combined.")
        return final_mask

    except Exception as e:
        print(f"An error occurred during SAM 2.1 processing: {e}")
        return None
