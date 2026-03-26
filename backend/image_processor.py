"""
image_processor.py
Converts a class map (0-5) → coloured RGB mask → Base64 PNG.
Color values are in BGR order (matching prediction.py which uses OpenCV).
"""

import base64
import numpy as np
import cv2

# ─── Color Map ────────────────────────────────────────────────────────────────
# Keys = class ID, values = BGR tuple (OpenCV native order, as used in prediction.py)
CLASS_COLORS_BGR = {
    0: (255, 255, 255),  # Background / Roads  – White
    1: (255, 0,   0  ),  # Water               – Blue  (BGR: B=255)
    2: (0,   255, 0  ),  # Dense Vegetation    – Green
    3: (0,   255, 255),  # Light Vegetation    – Yellow (BGR: G+R=255)
    4: (255, 0,   255),  # Dry Land            – Magenta (BGR: B+R=255)
    5: (0,   255, 255),  # Urban               – Cyan (BGR: G+B=255)  ← note: same as Yellow BGRwise
}

# Human-readable class names (used in API response)
CLASS_NAMES = {
    0: "Background",
    1: "Water",
    2: "Dense Vegetation",
    3: "Light Vegetation",
    4: "Dry Land",
    5: "Urban",
}

# ─── Color Mask ───────────────────────────────────────────────────────────────

def apply_color_mask(class_map: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) class map to a (H, W, 3) BGR image.
    Matches the loop in prediction.py exactly.
    """
    h, w = class_map.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color_bgr in CLASS_COLORS_BGR.items():
        output[class_map == class_id] = color_bgr

    return output  # BGR, ready for cv2 operations


def process_and_encode_mask(class_map: np.ndarray) -> str:
    """
    Pipeline:
      class_map → BGR colour mask → resize to 800×800 → PNG → Base64 string.
    Returns a data-URI string: "data:image/png;base64,<data>"
    """
    bgr_mask = apply_color_mask(class_map)

    # Resize to 800×800 for high-quality visualisation (INTER_NEAREST keeps hard boundaries)
    enlarged = cv2.resize(bgr_mask, (800, 800), interpolation=cv2.INTER_NEAREST)

    # Encode to PNG bytes then to Base64
    _, buffer = cv2.imencode(".png", enlarged)
    b64 = base64.b64encode(buffer).decode("utf-8")

    return f"data:image/png;base64,{b64}"
