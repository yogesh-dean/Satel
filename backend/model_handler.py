"""
model_handler.py
Loads the trained U-Net model and handles image preprocessing + prediction.
Normalization and color logic matches train.py and prediction.py exactly.
"""

import numpy as np
import cv2
import os
import warnings

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 128
NUM_CLASSES = 6

# Path to the saved Keras model (relative to backend/)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "multiclass_unet.keras")

# ─── Model Loader ─────────────────────────────────────────────────────────────
_model = None  # lazy-load singleton


def get_model():
    """Load and cache the Keras model from disk (lazy singleton)."""
    global _model
    if _model is not None:
        return _model

    try:
        import tensorflow as tf  # noqa: F401 – imported here to keep startup fast
        abs_path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model not found at: {abs_path}")
        _model = tf.keras.models.load_model(abs_path, compile=False)
        print(f"[model_handler] OK  Model loaded from {abs_path}")
    except Exception as e:
        print(f"[model_handler] WARN  Could not load model: {e}")
        print("[model_handler]       Running in MOCK mode (random predictions).")
        _model = None

    return _model


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes → resized (128×128), normalised float32 array.
    Normalisation: (img - mean) / (std + 1e-7)  — identical to train.py.
    Returns shape (1, 128, 128, 3) ready for model.predict().
    """
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)

    # cv2 reads in BGR – keep BGR so the channel order matches what the model
    # was trained on (train.py / prediction.py never convert to RGB).
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode the uploaded image. Make sure it is a valid JPG/PNG.")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = (img - np.mean(img)) / (np.std(img) + 1e-7)

    return np.expand_dims(img, axis=0)  # → (1, 128, 128, 3)


# ─── Inference ────────────────────────────────────────────────────────────────

def predict(image_bytes: bytes) -> np.ndarray:
    """
    Run inference on raw image bytes.
    Returns a (128, 128) uint8 class map with values 0-5.
    Falls back to a random map when the model is unavailable.
    """
    model = get_model()

    if model is None:
        # Mock output – useful during development / endpoint testing
        rng = np.random.default_rng(seed=42)
        return rng.integers(0, NUM_CLASSES, size=(IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    img_input = preprocess(image_bytes)
    pred = model.predict(img_input)[0]          # → (128, 128, 6)
    class_map = np.argmax(pred, axis=-1)        # → (128, 128)
    return class_map.astype(np.uint8)
