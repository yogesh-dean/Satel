"""
main.py  –  FastAPI backend for Satellite Image Semantic Segmentation
Endpoints:
  GET  /                     Health check
  GET  /api/classes          Class legend (names + colours)
  POST /api/analyze          Full segmentation + vulnerability analysis
  POST /api/segment-only     Segmentation mask only (no analysis)
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_handler import predict, get_model
from image_processor import process_and_encode_mask, CLASS_NAMES, CLASS_COLORS_BGR
from analyzer import analyze_vulnerability

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Satellite Image Segmentation API",
    description=(
        "Backend API for U-Net-based semantic segmentation of satellite images. "
        "Classifies land into 6 categories and evaluates environmental vulnerability."
    ),
    version="1.0.0",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Allowed file types ───────────────────────────────────────────────────────

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


def _read_and_validate(file: UploadFile) -> bytes:
    """Read file bytes and validate content-type."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Upload a JPG or PNG image.",
        )
    return file.file.read()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Confirm the server is running and the model status."""
    model_loaded = get_model() is not None
    return {
        "status": "running",
        "model_loaded": model_loaded,
        "model_mode": "real" if model_loaded else "mock (random predictions)",
        "message": "Satellite Image Segmentation API is active.",
    }


@app.get("/api/classes", tags=["Info"])
def get_class_legend():
    """
    Return the class legend: class ID, name, and BGR colour used in the output mask.
    """
    legend = []
    for class_id, name in CLASS_NAMES.items():
        b, g, r = CLASS_COLORS_BGR[class_id]
        legend.append({
            "class_id": class_id,
            "name": name,
            "color_bgr": list(CLASS_COLORS_BGR[class_id]),
            "color_rgb": [r, g, b],          # RGB for frontend convenience
            "color_hex": f"#{r:02X}{g:02X}{b:02X}",
        })
    return {"classes": legend}


@app.post("/api/analyze", tags=["Segmentation"])
async def analyze_image(file: UploadFile = File(...)):
    """
    Full pipeline endpoint:
      1. Preprocess satellite image
      2. Run U-Net inference → class map (0-5)
      3. Colour the class map → 800x800 Base64 PNG mask
      4. Calculate per-class pixel percentages
      5. Apply 5-module vulnerability rule engine:
           - Drought Risk  (dry land % vs veg %)
           - Flood Risk    (water % vs veg %)
           - Ecosystem Health (total veg %)
           - Urban Stress  (urban %)
           - Final Combined Status

    Returns JSON with segmentation mask (base64), percentages, and full vulnerability breakdown.
    """
    image_bytes = _read_and_validate(file)

    try:
        class_map = predict(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mask_b64       = process_and_encode_mask(class_map)
    analysis       = analyze_vulnerability(class_map)

    return {
        "filename": file.filename,
        "image_size_processed": "128×128",
        "output_mask_size": "800×800",
        "class_percentages": analysis["class_percentages"],
        "vulnerability": analysis["vulnerability"],
        "segmentation_mask_base64": mask_b64,
    }


@app.post("/api/segment-only", tags=["Segmentation"])
async def segment_only(file: UploadFile = File(...)):
    """
    Lightweight endpoint — returns only the coloured segmentation mask as Base64.
    Use this when you only need the visual output without vulnerability analysis.
    """
    image_bytes = _read_and_validate(file)

    try:
        class_map = predict(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mask_b64 = process_and_encode_mask(class_map)

    return {
        "filename": file.filename,
        "segmentation_mask_base64": mask_b64,
    }


@app.post("/api/vulnerability", tags=["Analysis"])
async def vulnerability_only(file: UploadFile = File(...)):
    """
    Returns only the land-cover percentages and vulnerability status (no mask image).
    Useful for lightweight risk analysis queries.
    """
    image_bytes = _read_and_validate(file)

    try:
        class_map = predict(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    analysis = analyze_vulnerability(class_map)

    return {
        "filename": file.filename,
        "class_percentages": analysis["class_percentages"],
        "vulnerability": analysis["vulnerability"],
    }
