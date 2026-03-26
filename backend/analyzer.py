"""
analyzer.py
Full vulnerability analysis engine based on segmented class percentages.
Covers: Drought, Flood, Ecosystem Health, Urban Stress, and Combined Final Risk.
"""

from typing import Dict, Any
import numpy as np
from image_processor import CLASS_NAMES

NUM_CLASSES = 6


def calculate_percentages(class_map: np.ndarray) -> Dict[str, float]:
    """Return {class_name: percentage} for all 6 classes, rounded to 2 dp."""
    total_pixels = class_map.size
    percentages: Dict[str, float] = {}
    for class_id, name in CLASS_NAMES.items():
        count = int(np.sum(class_map == class_id))
        percentages[name] = round((count / total_pixels) * 100, 2)
    return percentages


def determine_vulnerability(percentages: Dict[str, float]) -> Dict[str, Any]:
    """
    Apply the full 5-module vulnerability rule engine.
    Returns individual risk levels and one combined final status.
    """
    water      = percentages.get("Water", 0.0)
    dense_veg  = percentages.get("Dense Vegetation", 0.0)
    light_veg  = percentages.get("Light Vegetation", 0.0)
    dry        = percentages.get("Dry Land", 0.0)
    urban      = percentages.get("Urban", 0.0)
    veg_total  = dense_veg + light_veg

    # ── 1. Drought Risk ───────────────────────────────────────────────────────
    if dry > 50 and veg_total < 30:
        drought_risk = "HIGH"
        drought_detail = f"High risk: Dry land covers {dry:.1f}% of the area while vegetation is critically low ({veg_total:.1f}%). This indicates severe soil moisture deficit and potential for complete crop failure or desertification."
    elif dry > 30 and veg_total < 50:
        drought_risk = "MODERATE"
        drought_detail = f"Moderate risk: Elevated dry land presence ({dry:.1f}%) combined with reduced green cover ({veg_total:.1f}%). Requires careful water management and monitoring of soil health."
    else:
        drought_risk = "LOW"
        drought_detail = f"Low risk: Dry land ({dry:.1f}%) is within safe ecological limits, and current vegetation levels suggest adequate ground moisture retention."

    # ── 2. Flood Risk ─────────────────────────────────────────────────────────
    if water > 40 and veg_total < 40:
        flood_risk = "HIGH"
        flood_detail = f"Critical risk: Extensive water bodies ({water:.1f}%) paired with low natural absorption from vegetation ({veg_total:.1f}%). High probability of runoff and overflow in low-lying areas."
    elif water > 25:
        flood_risk = "MODERATE"
        flood_detail = f"Moderate risk: Water coverage is elevated at {water:.1f}%. Natural drainage capacity should be assessed, especially during heavy precipitation events."
    else:
        flood_risk = "LOW"
        flood_detail = f"Low risk: Water coverage ({water:.1f}%) is stable. Natural landscapes appear capable of handling standard drainage through existing soil and vegetation."

    # ── 3. Ecosystem Health ───────────────────────────────────────────────────
    if veg_total > 70:
        ecosystem = "HEALTHY"
        ecosystem_detail = f"Excellent: Robust vegetation cover ({veg_total:.1f}%) indicates a thriving, biodiverse environment with high carbon sequestration and stable local climate regulation."
    elif veg_total > 40:
        ecosystem = "MODERATE"
        ecosystem_detail = f"Stable: Moderate vegetation cover ({veg_total:.1f}%). The ecosystem is functioning but may benefit from reforestation or conservation efforts to improve resilience."
    else:
        ecosystem = "POOR"
        ecosystem_detail = f"Critical: Very low green cover ({veg_total:.1f}%). The ecosystem is highly fragmented and susceptible to erosion, heat island effects, and loss of local biodiversity."

    # ── 4. Urban Stress ───────────────────────────────────────────────────────
    if urban > 30:
        urban_risk = "HIGH"
        urban_detail = f"High stress: Artificial surfaces cover {urban:.1f}% of the land. High potential for urban heat islands, reduced soil permeability, and significant environmental pressure from human activity."
    elif urban > 15:
        urban_risk = "MODERATE"
        urban_detail = f"Moderate stress: Urban expansion ({urban:.1f}%) is noticeable. Urban planning should focus on integrating green corridors and permeable surfaces to mitigate stress."
    else:
        urban_risk = "LOW"
        urban_detail = f"Minimal stress: Low urban footprint ({urban:.1f}%). The natural landscape remains dominant, with very low pressure from built environments."

    # ── 5. Combined Final Vulnerability ──────────────────────────────────────
    if drought_risk == "HIGH":
        final_status = "HIGH DROUGHT VULNERABILITY"
    elif flood_risk == "HIGH":
        final_status = "HIGH FLOOD VULNERABILITY"
    elif ecosystem == "HEALTHY":
        final_status = "LOW VULNERABILITY (STABLE ECOSYSTEM)"
    elif urban_risk == "HIGH":
        final_status = "URBAN ENVIRONMENTAL STRESS"
    else:
        final_status = "MODERATE VULNERABILITY"

    return {
        "drought_risk":     drought_risk,
        "drought_detail":   drought_detail,
        "flood_risk":       flood_risk,
        "flood_detail":     flood_detail,
        "ecosystem_health": ecosystem,
        "ecosystem_detail": ecosystem_detail,
        "urban_risk":       urban_risk,
        "urban_detail":     urban_detail,
        "final_status":     final_status,
        "key_metrics": {
            "water_pct":    round(water, 2),
            "veg_total_pct": round(veg_total, 2),
            "dry_land_pct": round(dry, 2),
            "urban_pct":    round(urban, 2),
        },
    }


def analyze_vulnerability(class_map: np.ndarray) -> Dict[str, Any]:
    """Convenience wrapper used by main.py."""
    percentages = calculate_percentages(class_map)
    vulnerability = determine_vulnerability(percentages)
    return {
        "class_percentages": percentages,
        "vulnerability": vulnerability,
    }
