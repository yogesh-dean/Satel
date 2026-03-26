"""
test_api.py  -  Automated endpoint tests for the Segmentation API
Run:   python test_api.py          (server must be running on localhost:8000)
       python test_api.py --url http://localhost:8000   (custom URL)
"""

import sys
import io
import json
import struct
import zlib
import base64
import argparse
import requests

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_URL = "http://localhost:8000"

PASS = "[PASS]"
FAIL = "[FAIL]"

passed = 0
failed = 0


def test(name: str, ok: bool, info: str = ""):
    global passed, failed
    tag = PASS if ok else FAIL
    print(f"  {tag}  {name}" + (f"  →  {info}" if info else ""))
    if ok:
        passed += 1
    else:
        failed += 1


# ─── Tiny valid PNG factory ────────────────────────────────────────────────────

def make_png(width: int = 4, height: int = 4) -> bytes:
    """Create a minimal valid RGB PNG in memory without Pillow."""
    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)

    # IDAT (raw pixel rows, each prefixed with filter byte 0)
    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00"          # filter byte
        raw_rows += b"\x80\x40\x20" * width  # RGB triplets
    compressed = zlib.compress(raw_rows)
    idat = make_chunk(b"IDAT", compressed)

    # IEND
    iend = make_chunk(b"IEND", b"")

    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


SAMPLE_IMAGE = make_png()


# ─── Tests ────────────────────────────────────────────────────────────────────

def run_tests(base_url: str):
    print(f"\n{'='*55}")
    print(f"  Satellite Segmentation API  –  Endpoint Tests")
    print(f"  Target: {base_url}")
    print(f"{'='*55}\n")

    # ── 1. Health Check ─────────────────────────────────────────────
    print("GET /")
    try:
        r = requests.get(f"{base_url}/", timeout=10)
        test("Status 200",   r.status_code == 200)
        data = r.json()
        test("'status' key present",          "status"        in data)
        test("'model_loaded' key present",    "model_loaded"  in data)
        test("'model_mode' key present",      "model_mode"    in data)
    except Exception as e:
        test("Server reachable", False, str(e))

    # ── 2. Class Legend ─────────────────────────────────────────────
    print("\nGET /api/classes")
    try:
        r = requests.get(f"{base_url}/api/classes", timeout=10)
        test("Status 200", r.status_code == 200)
        data = r.json()
        classes = data.get("classes", [])
        test("Returns 6 classes",             len(classes) == 6)
        test("Each entry has 'class_id'",     all("class_id"  in c for c in classes))
        test("Each entry has 'name'",         all("name"      in c for c in classes))
        test("Each entry has 'color_hex'",    all("color_hex" in c for c in classes))
    except Exception as e:
        test("Legend accessible", False, str(e))

    # ── 3. Full Analyze ─────────────────────────────────────────────
    print("\nPOST /api/analyze")
    try:
        r = requests.post(
            f"{base_url}/api/analyze",
            files={"file": ("test_sat.png", SAMPLE_IMAGE, "image/png")},
            timeout=30,
        )
        test("Status 200", r.status_code == 200)
        data = r.json()
        test("'filename' present",                   "filename"                   in data)
        test("'class_percentages' present",          "class_percentages"          in data)
        test("'vulnerability' present",              "vulnerability"              in data)
        test("'segmentation_mask_base64' present",   "segmentation_mask_base64"   in data)

        pct = data.get("class_percentages", {})
        test("6 classes in percentages",             len(pct) == 6)

        total = sum(pct.values())
        test("Percentages sum ≈ 100%",
             99.0 <= total <= 101.0,
             f"sum={total:.2f}")

        vuln = data.get("vulnerability", {})
        # 5-module schema checks
        test("'drought_risk' present",      "drought_risk"     in vuln)
        test("'flood_risk' present",        "flood_risk"       in vuln)
        test("'ecosystem_health' present",  "ecosystem_health" in vuln)
        test("'urban_risk' present",        "urban_risk"       in vuln)
        test("'final_status' present",      "final_status"     in vuln)
        test("'key_metrics' present",       "key_metrics"      in vuln)

        expected_levels  = {"HIGH", "MODERATE", "LOW"}
        expected_eco     = {"HEALTHY", "MODERATE", "POOR"}
        expected_finals  = {
            "HIGH DROUGHT VULNERABILITY",
            "HIGH FLOOD VULNERABILITY",
            "LOW VULNERABILITY (STABLE ECOSYSTEM)",
            "URBAN ENVIRONMENTAL STRESS",
            "MODERATE VULNERABILITY",
        }
        test("drought_risk valid level",    vuln.get("drought_risk")     in expected_levels,  vuln.get("drought_risk"))
        test("flood_risk valid level",      vuln.get("flood_risk")       in expected_levels,  vuln.get("flood_risk"))
        test("ecosystem_health valid",      vuln.get("ecosystem_health") in expected_eco,     vuln.get("ecosystem_health"))
        test("urban_risk valid level",      vuln.get("urban_risk")       in expected_levels,  vuln.get("urban_risk"))
        test("final_status is valid",       vuln.get("final_status")     in expected_finals,  vuln.get("final_status"))

        mask_b64 = data.get("segmentation_mask_base64", "")
        test("Mask starts with data URI prefix",
             mask_b64.startswith("data:image/png;base64,"))
        raw = mask_b64.split(",", 1)[-1]
        decoded = base64.b64decode(raw)
        test("Mask Base64 decodes to non-empty PNG",
             decoded[:4] == b"\x89PNG",
             f"{len(decoded)} bytes")
    except Exception as e:
        test("Analyze endpoint worked", False, str(e))

    # ── 4. Segment Only ─────────────────────────────────────────────
    print("\nPOST /api/segment-only")
    try:
        r = requests.post(
            f"{base_url}/api/segment-only",
            files={"file": ("test_sat.png", SAMPLE_IMAGE, "image/png")},
            timeout=30,
        )
        test("Status 200", r.status_code == 200)
        data = r.json()
        test("'segmentation_mask_base64' present",   "segmentation_mask_base64" in data)
        test("'class_percentages' NOT present",      "class_percentages"        not in data)
    except Exception as e:
        test("Segment-only endpoint worked", False, str(e))

    # ── 5. Vulnerability Only ────────────────────────────────────────
    print("\nPOST /api/vulnerability")
    try:
        r = requests.post(
            f"{base_url}/api/vulnerability",
            files={"file": ("test_sat.png", SAMPLE_IMAGE, "image/png")},
            timeout=30,
        )
        test("Status 200", r.status_code == 200)
        data = r.json()
        test("'class_percentages' present",        "class_percentages"        in data)
        test("'vulnerability' present",            "vulnerability"            in data)
        test("'segmentation_mask_base64' NOT present",
             "segmentation_mask_base64" not in data)
    except Exception as e:
        test("Vulnerability-only endpoint worked", False, str(e))

    # ── 6. Invalid File Type (415) ───────────────────────────────────
    print("\nPOST /api/analyze  (invalid file type)")
    try:
        r = requests.post(
            f"{base_url}/api/analyze",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            timeout=10,
        )
        test("Returns 415 for non-image upload", r.status_code == 415)
    except Exception as e:
        test("Validation error handled", False, str(e))

    # ── Summary ─────────────────────────────────────────────────────
    total_tests = passed + failed
    print(f"\n{'='*55}")
    print(f"  Results: {passed}/{total_tests} passed  |  {failed} failed")
    print(f"{'='*55}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Segmentation API")
    parser.add_argument("--url", default=BASE_URL, help="Base URL of the API server")
    args = parser.parse_args()
    run_tests(args.url)
