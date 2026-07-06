# -*- coding: utf-8 -*-
"""
audit_bundle.py
===============
Validates the PyInstaller bundle has ALL required modules before shipping.
Run after every build: python audit_bundle.py
"""
import sys
import subprocess
from pathlib import Path

BUNDLE = Path("dist/OCR-UIF/_internal")

REQUIRED_MODULES = [
    # Core pipeline
    "flet", "fitz", "cv2", "numpy", "pandas", "joblib",
    "sklearn", "onnxruntime",
    # RapidOCR runtime deps (dynamically imported, often missed)
    "rapidocr_onnxruntime", "pyclipper", "shapely", "PIL", "six", "yaml",
]

REQUIRED_DATA_FILES = [
    "rapidocr_onnxruntime/config.yaml",
    "rapidocr_onnxruntime/models/ch_PP-OCRv3_det_infer.onnx",
    "rapidocr_onnxruntime/models/ch_PP-OCRv3_rec_infer.onnx",
    "rapidocr_onnxruntime/models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "models/tfidf_vectorizer.joblib",
    "models/svm_classifier_model.joblib",
    "ui/assets/disclaimer.md",
    "flet_desktop/app/flet-windows.zip",
]

def check_bundle():
    if not BUNDLE.exists():
        print(f"[FAIL] Bundle not found: {BUNDLE}")
        return False

    all_ok = True

    print("=" * 55)
    print("  OCR-UIF Bundle Audit")
    print("=" * 55)

    # Check required data files
    print("\n--- Data Files ---")
    for rel in REQUIRED_DATA_FILES:
        p = BUNDLE / rel
        if p.exists():
            print(f"  [OK]   {rel}")
        else:
            print(f"  [MISS] {rel}  <- MISSING!")
            all_ok = False

    # Check module presence in _internal
    print("\n--- Python Modules ---")
    for mod in REQUIRED_MODULES:
        # Check as folder or .pyd file
        candidates = list(BUNDLE.glob(f"{mod}*")) + list(BUNDLE.glob(f"_{mod}*"))
        if candidates:
            print(f"  [OK]   {mod}")
        else:
            print(f"  [MISS] {mod}  <- MISSING!")
            all_ok = False

    print("\n" + "=" * 55)
    if all_ok:
        print("  PASS: ALL CHECKS PASSED - bundle is complete")
    else:
        print("  FAIL: BUNDLE HAS MISSING FILES - rebuild required")
    print("=" * 55)
    return all_ok


if __name__ == "__main__":
    ok = check_bundle()
    sys.exit(0 if ok else 1)
