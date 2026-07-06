"""
ui/pipeline.py
==============
Background-thread pipeline adapter for the OCR-UIF desktop UI.

This module wraps the three existing CLI pipeline scripts
(build_dataset.py, process_all_files.py, generate_final_csv.py)
and exposes simple callables that:
  1. Accept a progress_callback(pct: float, message: str) to drive the UI.
  2. Run entirely in a worker thread (never on the UI thread).
  3. Write one output CSV per PDF to the user-selected output folder.

None of the original script logic is duplicated here; we import and
re-use the existing helper functions directly.
"""

from __future__ import annotations

import csv
import os
import sys

# Limit CPU core usage to at most 70% to avoid freezing other user processes
# ONNX Runtime, OpenMP, and BLAS libraries read these variables during initialization
cores_limit = max(1, int((os.cpu_count() or 2) * 0.70))
os.environ["OMP_NUM_THREADS"] = str(cores_limit)
os.environ["ONNXRUNTIME_NUM_THREADS"] = str(cores_limit)
os.environ["OPENBLAS_NUM_THREADS"] = str(cores_limit)
os.environ["MKL_NUM_THREADS"] = str(cores_limit)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores_limit)
os.environ["NUMEXPR_NUM_THREADS"] = str(cores_limit)
import re
import sys
import threading
from pathlib import Path
from typing import Callable, List, Tuple

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Helpers re-used from existing scripts
# ---------------------------------------------------------------------------

def _get_project_root() -> Path:
    """Return the root of the ocr-uif project (parent of this ui/ package)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def _models_dir() -> Path:
    return _get_project_root() / "models"


def _transcriptions_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "transcriptions"
    return _get_project_root() / "transcriptions"


# ---------------------------------------------------------------------------
# PHASE 1 — OCR
# ---------------------------------------------------------------------------

def run_ocr(
    pdf_folder: str,
    progress_cb: ProgressCallback,
    cancel_event: threading.Event = None,
) -> List[Tuple[str, str]]:
    """
    Scan *pdf_folder* recursively for PDF files, run RapidOCR on page 1,
    and cache the text to the project-level transcriptions/ directory in parallel.

    Returns a list of (pdf_filename, transcription_text) tuples.
    """
    import fitz  # PyMuPDF
    import cv2
    from rapidocr_onnxruntime import RapidOCR
    from concurrent.futures import ThreadPoolExecutor, as_completed

    trans_dir = _transcriptions_dir()
    trans_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDFs
    pdf_files: List[Tuple[str, Path]] = []
    for root, ds, files in os.walk(pdf_folder):
        # Prune output and environment directories
        ds[:] = [d for d in ds if d.lower() not in ["test", "dist", "build", "transcriptions", ".venv", ".git", "old"]]
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append((f, Path(root) / f))

    if not pdf_files:
        progress_cb(1.0, "⚠ No PDF files found in the selected folder.")
        return []

    progress_cb(0.0, f"Initializing RapidOCR — found {len(pdf_files)} PDF(s)…")
    ocr_engine = RapidOCR()

    results_dict = {}
    total = len(pdf_files)
    completed = 0
    lock = threading.Lock()

    def process_single_pdf(filename, path):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Proceso cancelado")
        nonlocal completed
        basename = Path(filename).stem
        cache_path = trans_dir / f"{basename}.txt"

        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
            with lock:
                completed += 1
                progress_cb(
                    completed / total,
                    f"[{completed}/{total}] Cached: {filename}",
                )
            return filename, text

        try:
            doc = fitz.open(str(path))
            page = doc[0]
            pix = page.get_pixmap(dpi=96)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n
            )
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            res, _ = ocr_engine(img)
            text = "\n".join([item[1] for item in res]) if res else ""
            cache_path.write_text(text, encoding="utf-8")
        except Exception as exc:
            text = ""
            with lock:
                completed += 1
                progress_cb(
                    completed / total,
                    f"  ✗ Error on {filename}: {exc}",
                )
            return filename, text

        with lock:
            completed += 1
            progress_cb(
                completed / total,
                f"[{completed}/{total}] OCR: {filename}…",
            )
        return filename, text

    # Set worker count to 70% of available CPU cores (minimum of 1)
    num_workers = max(1, int((os.cpu_count() or 2) * 0.70))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_pdf, fn, p): fn for fn, p in pdf_files}
        for future in as_completed(futures):
            if cancel_event and cancel_event.is_set():
                for f in futures:
                    f.cancel()
                raise RuntimeError("Proceso cancelado")
            fn, text = future.result()
            results_dict[fn] = text

    # Maintain the original order of the PDF files list
    results = [(fn, results_dict[fn]) for fn, _ in pdf_files]

    progress_cb(1.0, f"✓ OCR complete — {len(results)} file(s) processed.")
    return results


# ---------------------------------------------------------------------------
# PHASE 2 — Train / load SVM model
# ---------------------------------------------------------------------------

def run_nlp_and_predict(
    ocr_results: List[Tuple[str, str]],
    progress_cb: ProgressCallback,
) -> Tuple[object, object]:
    """
    Load a pre-trained SVM model from models/ if available,
    or train a fresh one from files_dataset.parquet.

    Returns (vectorizer, classifier).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV

    models_dir = _models_dir()
    vect_path = models_dir / "tfidf_vectorizer.joblib"
    clf_path = models_dir / "svm_classifier_model.joblib"

    if vect_path.exists() and clf_path.exists():
        progress_cb(0.2, "Loading pre-trained SVM model…")
        vectorizer = joblib.load(vect_path)
        clf = joblib.load(clf_path)
        progress_cb(1.0, "✓ Model loaded from disk.")
        return vectorizer, clf

    # Train from scratch
    parquet_path = _get_project_root() / "files_dataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            "files_dataset.parquet not found and no pre-trained model exists.\n"
            "Run build_dataset.py first to generate training data."
        )

    progress_cb(0.1, "Training TF-IDF + SVM model…")
    df_matched = pd.read_parquet(str(parquet_path))

    X_train_text = df_matched["extracted_text"].values
    y_train = df_matched["label_movement"].values

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train_text)

    progress_cb(0.5, "Fitting SVM classifier (this may take a minute)…")
    base_clf = SVC(kernel="linear", class_weight="balanced", random_state=42)
    clf = CalibratedClassifierCV(base_clf, ensemble=False)
    clf.fit(X_train_vec, y_train)

    models_dir.mkdir(exist_ok=True)
    joblib.dump(vectorizer, vect_path)
    joblib.dump(clf, clf_path)

    progress_cb(1.0, "✓ Model trained and saved.")
    return vectorizer, clf


# ---------------------------------------------------------------------------
# PHASE 3 — Extract metadata + write per-PDF CSVs
# ---------------------------------------------------------------------------

def run_output(
    ocr_results: List[Tuple[str, str]],
    vectorizer,
    clf,
    output_folder: str,
    progress_cb: ProgressCallback,
    pdf_folder: str = None,
    cancel_event: threading.Event = None,
) -> List[str]:
    """
    For every (filename, text) in *ocr_results*:
      - Predict movement (ALTA / BAJA).
      - Extract name and office number.
      - Write one .csv file named after the PDF to *output_folder*.

    Returns a list of written CSV file paths.
    """
    # Import helpers from the ui package directly
    from ui.extract_and_validate import (
        extract_office_number,
        extract_name,
        split_full_name,
        is_moral_entity,
        bigram_similarity,
    )

    # Load CSV database for name context matching (same as generate_final_csv.py)
    if getattr(sys, "frozen", False):
        root_dir = Path(sys.executable).parent
    else:
        root_dir = _get_project_root()

    csv_records: list = []
    
    # Collect candidate database directories
    dirs_to_check = [
        root_dir / "files",
        root_dir.parent / "files",
    ]
    if pdf_folder:
        dirs_to_check.append(Path(pdf_folder))
        dirs_to_check.append(Path(pdf_folder).parent)
    
    # Keep track of loaded CSV files to avoid duplicates
    loaded_files = set()
    
    for files_dir in dirs_to_check:
        if files_dir.exists():
            for r, ds, fs in os.walk(str(files_dir)):
                # Prune directory search to avoid loading outputs or temp files
                ds[:] = [d for d in ds if d.lower() not in ["test", "dist", "build", "transcriptions", ".venv", ".git", "old"]]
                for f in fs:
                    if f.lower().endswith(".csv") and not f.startswith("consolidated"):
                        full_path = os.path.join(r, f)
                        if full_path not in loaded_files:
                            try:
                                with open(full_path, encoding="utf-8", errors="ignore") as fh:
                                    reader = csv.DictReader(fh)
                                    for row in reader:
                                        csv_records.append({
                                            "nombre": row.get("NOMBRE", "").strip(),
                                            "paterno": row.get("PATERNO", "").strip(),
                                            "materno": row.get("MATERNO", "").strip(),
                                            "movimiento": row.get("MOVIMIENTO", "").strip(),
                                            "motivo": row.get("MOTIVO", "").strip(),
                                        })
                                loaded_files.add(full_path)
                            except Exception:
                                pass

    def _clean(s: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "", str(s).upper())

    def _segment_name(fullname: str):
        """Match extracted name against CSV database; fall back to rule split."""
        c_ext = _clean(fullname)
        if not c_ext or c_ext == "NA":
            return "", "", "", False

        best_rec = None
        best_sim = 0.0
        for rec in csv_records:
            gt_parts = [rec["nombre"], rec["paterno"], rec["materno"]]
            gt_full = " ".join(p.strip() for p in gt_parts if p)
            sim = bigram_similarity(c_ext, _clean(gt_full))
            if sim > best_sim:
                best_sim = sim
                best_rec = rec

        if best_rec and best_sim >= 0.60:
            is_co = is_moral_entity(fullname) or is_moral_entity(best_rec["nombre"])
            if is_co:
                return best_rec["nombre"].upper(), "", "", True
            return (
                best_rec["nombre"].upper(),
                best_rec["paterno"].upper(),
                best_rec["materno"].upper(),
                False,
            )

        is_co = is_moral_entity(fullname)
        nombre, paterno, materno = split_full_name(fullname, is_co)
        return nombre, paterno, materno, is_co

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    total = len(ocr_results)

    for idx, (filename, text) in enumerate(ocr_results):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Proceso cancelado")
        progress_cb(
            idx / total if total else 1.0,
            f"[{idx+1}/{total}] Generating CSV for {filename}…",
        )

        # Predict
        X_vec = vectorizer.transform([text])
        pred_movement = clf.predict(X_vec)[0]

        # Extract
        extracted_oficio = extract_office_number(text)
        extracted_fullname = extract_name(text)
        nombre, paterno, materno, _ = _segment_name(extracted_fullname)

        # Write per-PDF CSV
        stem = Path(filename).stem
        out_csv = out_dir / f"{stem}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["MOVIMIENTO", "NOMBRE", "PATERNO", "MATERNO", "MOTIVO",
                          "CIUDAD", "PAIS"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                "MOVIMIENTO": pred_movement,
                "NOMBRE": nombre,
                "PATERNO": paterno,
                "MATERNO": materno,
                "MOTIVO": extracted_oficio or "",
                "CIUDAD": "",
                "PAIS": "",
            })

        written.append(str(out_csv))

    progress_cb(1.0, f"✓ Output complete — {len(written)} CSV(s) written to {output_folder}")
    return written


# ---------------------------------------------------------------------------
# Full pipeline runner (convenience for the "Run All" button)
# ---------------------------------------------------------------------------

def run_full_pipeline(
    pdf_folder: str,
    output_folder: str,
    progress_cb: ProgressCallback,
    cancel_event: threading.Event = None,
) -> List[str]:
    """
    Run OCR → NLP/Model → Output in sequence.
    *progress_cb* is called with (0.0–1.0, message) throughout.
    Returns list of written CSV paths.
    """
    def phase_cb(phase_start: float, phase_end: float):
        """Scale a sub-phase 0–1 progress into the global 0–1 range."""
        span = phase_end - phase_start
        def cb(pct: float, msg: str):
            progress_cb(phase_start + pct * span, msg)
        return cb

    # Phase 1: OCR (0% → 40%)
    ocr_results = run_ocr(pdf_folder, phase_cb(0.0, 0.4), cancel_event)

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Proceso cancelado")

    # Phase 2: Model (40% → 65%)
    vectorizer, clf = run_nlp_and_predict(ocr_results, phase_cb(0.4, 0.65))

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Proceso cancelado")

    # Phase 3: Output (65% → 100%)
    written = run_output(ocr_results, vectorizer, clf, output_folder, phase_cb(0.65, 1.0), pdf_folder, cancel_event)

    return written
