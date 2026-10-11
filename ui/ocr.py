"""
ui/ocr.py
=========
Fase 1 del pipeline: OCR de la primera página de cada PDF con RapidOCR y el
modelo en_PP-OCRv4, con caché en disco por contenido del archivo.
"""

from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import ui.cpu_limits  # noqa: F401  (antes de numpy/onnxruntime)
import numpy as np

from ui.paths import models_dir, transcriptions_dir

ProgressCallback = Callable[[float, str], None]

# Cambiar esta etiqueta invalida la caché (p. ej. al cambiar modelo, DPI o páginas).
OCR_CACHE_TAG = "en_PP-OCRv4|dpi96|page1|v1"
RENDER_DPI = 96

# Carpetas que nunca contienen oficios a procesar
SKIPPED_DIRS = {"test", "dist", "build", "transcriptions", ".venv", ".git", "old"}

REC_MODEL = "en_PP-OCRv4_rec_infer.onnx"
REC_DICT = "en_dict.txt"


class Cancelled(RuntimeError):
    """El usuario pulsó Cancelar."""

    def __init__(self):
        super().__init__("Proceso cancelado")


@dataclass
class OcrResult:
    filename: str
    path: Path
    text: str = ""
    error: Optional[str] = None
    cached: bool = False


def find_pdfs(folder: str) -> List[Tuple[str, Path]]:
    """PDFs de *folder* y sus subcarpetas, en orden estable."""
    found: List[Tuple[str, Path]] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIPPED_DIRS)
        for f in sorted(files):
            if f.lower().endswith(".pdf"):
                found.append((f, Path(root) / f))
    return found


def cache_key(path: Path) -> str:
    """Huella del contenido del PDF + configuración de OCR."""
    h = hashlib.sha256(OCR_CACHE_TAG.encode("utf-8"))
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def _patch_rapidocr() -> None:
    """rapidocr_onnxruntime 1.2.x no pasa rec_keys_path al reconocedor: se
    quita el prefijo "rec_" de los parámetros. Depende de la versión fijada en
    requirements.txt; solo se aplica una vez por proceso."""
    from rapidocr_onnxruntime.utils import UpdateParameters

    original = UpdateParameters.update_rec_params
    if getattr(original, "_ocr_uif_patched", False):
        return

    def update_rec_params(self, config, rec_dict):
        if rec_dict:
            for k in list(rec_dict):
                if k.startswith("rec_"):
                    rec_dict[k.replace("rec_", "", 1)] = rec_dict[k]
        return original(self, config, rec_dict)

    update_rec_params._ocr_uif_patched = True
    UpdateParameters.update_rec_params = update_rec_params


def model_paths() -> Tuple[Path, Path]:
    """Rutas del modelo de reconocimiento y su diccionario. Falla si faltan, en
    vez de caer en silencio al modelo chino por defecto (resultados distintos)."""
    rec_model, rec_dict = models_dir() / REC_MODEL, models_dir() / REC_DICT
    if not (rec_model.exists() and rec_dict.exists()):
        raise FileNotFoundError(f"No se encontró el modelo OCR en {models_dir()}")
    return rec_model, rec_dict


def create_engine():
    """Motor RapidOCR con el modelo en_PP-OCRv4."""
    rec_model, rec_dict = model_paths()
    _patch_rapidocr()
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(rec_model_path=str(rec_model), rec_keys_path=str(rec_dict))


class _LazyEngine:
    """Crea el motor OCR solo cuando hace falta: si todos los PDFs ya están en
    caché, no se cargan los modelos ONNX (ahorra un par de segundos)."""

    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()

    def __call__(self, image):
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = create_engine()
        return self._engine(image)


def render_first_page(path: Path) -> np.ndarray:
    """Primera página del PDF como imagen RGB."""
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24.3
    except ImportError:
        import fitz

    with fitz.open(str(path)) as doc:
        pix = doc[0].get_pixmap(dpi=RENDER_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        return np.ascontiguousarray(img[:, :, :3])
    if pix.n == 1:
        return np.repeat(img, 3, axis=2)
    return img


def recognize(engine, image: np.ndarray) -> str:
    result, _ = engine(image)
    return "\n".join(item[1] for item in result) if result else ""


def ocr_pdf(engine, filename: str, path: Path, cache_dir: Path) -> OcrResult:
    """OCR de un PDF, usando la caché si ya se leyó antes. Nunca lanza excepción:
    los fallos vuelven en OcrResult.error."""
    try:
        cache_path = cache_dir / f"{cache_key(path)}.txt"
        if cache_path.exists():
            return OcrResult(filename, path, cache_path.read_text(encoding="utf-8"), cached=True)
        text = recognize(engine, render_first_page(path))
        if not text.strip():
            raise ValueError("el OCR no encontró texto en la primera página")
        cache_path.write_text(text, encoding="utf-8")
        return OcrResult(filename, path, text)
    except Exception as exc:  # cualquier fallo de un archivo no detiene el lote
        return OcrResult(filename, path, error=str(exc) or type(exc).__name__)


def run_ocr(
    pdf_folder: str,
    progress_cb: ProgressCallback,
    cancel_event: Optional[threading.Event] = None,
    engine=None,
    workers: Optional[int] = None,
) -> List[OcrResult]:
    """OCR en paralelo de todos los PDFs de *pdf_folder*, en el orden de find_pdfs().
    *workers* son los PDFs leídos a la vez (por defecto, todos los disponibles)."""
    pdf_files = find_pdfs(pdf_folder)
    if not pdf_files:
        progress_cb(1.0, "⚠ No PDF files found in the selected folder.")
        return []

    progress_cb(0.0, f"Initializing RapidOCR — found {len(pdf_files)} PDF(s)…")
    cache_dir = transcriptions_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Un solo motor compartido: probado más rápido que un motor por hilo
    # (cada sesión de ONNX Runtime abre su propio grupo de hilos).
    if engine is None:
        model_paths()  # avisa de inmediato si falta el modelo
        engine = _LazyEngine()

    total = len(pdf_files)
    results: List[Optional[OcrResult]] = [None] * total
    workers = ui.cpu_limits.clamp_cores(workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(ocr_pdf, engine, name, path, cache_dir): i
            for i, (name, path) in enumerate(pdf_files)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            if cancel_event is not None and cancel_event.is_set():
                for f in futures:
                    f.cancel()
                raise Cancelled()
            res = future.result()
            results[futures[future]] = res
            if res.error:
                msg = f"  ✗ Error on {res.filename}: {res.error}"
            else:
                msg = f"[{done}/{total}] {'Cached' if res.cached else 'OCR'}: {res.filename}"
            progress_cb(done / total, msg)

    progress_cb(1.0, f"[OK] OCR complete - {total} file(s) processed.")
    return results
