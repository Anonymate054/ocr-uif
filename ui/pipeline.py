"""
ui/pipeline.py
==============
Pipeline completo de OCR-UIF, pensado para correr en un hilo de trabajo
(nunca en el hilo de la interfaz):

  1. OCR de la primera página de cada PDF        → ui/ocr.py
  2. Carga (o entrenamiento) del clasificador     → run_nlp_and_predict()
  3. Extracción de oficio y nombre + un CSV por PDF → run_output()

Todas las fases reciben progress_cb(pct: float, message: str) para la UI.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

import ui.cpu_limits  # noqa: F401  (antes de numpy/onnxruntime)
import joblib

from ui.ocr import Cancelled, OcrResult, run_ocr
from ui.paths import models_dir, project_root

__all__ = [
    "Cancelled", "OcrResult", "PipelineResult", "run_ocr", "run_nlp_and_predict", "train_classifier",
    "build_record", "classify", "run_output", "run_full_pipeline", "format_summary",
]

ProgressCallback = Callable[[float, str], None]

# Codificación de los CSV de salida: la misma que usan los CSV de referencia
# (Windows-1252), para que Excel muestre bien la Ñ.
OUTPUT_ENCODING = "cp1252"
FIELDNAMES = ["MOVIMIENTO", "NOMBRE", "PATERNO", "MATERNO", "MOTIVO", "CIUDAD", "PAIS"]
ERROR_REPORT = "_errores.csv"
# CSV escritos que conviene revisar a mano (sin nombre o sin número de oficio)
REVIEW_REPORT = "_revisar.csv"


MOVEMENTS = ("ALTA", "BAJA")

# Por debajo de esta probabilidad, el ALTA/BAJA se marca para revisión manual.
# Con los 116 oficios de prueba la mínima es 0.63 (un caso dudoso real) y el
# resto queda arriba de 0.98.
MIN_CONFIDENCE = 0.90
# Términos del vocabulario TF-IDF que debe tener un oficio. Los 116 de prueba
# tienen 348 o más; un texto que no es oficio (una factura) tiene 2 o 3, y aun
# así el modelo le daría ALTA o BAJA con 0.99 de confianza.
MIN_KNOWN_TERMS = 100


class PipelineResult(list):
    """Lista de CSV escritos, con los archivos que fallaron en `.errors`, los
    CSV que hay que revisar a mano en `.review` y cuántos CSV salieron de cada
    movimiento en `.summary` ({"ALTA": n, "BAJA": m})."""

    def __init__(self, written=(), errors=None, summary=None, review=None):
        super().__init__(written)
        self.errors: List[Tuple[str, str]] = errors or []
        self.review: List[Tuple[str, str]] = review or []
        # Reporte para la persona que usa la app (_revisar.csv), si se escribió
        self.report: Optional[Path] = None
        self.summary: Dict[str, int] = {m: 0 for m in MOVEMENTS}
        self.summary.update(summary or {})

    @property
    def needs_review(self) -> int:
        """Archivos que alguien debe revisar a mano: CSV dudosos o incompletos
        y PDFs que no se pudieron leer."""
        return len(self.review) + len(self.errors)


def format_summary(result: PipelineResult) -> str:
    """Resumen corto para la UI: "12 clasificados · 8 altas · 4 bajas"."""
    s = result.summary
    total = sum(s.values())
    parts = [f"{total} clasificado{'' if total == 1 else 's'}"]
    for movement in MOVEMENTS:
        n = s.get(movement, 0)
        parts.append(f"{n} {movement.lower()}{'' if n == 1 else 's'}")
    # Etiquetas que no son ALTA/BAJA (un modelo reentrenado), por si acaso
    parts += [f"{n} {m.lower()}" for m, n in s.items() if m not in MOVEMENTS and n]
    return " · ".join(parts)


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()


# ---------------------------------------------------------------------------
# FASE 2 — Clasificador TF-IDF + SVM (ALTA / BAJA)
# ---------------------------------------------------------------------------

def train_classifier(texts, labels) -> Tuple[object, object]:
    """TF-IDF (palabras y pares de palabras) + SVM lineal calibrada, la
    configuración del modelo publicado en models/. Devuelve (vectorizer, classifier)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    base_clf = SVC(kernel="linear", class_weight="balanced", random_state=42)
    clf = CalibratedClassifierCV(base_clf, ensemble=False)
    clf.fit(vectorizer.fit_transform(texts), labels)
    return vectorizer, clf


def run_nlp_and_predict(ocr_results, progress_cb: ProgressCallback) -> Tuple[object, object]:
    """
    Carga el modelo SVM preentrenado de models/ o, si no existe, lo entrena
    con files_dataset.parquet. Devuelve (vectorizer, classifier).
    """
    vect_path = models_dir() / "tfidf_vectorizer.joblib"
    clf_path = models_dir() / "svm_classifier_model.joblib"

    if vect_path.exists() and clf_path.exists():
        progress_cb(0.2, "Loading pre-trained SVM model…")
        vectorizer = joblib.load(vect_path)
        clf = joblib.load(clf_path)
        progress_cb(1.0, "[OK] Model loaded from disk.")
        return vectorizer, clf

    import pandas as pd

    parquet_path = project_root() / "files_dataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            "files_dataset.parquet not found and no pre-trained model exists.\n"
            "Run scripts/train_classifier.py to train the model."
        )

    progress_cb(0.1, "Training TF-IDF + SVM model…")
    df_matched = pd.read_parquet(str(parquet_path))
    vectorizer, clf = train_classifier(df_matched["extracted_text"].values, df_matched["label_movement"].values)

    models_dir().mkdir(exist_ok=True)
    joblib.dump(vectorizer, vect_path)
    joblib.dump(clf, clf_path)

    progress_cb(1.0, "[OK] Model trained and saved.")
    return vectorizer, clf


# ---------------------------------------------------------------------------
# FASE 3 — Extracción de datos y CSV de salida
# ---------------------------------------------------------------------------

def build_record(text: str, movement: str, gazetteer: dict) -> Dict[str, str]:
    """Fila del CSV a partir del texto OCR de un oficio y su movimiento ya predicho."""
    from ui.extract_and_validate import extract_name, extract_office_number
    from ui.name_normalization import clean_name, restore_enye
    from ui.ner_segmenter import CORP_KEYWORDS, segment_entity_name

    fullname = clean_name(extract_name(text), gazetteer, CORP_KEYWORDS)
    is_company, nombre, paterno, materno = segment_entity_name(fullname)
    if not is_company:
        nombre, paterno, materno = (restore_enye(p, gazetteer) for p in (nombre, paterno, materno))
    return {
        "MOVIMIENTO": movement,
        "NOMBRE": nombre,
        "PATERNO": paterno,
        "MATERNO": materno,
        "MOTIVO": extract_office_number(text) or "",
        "CIUDAD": "",
        "PAIS": "",
    }


class Prediction(NamedTuple):
    movement: str
    confidence: Optional[float]  # None si el modelo no da probabilidades
    known_terms: Optional[int]   # None si el vectorizador no es TF-IDF


def classify(texts: List[str], vectorizer, clf) -> List[Prediction]:
    """ALTA/BAJA de cada texto, con su probabilidad y cuántos términos del
    vocabulario contiene. Una sola llamada al modelo para todo el lote."""
    if not texts:
        return []
    features = vectorizer.transform(texts)
    indptr = getattr(features, "indptr", None)
    known = [int(indptr[i + 1] - indptr[i]) for i in range(len(texts))] if indptr is not None else [None] * len(texts)
    if hasattr(clf, "predict_proba") and hasattr(clf, "classes_"):
        probs = clf.predict_proba(features)
        best = probs.argmax(axis=1)
        return [Prediction(str(clf.classes_[b]), float(p[b]), k) for b, p, k in zip(best, probs, known, strict=True)]
    return [Prediction(str(m), None, k) for m, k in zip(clf.predict(features), known, strict=True)]


def review_reasons(row: Dict[str, str], prediction: Optional[Prediction] = None) -> List[str]:
    """Por qué una fila ya escrita necesita revisión manual (lista vacía si no)."""
    reasons = []
    if not row.get("NOMBRE", "").strip():
        reasons.append("sin nombre")
    if row.get("MOTIVO", "").strip() in ("", "N/A"):
        reasons.append("sin número de oficio")
    if prediction is not None:
        if prediction.known_terms is not None and prediction.known_terms < MIN_KNOWN_TERMS:
            reasons.append("no parece un oficio de la UIF")
        elif prediction.confidence is not None and prediction.confidence < MIN_CONFIDENCE:
            reasons.append(f"{prediction.movement} dudoso ({prediction.confidence:.0%} de confianza)")
    return reasons


def _unique_csv_name(filename: str, used: set) -> str:
    """Nombre de salida único: dos PDFs homónimos en subcarpetas distintas no
    deben sobrescribir el CSV del otro."""
    stem = Path(filename).stem
    name, n = f"{stem}.csv", 2
    while name.lower() in used:
        name, n = f"{stem}_{n}.csv", n + 1
    used.add(name.lower())
    return name


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding=OUTPUT_ENCODING, errors="replace") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(report: Path, header: List[str], rows: List[Tuple[str, str]]) -> Optional[Path]:
    """Escribe el reporte si hay filas; si no, borra el de una corrida anterior."""
    if not rows:
        report.unlink(missing_ok=True)
        return None
    with open(report, "w", newline="", encoding=OUTPUT_ENCODING, errors="replace") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return report


def write_error_report(out_dir: Path, errors: List[Tuple[str, str]]) -> Optional[Path]:
    """Escribe _errores.csv si hubo fallos; si no, borra el de una corrida anterior."""
    return _write_report(out_dir / ERROR_REPORT, ["ARCHIVO", "ERROR"], errors)


READ_ERROR_REASON = "no se pudo leer el PDF (detalle en _errores.csv)"


def write_review_report(out_dir: Path, review: List[Tuple[str, str]],
                        errors: List[Tuple[str, str]] = ()) -> Optional[Path]:
    """Escribe _revisar.csv, el reporte para quien usa la app: los CSV dudosos o
    incompletos y los PDFs que no se pudieron leer. Si no hay nada, borra el anterior."""
    rows = list(review) + [(name, READ_ERROR_REASON) for name, _ in errors]
    return _write_report(out_dir / REVIEW_REPORT, ["ARCHIVO", "MOTIVO_REVISION"], rows)


def run_output(
    ocr_results: List[OcrResult],
    vectorizer,
    clf,
    output_folder: str,
    progress_cb: ProgressCallback,
    pdf_folder: Optional[str] = None,  # sin uso; se conserva por compatibilidad
    cancel_event: Optional[threading.Event] = None,
) -> PipelineResult:
    """
    Escribe un CSV por PDF leído. Los PDFs cuyo OCR falló no generan CSV y se
    listan en _errores.csv; los CSV sin nombre o sin número de oficio se listan
    en _revisar.csv. Devuelve las rutas escritas, con los fallos en `.errors` y
    los pendientes de revisión en `.review`.
    """
    from ui.ner_segmenter import get_gazetteer

    gazetteer = get_gazetteer()
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Los reportes de una corrida anterior no deben sobrevivir a una cancelación
    for stale in (ERROR_REPORT, REVIEW_REPORT):
        (out_dir / stale).unlink(missing_ok=True)

    ok = [r for r in ocr_results if not r.error]
    errors: List[Tuple[str, str]] = [(r.filename, r.error) for r in ocr_results if r.error]

    # Una sola llamada al clasificador para todo el lote
    predictions = classify([r.text for r in ok], vectorizer, clf)

    written: List[str] = []
    summary: Dict[str, int] = {}
    review: List[Tuple[str, str]] = []
    used_names: set = set()
    total = len(ok)
    for idx, (res, prediction) in enumerate(zip(ok, predictions, strict=True)):
        _check_cancel(cancel_event)
        progress_cb(idx / total, f"[{idx + 1}/{total}] Generating CSV for {res.filename}…")
        try:
            row = build_record(res.text, prediction.movement, gazetteer)
            out_csv = out_dir / _unique_csv_name(res.filename, used_names)
            write_csv(out_csv, [row])
            written.append(str(out_csv))
            reasons = review_reasons(row, prediction)
            if reasons:
                review.append((out_csv.name, "; ".join(reasons)))
            movement = str(row["MOVIMIENTO"])
            summary[movement] = summary.get(movement, 0) + 1
        except Exception as exc:  # un oficio raro no detiene el lote
            errors.append((res.filename, str(exc) or type(exc).__name__))

    report = write_review_report(out_dir, review, errors)
    if write_error_report(out_dir, errors):
        progress_cb(1.0, f"⚠ {len(written)} CSV(s) written, {len(errors)} file(s) with errors — see {ERROR_REPORT}")
    elif review:
        progress_cb(1.0, f"⚠ {len(written)} CSV(s) written, {len(review)} to review — see {REVIEW_REPORT}")
    else:
        progress_cb(1.0, f"[OK] Output complete - {len(written)} CSV(s) written to {output_folder}")
    result = PipelineResult(written, errors, summary, review)
    result.report = report
    return result


# ---------------------------------------------------------------------------
# Pipeline completo (botón "Procesar")
# ---------------------------------------------------------------------------

def _scaled(progress_cb: ProgressCallback, start: float, end: float) -> ProgressCallback:
    """Convierte el avance 0–1 de una fase al tramo [start, end] del total."""
    return lambda pct, msg: progress_cb(start + pct * (end - start), msg)


def run_full_pipeline(
    pdf_folder: str,
    output_folder: str,
    progress_cb: ProgressCallback,
    cancel_event: Optional[threading.Event] = None,
    workers: Optional[int] = None,
) -> PipelineResult:
    """OCR → modelo → CSV. Devuelve los CSV escritos; los fallos van en `.errors`
    y el conteo de altas y bajas en `.summary`. *workers*: PDFs leídos a la vez."""
    ocr_results = run_ocr(pdf_folder, _scaled(progress_cb, 0.0, 0.4), cancel_event, workers=workers)
    _check_cancel(cancel_event)
    vectorizer, clf = run_nlp_and_predict(ocr_results, _scaled(progress_cb, 0.4, 0.65))
    _check_cancel(cancel_event)
    return run_output(ocr_results, vectorizer, clf, output_folder,
                      _scaled(progress_cb, 0.65, 1.0), pdf_folder, cancel_event)
