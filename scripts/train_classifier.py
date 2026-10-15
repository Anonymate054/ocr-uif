"""
scripts/train_classifier.py
===========================
Vuelve a entrenar el clasificador ALTA/BAJA con el mismo OCR que usa la app.

Uso:
    python scripts/train_classifier.py <carpeta> [--models-dir models] [--workers N] [--dry-run]

<carpeta> tiene subcarpetas por fecha (p. ej. files/lpb_oficios_01_06_26) con los
PDFs y sus CSV de referencia. Cada PDF toma la etiqueta (columna MOVIMIENTO) del
CSV de su misma subcarpeta cuyo nombre coincide con el del PDF o, si el PDF no
lleva nombre (110_k_*, escaneos mx_m5071_*), del CSV con el mismo número de
oficio. Los PDFs sin CSV no se usan.

Guarda en <models-dir> el vectorizador TF-IDF, la SVM (la que usa la app) y la
regresión logística de referencia, y muestra la validación cruzada de cada una.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MOVEMENTS = ("ALTA", "BAJA")
OFFICE_RE = re.compile(r"110[/\-\s_]?[GKgk][/\-\s_]?\d+[/\-\s_]?\d+")


def clean_name(name: str) -> str:
    """Nombre de archivo sin prefijo numérico, sufijos (_cnbv, _vs), extensión,
    acentos ni separadores: "10_Simetría_Jurídica_cnbv.pdf" -> "simetriajuridica"."""
    name = unicodedata.normalize("NFKD", name.lower())
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"^\d+_", "", name)
    name = re.sub(r"(_cnbv|_vs|\.pdf|\.csv)", "", name)
    return re.sub(r"[_\-\s]+", "", name)


def office_number(text: str) -> Optional[str]:
    """Número de oficio normalizado: "110/K/2385/2026" -> "110_k_2385_2026"."""
    m = OFFICE_RE.search(text or "")
    return re.sub(r"[/\-\s_]+", "_", m.group(0).lower()) if m else None


def load_csv_labels(folder: Path) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Etiquetas de los CSV de *folder*: lista de {csv, dir, movement} y un índice
    por número de oficio."""
    records, by_office = [], {}
    for path in sorted(Path(folder).rglob("*.csv")):
        raw = path.read_bytes()
        for enc in ("utf-8-sig", "cp1252"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        for row in csv.DictReader(text.splitlines()):
            rec = {"csv": path.name, "dir": path.parent.name,
                   "movement": (row.get("MOVIMIENTO") or "").strip().upper()}
            records.append(rec)
            for value in row.values():
                office = office_number(str(value or ""))
                if office:
                    by_office[office] = rec
                    break
    return records, by_office


def label_for(pdf_name: str, pdf_dir: str, text: str, records, by_office) -> Optional[str]:
    """Etiqueta del PDF: primero por nombre en su subcarpeta, luego por oficio
    (del nombre del archivo o del texto). None si no hay CSV o la etiqueta no es
    ALTA/BAJA."""
    pdf_clean = clean_name(pdf_name)
    found = None
    for rec in records:
        if rec["dir"] == pdf_dir:
            csv_clean = clean_name(rec["csv"])
            if csv_clean and (csv_clean in pdf_clean or pdf_clean in csv_clean):
                found = rec
                break
    if found is None:
        for office in (office_number(pdf_name), office_number(text)):
            if office and office in by_office:
                found = by_office[office]
                break
    if found is None or found["movement"] not in MOVEMENTS:
        return None
    return found["movement"]


def build_dataset(ocr_results, folder: Path):
    """(textos, etiquetas, PDFs sin etiqueta) a partir de los resultados del OCR."""
    records, by_office = load_csv_labels(folder)
    texts, labels, skipped = [], [], []
    for r in ocr_results:
        movement = None if r.error else label_for(r.filename, r.path.parent.name, r.text, records, by_office)
        if movement is None:
            skipped.append(r.filename)
        else:
            texts.append(r.text)
            labels.append(movement)
    return texts, labels, skipped


def train_logistic(texts, labels, vectorizer):
    """Regresión logística de referencia sobre el mismo vectorizador."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(vectorizer.transform(texts), labels)
    return lr


def cross_validate(texts, labels) -> Dict[str, float]:
    """Exactitud y F1 macro de la SVM con validación cruzada estratificada 5x3."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    from ui.pipeline import train_classifier

    X, y = np.array(texts, dtype=object), np.array(labels)
    accs, f1s = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y):
        vec, clf = train_classifier(X[tr], y[tr])
        pred = clf.predict(vec.transform(X[te]))
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return {"accuracy": float(np.mean(accs)), "f1_macro": float(np.mean(f1s))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="no guarda los modelos")
    args = parser.parse_args(argv)

    import joblib

    from ui.ocr import run_ocr
    from ui.paths import models_dir
    from ui.pipeline import train_classifier

    out_dir = args.models_dir or models_dir()
    results = run_ocr(str(args.folder), lambda p, m: print(m), workers=args.workers)
    texts, labels, skipped = build_dataset(results, args.folder)
    counts = {m: labels.count(m) for m in MOVEMENTS}
    print(f"\n{len(texts)} PDFs con etiqueta {counts}; {len(skipped)} sin etiqueta: {skipped}")
    if min(counts.values()) < 5:
        print("Se necesitan al menos 5 PDFs de cada movimiento para entrenar.")
        return 1

    cv = cross_validate(texts, labels)
    print(f"Validación cruzada SVM: exactitud {cv['accuracy']:.2%}, F1 macro {cv['f1_macro']:.3f}")

    vectorizer, svm = train_classifier(texts, labels)
    lr = train_logistic(texts, labels, vectorizer)
    if args.dry_run:
        print("--dry-run: no se guardaron los modelos.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(svm, out_dir / "svm_classifier_model.joblib")
    joblib.dump(lr, out_dir / "logistic_regression_model.joblib")
    print(f"Modelos guardados en {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
