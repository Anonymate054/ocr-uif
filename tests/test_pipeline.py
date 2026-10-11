"""Fase de salida: un CSV por PDF, reporte de errores y codificación."""

import csv
import threading
from pathlib import Path

import pytest

import ui.pipeline as pipeline
from ui.ocr import Cancelled, OcrResult


class FakeVectorizer:
    def transform(self, texts):
        return list(texts)


class FakeClassifier:
    """BAJA si el texto dice "elimina", si no ALTA."""

    def __init__(self):
        self.batches = 0

    def predict(self, texts):
        self.batches += 1
        return ["BAJA" if "elimina" in t else "ALTA" for t in texts]


def _read(path: Path):
    with open(path, encoding=pipeline.OUTPUT_ENCODING, newline="") as fh:
        return list(csv.DictReader(fh))


BAJA_TEXT = "Oficio 110/K/2385/2026\nSe elimina de la Lista de Personas Bloqueadas a: ROSALVA SALDANA PENA, con RFC"
ALTA_TEXT = "Oficio 110/G/1/2026\nmedida preventiva impuesta a GRUPO FELIMO, S.A. DE C.V., con R.F.C. GFE"


def _ok(name, text, folder="x"):
    return OcrResult(name, Path(folder) / name, text)


def test_run_output_writes_one_csv_per_pdf(tmp_path, gaz):
    clf = FakeClassifier()
    results = [
        _ok("rosalva.pdf", BAJA_TEXT),
        _ok("grupo.pdf", ALTA_TEXT),
        OcrResult("roto.pdf", Path("roto.pdf"), error="PDF dañado"),
    ]
    written = pipeline.run_output(results, FakeVectorizer(), clf, str(tmp_path), lambda p, m: None)

    assert [Path(p).name for p in written] == ["rosalva.csv", "grupo.csv"]
    assert written.errors == [("roto.pdf", "PDF dañado")]
    assert clf.batches == 1  # el clasificador se llama una sola vez

    person = _read(tmp_path / "rosalva.csv")[0]
    assert (person["MOVIMIENTO"], person["NOMBRE"], person["PATERNO"], person["MATERNO"]) == (
        "BAJA", "ROSALVA", "SALDAÑA", "PEÑA")
    assert person["MOTIVO"] == "OFICIO NO. 110/K/2385/2026"

    company = _read(tmp_path / "grupo.csv")[0]
    assert (company["MOVIMIENTO"], company["NOMBRE"], company["PATERNO"]) == ("ALTA", "GRUPO FELIMO SA DE CV", "")

    assert _read(tmp_path / pipeline.ERROR_REPORT) == [{"ARCHIVO": "roto.pdf", "ERROR": "PDF dañado"}]


def test_csv_is_cp1252_for_excel(tmp_path, gaz):
    pipeline.run_output([_ok("r.pdf", BAJA_TEXT)], FakeVectorizer(), FakeClassifier(), str(tmp_path), lambda p, m: None)
    assert "SALDAÑA".encode("cp1252") in (tmp_path / "r.csv").read_bytes()


def test_same_pdf_name_in_two_folders(tmp_path, gaz):
    results = [_ok("a.pdf", BAJA_TEXT, "uno"), _ok("a.pdf", ALTA_TEXT, "dos"), _ok("A.pdf", ALTA_TEXT, "tres")]
    written = pipeline.run_output(results, FakeVectorizer(), FakeClassifier(), str(tmp_path), lambda p, m: None)
    assert [Path(p).name for p in written] == ["a.csv", "a_2.csv", "A_3.csv"]


def test_stale_error_report_is_removed(tmp_path, gaz):
    (tmp_path / pipeline.ERROR_REPORT).write_text("viejo")
    pipeline.run_output([_ok("r.pdf", BAJA_TEXT)], FakeVectorizer(), FakeClassifier(), str(tmp_path), lambda p, m: None)
    assert not (tmp_path / pipeline.ERROR_REPORT).exists()


def test_run_output_cancel(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        pipeline.run_output([_ok("r.pdf", BAJA_TEXT)], FakeVectorizer(), FakeClassifier(),
                            str(tmp_path), lambda p, m: None, cancel_event=cancel)


def test_run_output_without_pdfs(tmp_path):
    written = pipeline.run_output([], FakeVectorizer(), FakeClassifier(), str(tmp_path), lambda p, m: None)
    assert written == [] and written.errors == []


def test_full_pipeline_progress_is_monotonic(tmp_path, monkeypatch, gaz):
    monkeypatch.setattr(pipeline, "run_ocr", lambda folder, cb, cancel=None, workers=None: (cb(0.5, "ocr"), [_ok("r.pdf", BAJA_TEXT)])[1])
    monkeypatch.setattr(pipeline, "run_nlp_and_predict", lambda res, cb: (FakeVectorizer(), FakeClassifier()))
    progress = []
    written = pipeline.run_full_pipeline("in", str(tmp_path), lambda p, m: progress.append(p))
    assert len(written) == 1
    assert progress == sorted(progress) and progress[-1] == 1.0


# ---------------------------------------------------------------------------
# Resumen de altas y bajas al terminar
# ---------------------------------------------------------------------------

def test_summary_counts_altas_and_bajas(tmp_path, gaz):
    results = [
        _ok("rosalva.pdf", BAJA_TEXT),
        _ok("grupo.pdf", ALTA_TEXT),
        _ok("felimo.pdf", ALTA_TEXT),
        OcrResult("roto.pdf", Path("roto.pdf"), error="PDF dañado"),
    ]
    written = pipeline.run_output(results, FakeVectorizer(), FakeClassifier(), str(tmp_path), lambda p, m: None)
    # Los PDFs con error no cuentan como clasificados
    assert written.summary == {"ALTA": 2, "BAJA": 1}
    assert pipeline.format_summary(written) == "3 clasificados · 2 altas · 1 baja"


def test_summary_counts_numpy_labels(tmp_path, gaz):
    """El SVM real devuelve np.str_: el resumen debe seguir agrupando por texto."""
    import numpy as np

    class NumpyClassifier(FakeClassifier):
        def predict(self, texts):
            return np.array(super().predict(texts))

    written = pipeline.run_output([_ok("r.pdf", BAJA_TEXT), _ok("g.pdf", ALTA_TEXT)], FakeVectorizer(),
                                  NumpyClassifier(), str(tmp_path), lambda p, m: None)
    assert written.summary == {"ALTA": 1, "BAJA": 1}
    assert all(type(k) is str for k in written.summary)


def test_summary_without_pdfs(tmp_path):
    written = pipeline.run_output([], FakeVectorizer(), FakeClassifier(), str(tmp_path), lambda p, m: None)
    assert written.summary == {"ALTA": 0, "BAJA": 0}
    assert pipeline.format_summary(written) == "0 clasificados · 0 altas · 0 bajas"


def test_format_summary_singular():
    result = pipeline.PipelineResult(["a.csv"], summary={"ALTA": 1})
    assert pipeline.format_summary(result) == "1 clasificado · 1 alta · 0 bajas"


def test_full_pipeline_passes_cores_to_ocr(tmp_path, monkeypatch, gaz):
    seen = {}

    def fake_ocr(folder, cb, cancel=None, workers=None):
        seen["workers"] = workers
        return [_ok("r.pdf", BAJA_TEXT)]

    monkeypatch.setattr(pipeline, "run_ocr", fake_ocr)
    monkeypatch.setattr(pipeline, "run_nlp_and_predict", lambda res, cb: (FakeVectorizer(), FakeClassifier()))
    written = pipeline.run_full_pipeline("in", str(tmp_path), lambda p, m: None, workers=3)
    assert seen["workers"] == 3
    assert written.summary == {"ALTA": 0, "BAJA": 1}
