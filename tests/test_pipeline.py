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


BAJA_TEXT = "Oficio 110/K/2001/2026\nSe elimina de la Lista de Personas Bloqueadas a: MARTA IBANEZ PENA, con RFC"
ALTA_TEXT = "Oficio 110/G/1/2026\nmedida preventiva impuesta a GRUPO ALTAVISTA, S.A. DE C.V., con R.F.C. GFE"


def _ok(name, text, folder="x"):
    return OcrResult(name, Path(folder) / name, text)


def test_run_output_writes_one_csv_per_pdf(tmp_path, gaz):
    clf = FakeClassifier()
    results = [
        _ok("marta.pdf", BAJA_TEXT),
        _ok("grupo.pdf", ALTA_TEXT),
        OcrResult("roto.pdf", Path("roto.pdf"), error="PDF dañado"),
    ]
    written = pipeline.run_output(results, FakeVectorizer(), clf, str(tmp_path), lambda p, m: None)

    assert [Path(p).name for p in written] == ["marta.csv", "grupo.csv"]
    assert written.errors == [("roto.pdf", "PDF dañado")]
    assert clf.batches == 1  # el clasificador se llama una sola vez

    person = _read(tmp_path / "marta.csv")[0]
    assert (person["MOVIMIENTO"], person["NOMBRE"], person["PATERNO"], person["MATERNO"]) == (
        "BAJA", "MARTA", "IBAÑEZ", "PEÑA")
    assert person["MOTIVO"] == "OFICIO NO. 110/K/2001/2026"

    company = _read(tmp_path / "grupo.csv")[0]
    assert (company["MOVIMIENTO"], company["NOMBRE"], company["PATERNO"]) == ("ALTA", "GRUPO ALTAVISTA SA DE CV", "")

    assert _read(tmp_path / pipeline.ERROR_REPORT) == [{"ARCHIVO": "roto.pdf", "ERROR": "PDF dañado"}]


def test_csv_is_cp1252_for_excel(tmp_path, gaz):
    pipeline.run_output([_ok("r.pdf", BAJA_TEXT)], FakeVectorizer(), FakeClassifier(), str(tmp_path), lambda p, m: None)
    assert "IBAÑEZ".encode("cp1252") in (tmp_path / "r.csv").read_bytes()


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
        _ok("marta.pdf", BAJA_TEXT),
        _ok("grupo.pdf", ALTA_TEXT),
        _ok("altavista.pdf", ALTA_TEXT),
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


# ---------------------------------------------------------------------------
# CSV que hay que revisar a mano (_revisar.csv)
# ---------------------------------------------------------------------------

NO_NAME_TEXT = "Oficio 110/G/1/2026\n12345\n"
NO_OFFICE_TEXT = "Se elimina de la Lista de Personas Bloqueadas a: MARTA IBANEZ PENA, con RFC"


def test_incomplete_rows_are_listed_for_review(tmp_path, gaz):
    results = [_ok("bien.pdf", BAJA_TEXT), _ok("sin_nombre.pdf", NO_NAME_TEXT),
               _ok("sin_oficio.pdf", NO_OFFICE_TEXT)]
    written = pipeline.run_output(results, FakeVectorizer(), FakeClassifier(), str(tmp_path), lambda p, m: None)

    assert len(written) == 3  # el CSV se escribe igual, pero se marca
    assert written.review == [("sin_nombre.csv", "sin nombre"), ("sin_oficio.csv", "sin número de oficio")]
    assert _read(tmp_path / pipeline.REVIEW_REPORT) == [
        {"ARCHIVO": "sin_nombre.csv", "MOTIVO_REVISION": "sin nombre"},
        {"ARCHIVO": "sin_oficio.csv", "MOTIVO_REVISION": "sin número de oficio"},
    ]
    assert written.needs_review == 2
    assert written.report == tmp_path / pipeline.REVIEW_REPORT


@pytest.mark.parametrize("row, expected", [
    ({"NOMBRE": "MARTA", "MOTIVO": "OFICIO NO. 110/K/1/2026"}, []),
    ({"NOMBRE": " ", "MOTIVO": "N/A"}, ["sin nombre", "sin número de oficio"]),
    ({"NOMBRE": "GRUPO SA DE CV", "MOTIVO": ""}, ["sin número de oficio"]),
])
def test_review_reasons(row, expected):
    assert pipeline.review_reasons(row) == expected


def test_complete_run_removes_stale_review_report(tmp_path, gaz):
    (tmp_path / pipeline.REVIEW_REPORT).write_text("viejo")
    written = pipeline.run_output([_ok("r.pdf", BAJA_TEXT)], FakeVectorizer(), FakeClassifier(),
                                  str(tmp_path), lambda p, m: None)
    assert written.review == []
    assert not (tmp_path / pipeline.REVIEW_REPORT).exists()


def test_cancel_removes_reports_from_previous_run(tmp_path):
    """Al cancelar no debe quedar un _errores.csv o _revisar.csv de otra corrida."""
    for name in (pipeline.ERROR_REPORT, pipeline.REVIEW_REPORT):
        (tmp_path / name).write_text("viejo")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        pipeline.run_output([_ok("r.pdf", BAJA_TEXT)], FakeVectorizer(), FakeClassifier(),
                            str(tmp_path), lambda p, m: None, cancel_event=cancel)
    assert not (tmp_path / pipeline.ERROR_REPORT).exists()
    assert not (tmp_path / pipeline.REVIEW_REPORT).exists()


# ---------------------------------------------------------------------------
# Confianza del clasificador (v2.4.0)
# ---------------------------------------------------------------------------

class SparseVectorizer:
    """Como TF-IDF: matriz dispersa; cada palabra del texto es un término conocido."""

    def transform(self, texts):
        from scipy.sparse import csr_matrix

        rows = [len(t.split()) for t in texts]
        indptr = [0]
        for n in rows:
            indptr.append(indptr[-1] + n)
        indices = [i for n in rows for i in range(n)]
        width = max([1, *rows])
        return csr_matrix(([1.0] * len(indices), indices, indptr), shape=(len(texts), width))


class ProbaClassifier:
    """Devuelve la probabilidad fija que se le pida, por texto."""
    classes_ = ["ALTA", "BAJA"]

    def __init__(self, p_baja):
        self.p_baja = p_baja

    def predict_proba(self, features):
        import numpy as np

        return np.array([[1 - p, p] for p in self.p_baja])


def _long(text):
    """Texto con suficientes términos para no parecer basura."""
    return text + " palabra" * pipeline.MIN_KNOWN_TERMS


def test_classify_uses_probabilities():
    preds = pipeline.classify([_long("a"), _long("b")], SparseVectorizer(), ProbaClassifier([0.99, 0.2]))
    assert [(p.movement, round(p.confidence, 2)) for p in preds] == [("BAJA", 0.99), ("ALTA", 0.8)]
    assert all(p.known_terms > pipeline.MIN_KNOWN_TERMS for p in preds)


def test_classify_without_probabilities():
    preds = pipeline.classify(["elimina", "x"], FakeVectorizer(), FakeClassifier())
    assert [(p.movement, p.confidence, p.known_terms) for p in preds] == [("BAJA", None, None), ("ALTA", None, None)]
    assert pipeline.classify([], FakeVectorizer(), FakeClassifier()) == []


def test_doubtful_movement_goes_to_review(tmp_path, gaz):
    results = [_ok("seguro.pdf", _long(BAJA_TEXT)), _ok("dudoso.pdf", _long(BAJA_TEXT))]
    written = pipeline.run_output(results, SparseVectorizer(), ProbaClassifier([0.99, 0.63]),
                                  str(tmp_path), lambda p, m: None)
    assert written.summary == {"ALTA": 0, "BAJA": 2}
    assert written.review == [("dudoso.csv", "BAJA dudoso (63% de confianza)")]


def test_text_that_is_not_an_office_goes_to_review(tmp_path, gaz):
    written = pipeline.run_output([_ok("factura.pdf", BAJA_TEXT)], SparseVectorizer(), ProbaClassifier([0.99]),
                                  str(tmp_path), lambda p, m: None)
    assert written.review == [("factura.csv", "no parece un oficio de la UIF")]


@pytest.mark.parametrize("prediction, expected", [
    (pipeline.Prediction("ALTA", 0.95, 400), []),
    (pipeline.Prediction("ALTA", 0.5, 400), ["ALTA dudoso (50% de confianza)"]),
    (pipeline.Prediction("BAJA", 0.99, 3), ["no parece un oficio de la UIF"]),
    (pipeline.Prediction("BAJA", None, None), []),
])
def test_review_reasons_with_prediction(prediction, expected):
    row = {"NOMBRE": "MARTA", "MOTIVO": "OFICIO NO. 110/K/1/2026"}
    assert pipeline.review_reasons(row, prediction) == expected


def test_published_model_flags_text_that_is_not_an_office():
    """Con el modelo real: una factura da BAJA con 0.99, pero tiene pocos términos."""
    import joblib

    models = pipeline.models_dir()
    vect, clf = models / "tfidf_vectorizer.joblib", models / "svm_classifier_model.joblib"
    if not (vect.exists() and clf.exists()):
        pytest.skip("modelos no disponibles")
    [pred] = pipeline.classify(["factura de luz recibo total a pagar periodo de consumo"],
                               joblib.load(vect), joblib.load(clf))
    assert pred.known_terms < pipeline.MIN_KNOWN_TERMS
    assert pipeline.review_reasons({"NOMBRE": "X", "MOTIVO": "OFICIO NO. 1"}, pred) == [
        "no parece un oficio de la UIF"]


def test_review_report_includes_unreadable_pdfs(tmp_path, gaz):
    """El reporte para quien usa la app junta todo lo que hay que revisar a mano."""
    results = [_ok("bien.pdf", BAJA_TEXT), OcrResult("roto.pdf", Path("roto.pdf"), error="PDF dañado")]
    written = pipeline.run_output(results, FakeVectorizer(), FakeClassifier(), str(tmp_path), lambda p, m: None)
    assert written.needs_review == 1
    assert _read(written.report) == [{"ARCHIVO": "roto.pdf", "MOTIVO_REVISION": pipeline.READ_ERROR_REASON}]


def test_no_report_when_everything_is_fine(tmp_path, gaz):
    written = pipeline.run_output([_ok("bien.pdf", BAJA_TEXT)], FakeVectorizer(), FakeClassifier(),
                                  str(tmp_path), lambda p, m: None)
    assert written.needs_review == 0 and written.report is None
