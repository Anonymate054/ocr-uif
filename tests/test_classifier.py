"""Clasificador ALTA/BAJA: etiquetado para entrenar, entrenamiento y modelo publicado."""

import re
from pathlib import Path

import joblib
import pytest

from scripts.train_classifier import (
    build_dataset, clean_name, label_for, load_csv_labels, office_number,
)
from ui.ocr import OcrResult
from ui.paths import models_dir
from ui.pipeline import train_classifier

HEADER = "MOVIMIENTO,NOMBRE,PATERNO,MATERNO,MOTIVO,CIUDAD,PAIS\n"

ALTA_TEXT = (
    "Oficio 110/G/1375/2026. Las entidades financieras a que se refiere el articulo 115 de la "
    "Ley de Instituciones de Credito y la Ley General de Organizaciones y Actividades Auxiliares "
    "del Credito, en las listas de personas bloqueadas mediante oficio emitido"
)
BAJA_TEXT = (
    "Oficio 110/K/2385/2026. En cumplimiento a la suspension definitiva dictada en el juicio de "
    "amparo, con fundamento en el mismo ordenamiento, se resuelve: Primero. Se elimina a dicha "
    "persona de la Lista. Segundo. Para los efectos en los terminos"
)


def test_clean_name():
    assert clean_name("10_Simetría_Jurídica_cnbv.pdf") == "simetriajuridica"
    assert clean_name("meiyu_lei.csv") == "meiyulei"
    assert clean_name("2_marco_antonio_vel_zquez_jir_n.pdf") == "marcoantoniovelzquezjirn"


def test_office_number():
    assert office_number("OFICIO NO. 110/K/3019/2026") == "110_k_3019_2026"
    assert office_number("110_k_2968_2026_vs.pdf") == "110_k_2968_2026"
    assert office_number("sin oficio") is None
    assert office_number(None) is None


@pytest.fixture
def labeled_folder(tmp_path):
    d1 = tmp_path / "lpb_oficios_01_06_26"
    d2 = tmp_path / "lpb_oficios_05_06_26"
    d1.mkdir()
    d2.mkdir()
    (d1 / "meiyu_lei.csv").write_text(HEADER + "ALTA,MEIYU,LEI,,OFICIO NO. 110/G/1375/2026,,\n", encoding="utf-8")
    (d2 / "diana_vanessa_madrid_lupian.csv").write_bytes(
        (HEADER + "BAJA,DIANA VANESSA,MADRID,LUPIÁN,,OFICIO NO. 110/K/3019/2026,\n").encode("cp1252"))
    (d2 / "raro.csv").write_text(HEADER + "OTRO,X,Y,,OFICIO NO. 110/K/1/2026,,\n", encoding="utf-8")
    return tmp_path


def test_load_csv_labels_reads_utf8_and_cp1252(labeled_folder):
    records, by_office = load_csv_labels(labeled_folder)
    assert {r["movement"] for r in records} == {"ALTA", "BAJA", "OTRO"}
    assert by_office["110_k_3019_2026"]["csv"] == "diana_vanessa_madrid_lupian.csv"


def test_label_for_by_name_then_by_office(labeled_folder):
    records, by_office = load_csv_labels(labeled_folder)
    # por nombre, solo dentro de la misma subcarpeta
    assert label_for("11_meiyu_lei.pdf", "lpb_oficios_01_06_26", "", records, by_office) == "ALTA"
    assert label_for("11_meiyu_lei.pdf", "lpb_oficios_05_06_26", "", records, by_office) is None
    # por oficio: del nombre del archivo o del texto
    assert label_for("110_k_3019_2026_vs.pdf", "x", "", records, by_office) == "BAJA"
    assert label_for("mx_m5071.pdf", "x", "Oficio 110/K/3019/2026", records, by_office) == "BAJA"
    # etiquetas que no son ALTA/BAJA no se usan
    assert label_for("110_k_1_2026.pdf", "x", "", records, by_office) is None


def test_build_dataset_skips_unlabeled_and_errors(labeled_folder):
    d1 = labeled_folder / "lpb_oficios_01_06_26"
    results = [
        OcrResult("11_meiyu_lei.pdf", d1 / "11_meiyu_lei.pdf", ALTA_TEXT),
        OcrResult("sin_csv.pdf", d1 / "sin_csv.pdf", "Oficio 110/G/9999/2026"),
        OcrResult("110_k_3019_2026_vs.pdf", d1 / "110_k_3019_2026_vs.pdf", error="ilegible"),
    ]
    texts, labels, skipped = build_dataset(results, labeled_folder)
    assert labels == ["ALTA"] and texts == [ALTA_TEXT]
    assert skipped == ["sin_csv.pdf", "110_k_3019_2026_vs.pdf"]


def test_train_classifier_separates_classes():
    texts = [ALTA_TEXT + f" caso {i}" for i in range(6)] + [BAJA_TEXT + f" caso {i}" for i in range(6)]
    labels = ["ALTA"] * 6 + ["BAJA"] * 6
    vec, clf = train_classifier(texts, labels)
    assert list(clf.predict(vec.transform([ALTA_TEXT, BAJA_TEXT]))) == ["ALTA", "BAJA"]
    assert clf.predict_proba(vec.transform([BAJA_TEXT])).shape == (1, 2)


@pytest.fixture(scope="module")
def shipped():
    paths = [models_dir() / n for n in
             ("tfidf_vectorizer.joblib", "svm_classifier_model.joblib", "logistic_regression_model.joblib")]
    if not all(p.exists() for p in paths):
        pytest.skip("modelos no disponibles")
    return [joblib.load(p) for p in paths]


def test_shipped_models_share_the_vectorizer(shipped):
    vec, svm, lr = shipped
    n = len(vec.get_feature_names_out())
    assert list(svm.classes_) == ["ALTA", "BAJA"] == list(lr.classes_)
    assert svm.calibrated_classifiers_[0].estimator.coef_.shape[1] == n
    assert lr.coef_.shape[1] == n


def test_shipped_vocabulary_has_no_merged_words(shipped):
    """El modelo se entrena con el OCR que separa palabras: el modelo anterior,
    entrenado con texto pegado, tenía 1,032 términos con palabras de 20+ letras."""
    vec = shipped[0]
    merged = [f for f in vec.get_feature_names_out() if any(len(w) >= 20 for w in f.split())]
    assert len(merged) <= 10, merged
    assert not any(re.fullmatch(r"titulardeesta\w+", f) for f in vec.get_feature_names_out())


def test_shipped_svm_classifies_typical_texts(shipped):
    vec, svm, _ = shipped
    assert list(svm.predict(vec.transform([ALTA_TEXT, BAJA_TEXT]))) == ["ALTA", "BAJA"]
