"""Separación en nombre / paterno / materno y detección de empresas."""

import pytest

import ui.ner_segmenter as ner
from ui.ner_segmenter import segment_entity_name


@pytest.fixture(autouse=True)
def _needs_model(gaz):
    """Estas pruebas usan el modelo CRF real (vía la fixture gaz)."""


@pytest.mark.parametrize("name, expected", [
    ("MEIYU LEI", (False, "MEIYU", "LEI", "")),
    ("CHEN WANXUAN", (False, "CHEN", "WANXUAN", "")),
    ("JIANCHENG ZHEN", (False, "JIANCHENG", "ZHEN", "")),
    ("MOHAMMED TUFAIL", (False, "MOHAMMED", "TUFAIL", "")),
])
def test_foreign_names_are_people(name, expected):
    assert segment_entity_name(name) == expected


@pytest.mark.parametrize("name, expected", [
    ("JOSE RICARDO MEDINA HERNANDEZ", (False, "JOSE RICARDO", "MEDINA", "HERNANDEZ")),
    ("CARLOS SOLIS DE LA TORRE", (False, "CARLOS", "SOLIS", "DE LA TORRE")),
    ("MARIA DEL CARMEN CASTRO BELTRAN", (False, "MARIA DEL CARMEN", "CASTRO", "BELTRAN")),
    ("ISAI MARTINEZ ZEPEDA", (False, "ISAI", "MARTINEZ", "ZEPEDA")),
])
def test_mexican_names(name, expected):
    assert segment_entity_name(name) == expected


@pytest.mark.parametrize("name", [
    "TECNACAR DE MEXICO SA DE CV",
    "SIMETRIA JURIDICA SC",
    "MKL 2022 SC",
    "GRUPO FELIMO SA DE CV",
    "ASOCIACION PATRONAL REGION ZAMORA AC",
    "SINDICATO PROGRESISTA DE GRASAS Y ACEITES VEGETALES Y ACTIVIDAD AGROPECUARIA",
])
def test_companies(name):
    assert segment_entity_name(name) == (True, name, "", "")


@pytest.mark.parametrize("name", ["", "N/A", "  "])
def test_empty(name):
    assert segment_entity_name(name) == (False, "", "", "")


@pytest.mark.parametrize("tokens, expected", [
    (["LEI"], ("LEI", "", "")),
    (["MEIYU", "LEI"], ("MEIYU", "LEI", "")),
    (["ANA", "LOPEZ", "RUIZ"], ("ANA", "LOPEZ", "RUIZ")),
    (["ANA", "SOFIA", "LOPEZ", "RUIZ"], ("ANA SOFIA", "LOPEZ", "RUIZ")),
])
def test_positional_split(tokens, expected):
    assert ner._positional_split(tokens) == expected


def test_without_model_uses_company_words(monkeypatch):
    monkeypatch.setattr(ner, "_crf_tags", lambda tokens: None)
    assert segment_entity_name("GRUPO FELIMO SA DE CV")[0] is True
    assert segment_entity_name("MEIYU LEI") == (False, "MEIYU", "LEI", "")
