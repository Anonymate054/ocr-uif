"""Separación en nombre / paterno / materno y detección de empresas."""

import pytest

import ui.ner_segmenter as ner
from ui.ner_segmenter import segment_entity_name


@pytest.fixture(autouse=True)
def _needs_model(gaz):
    """Estas pruebas usan el modelo CRF real (vía la fixture gaz)."""


@pytest.mark.parametrize("name, expected", [
    ("XIULAN MEI", (False, "XIULAN", "MEI", "")),
    ("ZHOU JINGYI", (False, "ZHOU", "JINGYI", "")),
    ("WEIMING ZHAO", (False, "WEIMING", "ZHAO", "")),
    ("MOHAMMED RASHEED", (False, "MOHAMMED", "RASHEED", "")),
])
def test_foreign_names_are_people(name, expected):
    assert segment_entity_name(name) == expected


@pytest.mark.parametrize("name, expected", [
    ("JOSE ALBERTO NAVA HERNANDEZ", (False, "JOSE ALBERTO", "NAVA", "HERNANDEZ")),
    ("CARLOS NAVA DE LA TORRE", (False, "CARLOS", "NAVA", "DE LA TORRE")),
    ("MARIA DEL CARMEN SOTO BELTRAN", (False, "MARIA DEL CARMEN", "SOTO", "BELTRAN")),
    ("RAUL ORTEGA ZEPEDA", (False, "RAUL", "ORTEGA", "ZEPEDA")),
])
def test_mexican_names(name, expected):
    assert segment_entity_name(name) == expected


@pytest.mark.parametrize("name", [
    "TECNOAUTO DEL NORTE SA DE CV",
    "SINERGIA JURIDICA SC",
    "QRS 2020 SC",
    "GRUPO ALTAVISTA SA DE CV",
    "ASOCIACION PATRONAL REGION CENTRO AC",
    "SINDICATO UNIDO DE TRABAJADORES DEL CAMPO Y ACTIVIDAD AGROPECUARIA",
])
def test_companies(name):
    assert segment_entity_name(name) == (True, name, "", "")


@pytest.mark.parametrize("name", ["", "N/A", "  "])
def test_empty(name):
    assert segment_entity_name(name) == (False, "", "", "")


@pytest.mark.parametrize("tokens, expected", [
    (["MEI"], ("MEI", "", "")),
    (["XIULAN", "MEI"], ("XIULAN", "MEI", "")),
    (["ANA", "LOPEZ", "RUIZ"], ("ANA", "LOPEZ", "RUIZ")),
    (["ANA", "SOFIA", "LOPEZ", "RUIZ"], ("ANA SOFIA", "LOPEZ", "RUIZ")),
])
def test_positional_split(tokens, expected):
    assert ner._positional_split(tokens) == expected


def test_without_model_uses_company_words(monkeypatch):
    monkeypatch.setattr(ner, "_crf_tags", lambda tokens: None)
    assert segment_entity_name("GRUPO ALTAVISTA SA DE CV")[0] is True
    assert segment_entity_name("XIULAN MEI") == (False, "XIULAN", "MEI", "")
