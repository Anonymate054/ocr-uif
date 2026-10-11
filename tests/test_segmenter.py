"""Separación en nombre / paterno / materno y detección de empresas."""

import pytest

import ui.ner_segmenter as ner
from ui.ner_segmenter import segment_entity_name


@pytest.fixture(autouse=True)
def _needs_model(gaz):
    """Estas pruebas usan el modelo CRF real (vía la fixture gaz)."""


@pytest.mark.parametrize("name, expected", [
    ("PERSONA SINTETICA", (False, "SINTETICO", "SINTETICO", "")),
    ("PERSONA SINTETICA", (False, "SINTETICO", "SINTETICO", "")),
    ("PERSONA SINTETICA", (False, "SINTETICO", "SINTETICO", "")),
    ("MOHAMMED TUFAIL", (False, "MOHAMMED", "TUFAIL", "")),
])
def test_foreign_names_are_people(name, expected):
    assert segment_entity_name(name) == expected


@pytest.mark.parametrize("name, expected", [
    ("PERSONA SINTETICA", (False, "JOSE RICARDO", "MEDINA", "HERNANDEZ")),
    ("PERSONA SINTETICA", (False, "CARLOS", "SOLIS", "DE LA TORRE")),
    ("MARIA DEL CARMEN CASTRO BELTRAN", (False, "MARIA DEL CARMEN", "CASTRO", "BELTRAN")),
    ("PERSONA SINTETICA", (False, "ISAI", "MARTINEZ", "ZEPEDA")),
])
def test_mexican_names(name, expected):
    assert segment_entity_name(name) == expected


@pytest.mark.parametrize("name", [
    "EMPRESA SINTETICA",
    "EMPRESA SINTETICA",
    "EMPRESA SINTETICA",
    "EMPRESA SINTETICA",
    "EMPRESA SINTETICA",
    "EMPRESA SINTETICA",
])
def test_companies(name):
    assert segment_entity_name(name) == (True, name, "", "")


@pytest.mark.parametrize("name", ["", "N/A", "  "])
def test_empty(name):
    assert segment_entity_name(name) == (False, "", "", "")


@pytest.mark.parametrize("tokens, expected", [
    (["SINTETICO"], ("SINTETICO", "", "")),
    (["SINTETICO", "SINTETICO"], ("SINTETICO", "SINTETICO", "")),
    (["ANA", "LOPEZ", "RUIZ"], ("ANA", "LOPEZ", "RUIZ")),
    (["ANA", "SOFIA", "LOPEZ", "RUIZ"], ("ANA SOFIA", "LOPEZ", "RUIZ")),
])
def test_positional_split(tokens, expected):
    assert ner._positional_split(tokens) == expected


def test_without_model_uses_company_words(monkeypatch):
    monkeypatch.setattr(ner, "_crf_tags", lambda tokens: None)
    assert segment_entity_name("EMPRESA SINTETICA")[0] is True
    assert segment_entity_name("PERSONA SINTETICA") == (False, "SINTETICO", "SINTETICO", "")
