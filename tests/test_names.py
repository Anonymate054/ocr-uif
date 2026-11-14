"""
Pruebas de regresión de la extracción de nombres.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests
"""

import pytest

from ui.extract_and_validate import extract_name, extract_office_number
from ui.name_normalization import clean_name, fold_accents, restore_enye
from ui.ner_segmenter import CORP_KEYWORDS


# ── Acentos y Ñ ──────────────────────────────────────────────────────────────

def test_fold_accents_keeps_enye():
    assert fold_accents("José Ramírez Peña") == "JOSE RAMIREZ PEÑA"


@pytest.mark.parametrize("ocr, expected", [
    ("SALDANA", "SALDAÑA"),
    ("PENA", "PEÑA"),
    ("VILLASENOR", "VILLASEÑOR"),
    ("CASTANEDA", "CASTAÑEDA"),
    ("ORTIZ", "ORTIZ"),          # sin forma con Ñ: no se toca
    ("ESTRADA", "ESTRADA"),
])
def test_restore_enye(gaz, ocr, expected):
    assert restore_enye(ocr, gaz) == expected


@pytest.mark.parametrize("ocr, expected", [
    ("ELENA DE LA CRUZ RAMFREZ", "ELENA DE LA CRUZ RAMIREZ"),  # Í leída como F
    ("PABLO ANDRES LUNA MEJFA", "PABLO ANDRES LUNA MEJIA"),
    ("RAUL ORTEGA ZEPEDAE", "RAUL ORTEGA ZEPEDA"),       # letra sobrante
    ("LUCIA MORALES CORTESS", "LUCIA MORALES CORTES"),
    ("NOMBREE SOFIA DELGADO RIOS", "SOFIA DELGADO RIOS"),
    ("NORTIA INNOVACION CORPORATIVA SCE", "NORTIA INNOVACION CORPORATIVA SC"),
    ("VITALFIX HEALTH AND SPORT", "VITALFIX HEALTH AND SPORT"),  # palabras desconocidas intactas
])
def test_clean_name_repairs_ocr_errors(gaz, ocr, expected):
    assert clean_name(ocr, gaz, CORP_KEYWORDS) == expected


# ── Extracción desde el texto del oficio ─────────────────────────────────────

def test_extract_name_with_accents_and_enye():
    text = "Se elimina de la Lista de Personas Bloqueadas a: MARTA IBAÑEZ PEÑA, con R.F.C. XXXX"
    assert extract_name(text) == "MARTA IBAÑEZ PEÑA"


def test_extract_name_accented_vowels_are_not_cut():
    text = "la quejosa María José Velázquez Ibáñez."
    assert extract_name(text) == "MARÍA JOSÉ VELÁZQUEZ IBÁÑEZ"


@pytest.mark.parametrize("text, expected", [
    ("medida preventiva impuesta, a\nCONSTRUCTORA RIO DE OCCIDENTE S.DE R.L.DE C.V. COn R.F.C.CRO000101AA1",
     "CONSTRUCTORA RIO DE OCCIDENTE S DE RL DE CV"),
    ("medida preventiva impuesta a GRUPO\nALIMENTICIO DE LA SIERRA, S. DE R.L. DE C.V., con R.F.C. GAS000101AA2;",
     "GRUPO ALIMENTICIO DE LA SIERRA S DE RL DE CV"),
    ("medida preventiva impuesta, a\nCORPORATIVO KLM S.C. DE R.L., con R.F.C. CKL000101AA3;",
     "CORPORATIVO KLM SC DE RL"),
    ("impuesta a ZYX\nPROFESIONALES EN SEGURIDAD, S.A. DE C.V., con R.F.C. ZPS000101AA4;",
     "ZYX PROFESIONALES EN SEGURIDAD SA DE CV"),
])
def test_extract_name_keeps_corporate_suffix(text, expected):
    assert extract_name(text) == expected


def test_extract_office_number():
    assert extract_office_number("Oficio No. 110/G/1000/2026") == "OFICIO NO. 110/G/1000/2026"
    assert extract_office_number("sin oficio") == "N/A"


def test_extract_office_number_with_dashes():
    assert extract_office_number("oficio 110-k-2000-2026") == "OFICIO NO. 110/K/2000/2026"


# Oficios 110/K: la tabla "Persona / RFC" va entre la frase y el nombre
@pytest.mark.parametrize("header", ["Persona\nRFC", "Persona\nRFC.", "PERSONA\nRFC", "Nombre"])
def test_extract_name_skips_table_headers(header):
    text = (
        "Se elimina de la Lista de Personas Bloqueadas a:\n"
        f"{header}\nWEIMING ZHAO\nZAWE000101AB1\nDe acuerdo con lo anterior"
    )
    assert extract_name(text) == "WEIMING ZHAO"


def test_extract_name_spelled_out_corporate_suffix():
    text = (
        "Se elimina provisionalmente de la Lista de Personas Bloqueadas a:\nPERSONA\nRFC\n"
        "TECNOAUTO DEL NORTE, SOCIEDAD\nANONIMA DE CAPITAL VARIABLE\nTME123456AB1"
    )
    assert extract_name(text) == "TECNOAUTO DEL NORTE SA DE CV"


def test_extract_name_from_table():
    text = "Relacion de personas\nNOMBRE\nRFC\n1\nJOSE ALBERTO NAVA HERNANDEZ\nNAHA000101AB2"
    assert extract_name(text) == "JOSE ALBERTO NAVA HERNANDEZ"


def test_extract_name_without_name_returns_na():
    assert extract_name("texto sin datos relevantes") == "N/A"
