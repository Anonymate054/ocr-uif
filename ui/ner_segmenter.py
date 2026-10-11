"""
ui/ner_segmenter.py
===================
Intelligent Named Entity Recognition (NER) segmenter for Mexican individuals and corporate entities.
Uses a lightweight Conditional Random Field (CRF) sequence tagger trained on official SAT and SEP datasets.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

import joblib

from ui.paths import models_dir as _models_dir

_CRF_MODEL = None
_GAZETTEER = None

PREPOSITIONS: Set[str] = {'DE', 'DEL', 'LA', 'LAS', 'LOS', 'SAN', 'SANTA', 'Y', 'VON', 'VAN'}
CORP_KEYWORDS: Set[str] = {
    'SA', 'CV', 'SC', 'AC', 'SAPI', 'RL', 'SAS', 'SOFOM', 'SNC', 'SDR',
    'LLC', 'LTD', 'LIMITED', 'INC', 'INCORPORATED', 'CORP', 'CORPORATION', 'GMBH',
    'AG', 'BV', 'NV', 'PLC', 'SARL', 'CO', 'COMPANY', 'HOLDINGS', 'HOLDING',
    'INTERNATIONAL', 'ENTERPRISES', 'VENTURES', 'CAPITAL', 'PARTNERS', 'INVESTMENTS',
    'TECHNOLOGIES', 'TECH', 'SOLUTIONS', 'SYSTEMS', 'WORLDWIDE',
    'SINDICATO', 'ASOCIACION', 'CORPORACION', 'CLINICA', 'CONSTRUCTORA', 'GRUPO',
    'SERVICIOS', 'LOGISTICA', 'PROYECTOS', 'DESARROLLOS', 'EDIFICACIONES',
    'INNOVACION', 'PATRONAL', 'CONFEDERACION', 'JURIDICA', 'JURIDICO',
    'COMPETITIVIDAD', 'PRODUCTIVIDAD', 'INTEGRADORA', 'EMPRESARIAL', 'AMERICANA',
    'GLOBAL', 'MINING', 'ACCESORIOS', 'NOVEDADES', 'ADMINISTRATIVO', 'PLANEACION',
    'LEGAL', 'COMERCIAL', 'DISTRIBUIDORA', 'INDUSTRIA', 'INDUSTRIAS', 'OPERADORA'
}

def _load_model_and_gazetteer() -> None:
    global _CRF_MODEL, _GAZETTEER
    if _CRF_MODEL is None or _GAZETTEER is None:
        model_path = _models_dir() / "crf_ner_model.joblib"
        gaz_path = _models_dir() / "gazetteer.joblib"
        
        if model_path.exists() and gaz_path.exists():
            try:
                _CRF_MODEL = joblib.load(str(model_path))
                _GAZETTEER = joblib.load(str(gaz_path))
            except Exception:
                _CRF_MODEL = None
                _GAZETTEER = None

def get_gazetteer() -> dict:
    _load_model_and_gazetteer()
    return _GAZETTEER or {"surnames": set(), "firstnames": set()}

def _word2features(sent: List[str], i: int, surnames: Set[str], firstnames: Set[str]) -> dict:
    word = sent[i]
    w_upper = word.upper()
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word.isupper()': word.isupper(),
        'word.isdigit()': word.isdigit(),
        'is_in_surnames': w_upper in surnames,
        'is_in_firstnames': w_upper in firstnames,
        'is_prep': w_upper in PREPOSITIONS,
        'is_corp_kw': w_upper in CORP_KEYWORDS,
        'sent_has_corp_kw': any(w.upper() in CORP_KEYWORDS for w in sent),
        'sent_len': len(sent),
        'length': len(word),
    }
    
    if i > 0:
        prev_word = sent[i-1]
        pw_upper = prev_word.upper()
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.isupper()': prev_word.isupper(),
            '-1:is_in_surnames': pw_upper in surnames,
            '-1:is_in_firstnames': pw_upper in firstnames,
            '-1:is_prep': pw_upper in PREPOSITIONS,
            '-1:is_corp_kw': pw_upper in CORP_KEYWORDS,
        })
    else:
        features['BOS'] = True

    if i > 1:
        prev2_word = sent[i-2]
        features.update({
            '-2:word.lower()': prev2_word.lower(),
            '-2:is_prep': prev2_word.upper() in PREPOSITIONS,
        })

    if i < len(sent) - 1:
        next_word = sent[i+1]
        nw_upper = next_word.upper()
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.isupper()': next_word.isupper(),
            '+1:is_in_surnames': nw_upper in surnames,
            '+1:is_in_firstnames': nw_upper in firstnames,
            '+1:is_prep': nw_upper in PREPOSITIONS,
            '+1:is_corp_kw': nw_upper in CORP_KEYWORDS,
        })
    else:
        features['EOS'] = True

    if i < len(sent) - 2:
        next2_word = sent[i+2]
        features.update({
            '+2:word.lower()': next2_word.lower(),
            '+2:is_corp_kw': next2_word.upper() in CORP_KEYWORDS,
        })

    return features

# Palabras que delatan una persona moral aunque el OCR haya perdido el sufijo
# societario. Es un conjunto aparte de CORP_KEYWORDS porque CORP_KEYWORDS es
# una variable del modelo CRF ya entrenado y cambiarla alteraría sus resultados.
COMPANY_WORDS: Set[str] = CORP_KEYWORDS | {
    'SOCIEDAD', 'EMPRESA', 'EMPRESAS', 'CORPORATIVO', 'CORPORATIVA', 'COMERCIALIZADORA',
    'INMOBILIARIA', 'CONSTRUCCIONES', 'CONSTRUCTOR', 'DESARROLLO', 'DESARROLLADORA',
    'CONSULTORES', 'CONSULTORIA', 'ASESORES', 'ESTUDIO', 'ARQUITECTOS', 'INGENIERIA',
    'TRANSPORTES', 'IMPORTADORA', 'EXPORTADORA', 'PROMOTORA', 'INVERSIONES', 'FINANCIERA',
    'AGENCIA', 'FUNDACION', 'INSTITUTO', 'CENTRO', 'BANCO', 'NEGOCIOS', 'SERVICIO',
    'PROFESIONALES', 'PROGRAMAS', 'SUMINISTROS', 'SOLUCIONES', 'TECNOLOGIA', 'TECNOLOGIAS',
}

_PERSON_TAGS = ("FIRST_NAME", "SECOND_NAME", "PAT_SURNAME", "MAT_SURNAME")


def _tag_type(tag: str) -> str:
    """B-PAT_SURNAME -> PAT_SURNAME"""
    return tag.split("-")[-1]


def _positional_split(tokens: List[str]) -> Tuple[str, str, str]:
    """Orden mexicano sin modelo: nombre(s), apellido paterno, apellido materno.
    Sirve también para nombres extranjeros que el CRF no conoce (MEIYU LEI)."""
    if len(tokens) == 1:
        return tokens[0], "", ""
    if len(tokens) == 2:
        return tokens[0], tokens[1], ""
    return " ".join(tokens[:-2]), tokens[-2], tokens[-1]


def _crf_tags(tokens: List[str]) -> Optional[List[str]]:
    if _CRF_MODEL is None or _GAZETTEER is None:
        return None
    try:
        surnames = _GAZETTEER.get("surnames", set())
        firstnames = _GAZETTEER.get("firstnames", set())
        features = [_word2features(tokens, i, surnames, firstnames) for i in range(len(tokens))]
        return [_tag_type(t) for t in _CRF_MODEL.predict([features])[0]]
    except Exception:
        return None


def segment_entity_name(fullname: str) -> Tuple[bool, str, str, str]:
    """
    Clasifica y separa un nombre:
      - empresa: (True, razón social completa, "", "")
      - persona: (False, nombre(s), apellido paterno, apellido materno)
    """
    if not fullname or fullname.strip() in ("", "N/A", "NA"):
        return False, "", "", ""

    clean_str = re.sub(r"[ \t]+", " ", str(fullname)).strip().upper()
    tokens = clean_str.split()
    if not tokens:
        return False, "", "", ""

    _load_model_and_gazetteer()
    tags = _crf_tags(tokens)
    has_company_word = any(re.sub(r"\W", "", t) in COMPANY_WORDS for t in tokens)

    if tags is None:
        # Sin modelo: empresa solo si tiene una palabra societaria
        if has_company_word:
            return True, clean_str, "", ""
        return (False, *_positional_split(tokens))

    n_company = tags.count("COMPANY")
    n_person = sum(tags.count(t) for t in ("FIRST_NAME", "PAT_SURNAME", "MAT_SURNAME"))
    crf_says_company = n_company > n_person
    # El CRF marca como empresa casi cualquier nombre de 2-3 palabras que no
    # conoce (MEIYU LEI, CHEN WANXUAN). Sin una palabra societaria, eso es una
    # persona con nombre extranjero.
    if crf_says_company and (has_company_word or len(tokens) > 3):
        return True, clean_str, "", ""

    if n_company or not all(t in _PERSON_TAGS for t in tags):
        return (False, *_positional_split(tokens))

    by_tag = {t: [tok for tok, tag in zip(tokens, tags) if tag == t] for t in _PERSON_TAGS}
    nombre = " ".join(by_tag["FIRST_NAME"] + by_tag["SECOND_NAME"])
    paterno = " ".join(by_tag["PAT_SURNAME"])
    materno = " ".join(by_tag["MAT_SURNAME"])
    if not paterno and not materno and len(tokens) >= 2:
        # Persona sin apellidos separados: el último token es el apellido
        nombre, paterno = " ".join(tokens[:-1]), tokens[-1]
    return False, nombre, paterno, materno
