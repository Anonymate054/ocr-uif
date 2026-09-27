"""
ui/ner_segmenter.py
===================
Intelligent Named Entity Recognition (NER) segmenter for Mexican individuals and corporate entities.
Uses a lightweight Conditional Random Field (CRF) sequence tagger trained on official SAT and SEP datasets.
"""

from __future__ import annotations

import os
import sys
import re
from pathlib import Path
from typing import Tuple, List, Set
import joblib

def _get_project_root() -> Path:
    """Return the runtime root of the application (handles PyInstaller bundle)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent

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

def _load_model_and_gazetteer():
    global _CRF_MODEL, _GAZETTEER
    if _CRF_MODEL is None or _GAZETTEER is None:
        models_dir = _get_project_root() / "models"
        model_path = models_dir / "crf_ner_model.joblib"
        gaz_path = models_dir / "gazetteer.joblib"
        
        if model_path.exists() and gaz_path.exists():
            try:
                _CRF_MODEL = joblib.load(str(model_path))
                _GAZETTEER = joblib.load(str(gaz_path))
            except Exception:
                _CRF_MODEL = None
                _GAZETTEER = None

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

def segment_entity_name(fullname: str) -> Tuple[bool, str, str, str]:
    """
    Intelligently classifies and segments an entity string into:
    (is_company, nombre, paterno, materno)
    
    - If company: (True, full_company_name, "", "")
    - If person:  (False, full_first_and_second_names, pat_surname, mat_surname)
    """
    if not fullname or fullname.strip() in ("", "N/A", "NA"):
        return False, "", "", ""
        
    clean_str = re.sub(r"[ \t]+", " ", str(fullname)).strip().upper()
    tokens = clean_str.split()
    if not tokens:
        return False, "", "", ""
        
    _load_model_and_gazetteer()
    
    # If CRF model is available, use it for high-precision sequence tagging
    if _CRF_MODEL is not None and _GAZETTEER is not None:
        try:
            surnames = _GAZETTEER.get("surnames", set())
            firstnames = _GAZETTEER.get("firstnames", set())
            
            features = [_word2features(tokens, i, surnames, firstnames) for i in range(len(tokens))]
            predicted_tags = _CRF_MODEL.predict([features])[0]
            
            first_names = []
            second_names = []
            pat_surnames = []
            mat_surnames = []
            company_tokens = []
            
            for token, tag in zip(tokens, predicted_tags):
                tag_type = tag.split("-")[-1] if "-" in tag else tag
                if tag_type == "COMPANY":
                    company_tokens.append(token)
                elif tag_type == "FIRST_NAME":
                    first_names.append(token)
                elif tag_type == "SECOND_NAME":
                    second_names.append(token)
                elif tag_type == "PAT_SURNAME":
                    pat_surnames.append(token)
                elif tag_type == "MAT_SURNAME":
                    mat_surnames.append(token)
                    
            is_company = len(company_tokens) > (len(first_names) + len(pat_surnames) + len(mat_surnames))
            
            if is_company:
                return True, clean_str, "", ""
            else:
                nombre = " ".join(first_names + second_names)
                paterno = " ".join(pat_surnames)
                materno = " ".join(mat_surnames)
                # Fallback if person was detected but no surnames separated
                if not paterno and not materno and len(tokens) >= 2:
                    paterno = tokens[-1]
                    nombre = " ".join(tokens[:-1])
                return False, nombre, paterno, materno
        except Exception:
            pass

    # Robust fallback heuristic if model is absent
    is_company = any(w in CORP_KEYWORDS for w in tokens)
    if is_company:
        return True, clean_str, "", ""
    if len(tokens) == 1:
        return False, tokens[0], "", ""
    elif len(tokens) == 2:
        return False, tokens[0], tokens[1], ""
    elif len(tokens) == 3:
        return False, tokens[0], tokens[1], tokens[2]
    else:
        return False, " ".join(tokens[:-2]), tokens[-2], tokens[-1]
