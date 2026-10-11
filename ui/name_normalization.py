"""
ui/name_normalization.py
========================
Normalización de nombres para la salida CSV.

El modelo de reconocimiento en_PP-OCRv4 solo conoce caracteres ASCII
(models/en_dict.txt), así que nunca devuelve Ñ ni vocales acentuadas:
  - SALDAÑA se lee como SALDANA
  - RAMÍREZ se lee como RAMFREZ (la Í se confunde con F)
Este módulo corrige esos casos usando el gazetteer de nombres y apellidos
(models/gazetteer.joblib) y deja los nombres en el formato de los CSV de
referencia: mayúsculas, vocales sin acento y Ñ conservada.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, FrozenSet, Iterable

# Confusiones típicas del OCR en letras acentuadas o similares.
_CHAR_CONFUSIONS = {"F": "I", "6": "O", "0": "O", "1": "I"}

# Encabezados de tabla que a veces el OCR pega al nombre ("NOMBREE LILIANA ...").
_HEADER_TOKENS = {"NOMBRE", "NOMBREE", "NOMBRES"}


def fold_accents(text: str) -> str:
    """Mayúsculas y vocales sin acento, conservando la Ñ (formato de referencia)."""
    if not text:
        return ""
    s = str(text).upper().replace("Ñ", "\0")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.replace("\0", "Ñ")


def _vocabulary(gazetteer: dict, extra: Iterable[str] = ()) -> FrozenSet[str]:
    words = set()
    for key in ("surnames", "firstnames"):
        for entry in gazetteer.get(key, ()) or ():
            words.update(fold_accents(entry).split())
    words.update(extra)
    return frozenset(words)


# El gazetteer se carga una sola vez por proceso; se indexa por id() para no
# recalcular el vocabulario en cada PDF.
_VOCAB_CACHE: Dict[tuple, tuple] = {}


def _get_vocab(gazetteer: dict, extra: FrozenSet[str]):
    key = (id(gazetteer), extra)
    cached = _VOCAB_CACHE.get(key)
    if cached is None:
        vocab = _vocabulary(gazetteer, extra)
        # SALDANA -> SALDAÑA, solo cuando la forma con Ñ es un nombre conocido
        enye = {w.replace("Ñ", "N"): w for w in vocab if "Ñ" in w}
        cached = (vocab, enye)
        _VOCAB_CACHE[key] = cached
    return cached


def repair_token(token: str, vocab: FrozenSet[str], short_words: FrozenSet[str] = frozenset()) -> str:
    """Corrige un token que el OCR dañó, solo si el resultado es una palabra conocida."""
    if not token or token in vocab or not token.isalnum():
        return token
    # Í leída como F, O leída como 6, etc.
    swapped = "".join(_CHAR_CONFUSIONS.get(c, c) for c in token)
    if swapped != token and swapped in vocab:
        return swapped
    for i, c in enumerate(token):
        if c in _CHAR_CONFUSIONS:
            cand = token[:i] + _CHAR_CONFUSIONS[c] + token[i + 1:]
            if cand in vocab:
                return cand
    # Letra sobrante al final por el borde de la tabla: ZEPEDAE, CORTESS, SCE.
    # Solo una E final o una S duplicada (ARANDAS es un nombre, no ARANDA + S).
    # En tokens cortos solo si queda un sufijo societario (SCE -> SC).
    stripped = token[:-1]
    extra_letter = token[-1] == "E" or token.endswith("SS")
    if extra_letter and stripped in vocab and (len(token) > 3 or stripped in short_words):
        return stripped
    return token


def clean_name(fullname: str, gazetteer: dict, extra_words: Iterable[str] = ()) -> str:
    """Quita acentos, encabezados pegados y corrige errores de OCR token a token."""
    if not fullname:
        return ""
    extra = frozenset(extra_words)
    vocab, _ = _get_vocab(gazetteer, extra)
    tokens = fold_accents(fullname).split()
    while len(tokens) > 1 and tokens[0] in _HEADER_TOKENS:
        tokens = tokens[1:]
    return " ".join(repair_token(t, vocab, extra) for t in tokens)


def restore_enye(name: str, gazetteer: dict) -> str:
    """Devuelve la Ñ que el OCR no puede leer (SALDANA -> SALDAÑA, PENA -> PEÑA)."""
    if not name:
        return name
    _, enye = _get_vocab(gazetteer, frozenset())
    return " ".join(enye.get(t, t) for t in name.split())
