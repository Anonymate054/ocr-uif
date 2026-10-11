"""
ui/extract_and_validate.py
==========================
Extracción del número de oficio y del nombre de la persona o empresa a partir
del texto OCR de la primera página de un oficio de la UIF.

El nombre se busca en tres pasos, del más preciso al más general:
  1. Frases del oficio ("Se elimina ... a:", "impuesta a", "quejosa", ...).
  2. Tablas con encabezado NOMBRE / PERSONA.
  3. La primera línea larga en mayúsculas que no sea un encabezado.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

# ── Número de oficio ─────────────────────────────────────────────────────────

# 110/K/2924/2026, 110-G-1329-2026, 110 K 2924 2026...
_OFICIO_RE = re.compile(r"110[/\-\s_]?[GKgk][/\-\s_]?\d+[/\-\s_]?\d+")
_OFICIO_SEP_RE = re.compile(r"[/\-\s_]+")


def extract_office_number(text: str) -> str:
    """Número de oficio normalizado ("OFICIO NO. 110/K/2924/2026") o "N/A"."""
    m = _OFICIO_RE.search(text)
    if m:
        parts = _OFICIO_SEP_RE.split(m.group(0).upper())
        if len(parts) >= 4:
            return f"OFICIO NO. {parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}"
    return "N/A"


# ── Sufijos societarios ──────────────────────────────────────────────────────

# Variantes de OCR y forma estándar, en el orden en que deben probarse
_SOCIEDADES = [
    (r"\bSOCIEDAD\s+AN[OÓ]NIMA\s+PROMOTORA\s+DE\s+INVERSI[OÓ]N\s+DE\s+CAPITAL\s+VARIABLE\b", "SAPI DE CV"),
    (r"\bSOCIEDAD\s+DE\s+RESPONSABILIDAD\s+LIMITADA\s+DE\s+CAPITAL\s+VARIABLE\b", "S DE RL DE CV"),
    (r"\bSOCIEDAD\s+AN[OÓ]NIMA\s+DE\s+CAPITAL\s+VARIABLE\b", "SA DE CV"),
    (r"\bSOCIEDAD\s+CIVIL\b", "SC"),
    (r"\bASOCIACI[OÓ]N\s+CIVIL\b", "AC"),
    (r"\bS[\.\,\s]*A[\.\,\s]*P[\.\,\s]*I[\.\,\s]*(?:DE[\.\,\s]*)?C[\.\,\s]*V\b", "SAPI DE CV"),
    (r"\bS[\.\,\s]*DE[\.\,\s]*R[\.\,\s]*L[\.\,\s]*(?:DE[\.\,\s]*)?C[\.\,\s]*V\b", "S DE RL DE CV"),
    (r"\bS[\.\,\s]*A[\.\,\s]*(?:DE[\.\,\s]*)?C[\.\,\s]*V\b", "SA DE CV"),
    (r"\bS[\.\,\s]*A[\.\,\s]*S[\.\,\s]*(?:DE[\.\,\s]*)?C[\.\,\s]*V\b", "SAS DE CV"),
    (r"\bS[\.\,\s]*C[\.\,\s]*(?:DE[\.\,\s]*)?R[\.\,\s]*L\b", "SC DE RL"),
    (r"\bS[\.\,\s]*C\b", "SC"),
    (r"\bA[\.\,\s]*C\b", "AC"),
]
# En el texto completo solo en mayúsculas (en minúsculas sería texto corrido
# del oficio); en el nombre ya extraído, sin distinguir mayúsculas.
_SUFFIX_IN_TEXT = [(re.compile(r",?[ \t]*" + pat), " " + rep) for pat, rep in _SOCIEDADES]
_SUFFIX_IN_NAME = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in _SOCIEDADES]


def _join_corporate_suffixes(text: str) -> str:
    """Convierte ", S. DE R.L. DE C.V." en " S DE RL DE CV" para que los patrones
    de nombre no corten la razón social en los puntos y comas del sufijo."""
    for rx, rep in _SUFFIX_IN_TEXT:
        text = rx.sub(rep, text)
    return text


# ── Validación de candidatos ─────────────────────────────────────────────────

# Frases genéricas que los patrones pueden capturar en lugar de un nombre,
# incluidos los encabezados de tabla (PERSONA / RFC) de los oficios 110/K.
_REJECTS = frozenset({
    "DICHA PERSONA", "DICHAS PERSONAS", "LA QUEJOSA", "EL QUEJOSO", "LAS QUEJOSAS",
    "LOS QUEJOSOS", "EL ELIMINADO", "LOS ELIMINADOS", "LA SOCIEDAD", "LAS SOCIEDADES",
    "PERSONA", "NOMBRE", "RFC",
})

# Si aparecen dentro del candidato, es una frase del oficio y no un nombre
_INVALID_KEYWORDS = (
    "LISTA DE PERSONAS", "LISTA", "BLOQUEADAS", "CONGELACION", "ACTIVOS", "SANCIONES",
    "RESOLUCION", "ACUERDO", "OFICIO", "ELIMINACION", "CONGELAR", "BLOQUEAR",
    "ESTADO ISLAMICO", "CONSEJO DE SEGURIDAD", "NACIONES UNIDAS", "JUICIO DE AMPARO",
    "TRIBUNAL", "COLEGIADO", "JUZGADO", "DISTRITO", "REVISION", "PROHIBICION", "EMBARGO",
    "DE VIAJAR", "ARMAS", "N.O", "MATERIA", "HACIENDA", "SECRETARIA",
)


def _is_valid_candidate(name_upper: str) -> bool:
    return name_upper not in _REJECTS and not any(kw in name_upper for kw in _INVALID_KEYWORDS)


# ── Limpieza del nombre extraído ─────────────────────────────────────────────

# Nombres compuestos que el OCR pega ("JOSERICARDO") y que sí existen juntos
_MERGE_EXEMPT = frozenset({
    "ANABEL", "MARIANA", "JOSEFINA", "SANTIAGO", "LUISA", "ANAHI",
    "ANALIA", "MARIBEL", "MARISOL", "MARICELA", "ANTONIA", "ANTONIO",
    "ANASTASIA", "ANSELMO", "MARITZA", "MARINA",
})
_MERGE_PREFIXES = ("JOSE", "MARIA", "JUAN", "LUIS", "ANA", "SAN")
_TRAILING_PUNCT_RE = re.compile(r"[\.\,\;\:\-\_\s]+$")
_LEADING_PUNCT_RE = re.compile(r"^[\.\,\;\:\-\_\s]+")
_SPACES_RE = re.compile(r"\s+")


def _known_names() -> tuple:
    try:
        from ui.ner_segmenter import get_gazetteer
        gaz = get_gazetteer()
        return gaz.get("firstnames", set()), gaz.get("surnames", set())
    except Exception:
        return set(), set()


def _split_merged_word(word: str, firstnames, surnames) -> Iterable[str]:
    """JOSERICARDO -> JOSE RICARDO, solo si RICARDO es un nombre conocido."""
    upper = word.upper()
    if upper in firstnames or upper in surnames or upper in _MERGE_EXEMPT:
        return (word,)
    for prefix in _MERGE_PREFIXES:
        if upper.startswith(prefix) and len(upper) > len(prefix):
            rest = word[len(prefix):]
            if rest.isalpha() and (rest.upper() in firstnames or (not firstnames and len(rest) >= 4)):
                return (word[:len(prefix)], rest)
    return (word,)


def _clean_ocr_name(name: str) -> str:
    firstnames, surnames = _known_names()
    words = [part for w in name.split() for part in _split_merged_word(w, firstnames, surnames)]
    result = " ".join(words)
    for rx, rep in _SUFFIX_IN_NAME:
        result = rx.sub(rep, result)
    result = _LEADING_PUNCT_RE.sub("", _TRAILING_PUNCT_RE.sub("", result))
    return _SPACES_RE.sub(" ", result).strip()


# ── Paso 1: frases del oficio ────────────────────────────────────────────────

_NAME = r"([a-zA-ZÁÉÍÓÚÜÑáéíóúüñ\s\n\.,_-]{3,120}?)"
_RFC = r"\s*con\s*R\.?F\.?C\.?"
# Encabezados de tabla entre la frase y el nombre ("a:\nPersona\nRFC.\nJIANCHENG...")
_HEADER_LINES = r"(?:(?:Persona|Nombre|RFC|N\.?o)\.?[ \t]*\n\s*)*"
_ITEM_NUMBER = r"(?:\d\)\s*)?"
_END_DEFAULT = rf"(?:,|{_RFC}|;|\s*de\s*la\s*Lista|\.|\n\n)"

_PHRASE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    # Se elimina de la Lista de Personas Bloqueadas a: <NAME>
    rf"elimina,?\s*(?:de\s+la\s+Lista\s+de\s+Personas\s+Bloqueadas\s+)?a\s*:\s*{_HEADER_LINES}{_ITEM_NUMBER}{_NAME}(?:\n\n|\r|\n|;|,|{_RFC}|\.)",
    # eliminación de <NAME>, con RFC
    rf"eliminaci[oó6]n,?\s*de\s*{_NAME}(?:,|{_RFC}|;|\s*de\s*la\s*Lista|\n\n)",
    # impuesta a <NAME> / impuestaa <NAME>
    rf"impuesta,?\s*a\s*{_NAME}{_END_DEFAULT}",
    rf"impuestaa,?\s*{_NAME}{_END_DEFAULT}",
    # hace a <NAME>
    rf"hace,?\s*a\s*{_NAME}{_END_DEFAULT}",
    # sanciones a: 1) <NAME>
    rf"sanciones,?\s*a\s*:\s*{_HEADER_LINES}{_ITEM_NUMBER}{_NAME}(?:;|,|{_RFC}|\.|\n\n)",
    # suspendidos a <NAME> / suspendidosa <NAME>
    rf"suspendidos,?\s*a\s*{_NAME}{_END_DEFAULT}",
    rf"suspendidosa,?\s*{_NAME}{_END_DEFAULT}",
    # amparo: quejosa <NAME>
    rf"quejosa\s+{_NAME}(?:\.|\n\n|con\s+R\.?F\.?C\.?)",
)]
_CANDIDATE_TAIL_RE = re.compile(
    r"[,;\.\s]+(?:con|de|la|R\.?F\.?C\.?|C\.?O\.?N\.?|con\s+R\.?F\.?C\.?)*$", re.IGNORECASE
)
_CANDIDATE_HEAD_RE = re.compile(r"^[,\s\.]+")


def _from_phrases(text: str) -> Optional[str]:
    for rx in _PHRASE_PATTERNS:
        for m in rx.finditer(text):
            cand = _SPACES_RE.sub(" ", m.group(1).strip())
            cand = _CANDIDATE_HEAD_RE.sub("", _CANDIDATE_TAIL_RE.sub("", cand).strip()).strip()
            if cand and _is_valid_candidate(cand.upper()):
                return _clean_ocr_name(cand.upper())
    return None


# ── Paso 2: tablas con encabezado ────────────────────────────────────────────

_TABLE_HEADERS = frozenset({"NOMBRE", "N.O NOMBRE", "N.O", "NOMBRE RFC", "FC NOMBRE", "PERSONA"})
_TABLE_SKIP = frozenset({"RFC", "FC", "N.O", "CURP", "FECHA", "NOMBRE", "PERSONA", "N.O."})
_TABLE_FOOTERS = ("av. constituyentes", "margarita maza", "pagina")
_DATE_RE = re.compile(r"\d{2}/\d{2}/\d{4}")
_NON_LETTERS_RE = re.compile(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")


def _looks_like_code(line: str) -> bool:
    """Una sola palabra con dígitos (RFC, CURP, folio)."""
    return len(line.split()) == 1 and any(c.isdigit() for c in line)


def _mostly_upper(line: str) -> bool:
    letters = _NON_LETTERS_RE.sub("", line)
    return bool(letters) and sum(c.isupper() for c in letters) > len(letters) * 0.5


def _from_table(lines: list) -> Optional[str]:
    for i, line in enumerate(lines):
        if line.upper() not in _TABLE_HEADERS:
            continue
        name_lines = []
        for cand in lines[i + 1:i + 10]:
            if cand.upper() in _TABLE_SKIP:
                continue
            if _DATE_RE.search(cand):
                break
            if cand.isascii() and cand.isdigit():
                if int(cand) < 100:
                    continue  # número de renglón
                break
            if any(f in cand.lower() for f in _TABLE_FOOTERS):
                break
            if _looks_like_code(cand) and len(cand) > 8:
                break  # RFC: termina el nombre
            if _mostly_upper(cand):
                name_lines.append(cand)
        filtered = [l for l in name_lines if not _looks_like_code(l)]
        if filtered:
            name = " ".join(filtered).upper()
            if _is_valid_candidate(name):
                return _clean_ocr_name(name)
    return None


# ── Paso 3: primera línea en mayúsculas ──────────────────────────────────────

_HEADER_KEYWORDS = (
    "HACIENDA", "INTELI", "FINAN", "UNIDAD", "MEXICO", "RESOLU", "PRESENTE", "OFICIO",
    "DIREC", "GENERAL", "LISTA", "PERSONA", "BLOQ", "PROCED", "GARANT", "AUDIENCIA", "PAGINA",
)


def _from_uppercase_line(lines: list) -> Optional[str]:
    for line in lines:
        if len(line) > 10 and line.isupper() and not any(c.isdigit() for c in line):
            if any(k in line for k in _HEADER_KEYWORDS):
                continue
            if _is_valid_candidate(line):
                return _clean_ocr_name(line)
    return None


_BLANKS_RE = re.compile(r"[ \t]+")


def extract_name(text: str) -> str:
    """Nombre de la persona o razón social del oficio, en mayúsculas, o "N/A"."""
    text_clean = _join_corporate_suffixes(_BLANKS_RE.sub(" ", text))
    lines = [s for s in (line.strip() for line in text_clean.split("\n")) if s]
    return _from_phrases(text_clean) or _from_table(lines) or _from_uppercase_line(lines) or "N/A"
