"""
ui/paths.py
===========
Rutas de la aplicación, válidas tanto desde el código fuente como desde el
ejecutable de PyInstaller.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """True cuando corre dentro del .exe de PyInstaller."""
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """Raíz con models/ y ui/assets/ (la carpeta temporal del .exe o la del repo)."""
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    return project_root() / "models"


def transcriptions_dir() -> Path:
    """Caché de OCR. En el .exe va al perfil del usuario, nunca junto al ejecutable,
    para que las transcripciones no viajen si se copia la carpeta de la app."""
    if is_frozen():
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".ocr-uif")
        return Path(base) / "OCR-UIF" / "transcriptions"
    return project_root() / "transcriptions"
