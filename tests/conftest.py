"""Configuración común de las pruebas: permite `import ui...` desde la raíz."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.ner_segmenter import get_gazetteer  # noqa: E402


@pytest.fixture(scope="session")
def gaz():
    g = get_gazetteer()
    if not g.get("surnames"):
        pytest.skip("models/gazetteer.joblib no disponible")
    return g
