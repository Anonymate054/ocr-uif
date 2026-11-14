"""Versión de la app: VERSION, el respaldo de ui/version.py y bump_version.py."""

import importlib.util
from pathlib import Path

import pytest

import ui.version as version

ROOT = Path(__file__).resolve().parent.parent


def _load_bump():
    spec = importlib.util.spec_from_file_location("bump_version", ROOT / "scripts" / "bump_version.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_version_matches_version_file():
    """Si el VERSION no se empaqueta, la app debe mostrar la misma versión."""
    assert version.DEFAULT_VERSION == (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def test_resolved_version_is_the_version_file():
    assert version.__version__ == (ROOT / "VERSION").read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("semver, ui", [("2.3.0", "demo - v 2.3"), ("2.3.5", "demo - v 2.3.5")])
def test_format_ui_version(semver, ui):
    assert version.format_ui_version(semver) == ui


def test_parse_semver_falls_back_to_default():
    assert version.parse_semver("basura") == version.parse_semver(version.DEFAULT_VERSION)


@pytest.mark.parametrize("action, expected", [("patch", "2.3.5"), ("minor", "2.4.0"), ("major", "3.0.0")])
def test_bump(action, expected):
    assert _load_bump().bump(action, "2.3.4") == expected


def test_write_version_updates_both_files(tmp_path):
    bump = _load_bump()
    vfile = tmp_path / "VERSION"
    vpy = tmp_path / "ui" / "version.py"
    vpy.parent.mkdir()
    vfile.write_text("1.0.0\n", encoding="utf-8")
    vpy.write_text('x = 1\nDEFAULT_VERSION = "1.0.0"\n', encoding="utf-8")

    bump.write_version("1.0.1", vfile, vpy)

    assert vfile.read_text(encoding="utf-8") == "1.0.1\n"
    assert 'DEFAULT_VERSION = "1.0.1"' in vpy.read_text(encoding="utf-8")


def test_write_version_rejects_invalid_semver(tmp_path):
    with pytest.raises(ValueError):
        _load_bump().write_version("2.3", tmp_path / "VERSION", tmp_path / "v.py")
