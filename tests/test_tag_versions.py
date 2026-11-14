"""scripts/tag_versions.py: etiqueta cada versión de VERSION que no la tenga."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git no disponible")


def _load():
    spec = importlib.util.spec_from_file_location("tag_versions", ROOT / "scripts" / "tag_versions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo, version=None, msg="cambio"):
    if version is not None:
        (repo / "VERSION").write_text(f"{version}\n", encoding="utf-8")
    (repo / "otro.txt").write_text(msg, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    return tmp_path


def test_missing_tags_one_per_version(repo):
    tv = _load()
    c1 = _commit(repo, "1.0.0", "v1")
    c101 = _commit(repo, "1.0.1", "v101")
    _commit(repo, None, "arreglo sin cambiar versión")
    _git(repo, "tag", "-a", "v1.0.0", c1, "-m", "x")
    c3 = _commit(repo, "1.1.0", "v110")
    c4 = _commit(repo, None, "merge simulado")

    # Versión vieja: el commit que la fijó; versión actual: la punta (el merge)
    assert tv.missing_tags("HEAD", repo) == [("v1.0.1", c101), ("v1.1.0", c4)]
    assert c3 != c4


def test_no_missing_tags(repo):
    tv = _load()
    c1 = _commit(repo, "2.0.0")
    _git(repo, "tag", "-a", "v2.0.0", c1, "-m", "x")
    assert tv.missing_tags("HEAD", repo) == []


def test_ignores_invalid_versions(repo):
    tv = _load()
    _commit(repo, "no-es-version")
    c = _commit(repo, "0.1.0")
    assert tv.missing_tags("HEAD", repo) == [("v0.1.0", c)]
