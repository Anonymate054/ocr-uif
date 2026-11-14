"""scripts/build_exe.py: comando de flet pack sin rutas de una máquina."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("build_exe", ROOT / "scripts" / "build_exe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pairs(cmd, flag):
    return [cmd[i + 1] for i, c in enumerate(cmd) if c == flag]


def test_pack_command_bundles_what_the_app_reads(tmp_path):
    be = _load()
    cmd = be.pack_command(rapidocr_dir=tmp_path / "rapidocr_onnxruntime")
    data = _pairs(cmd, "--add-data")
    assert "VERSION:." in data and "models:models" in data and "ui/assets:ui/assets" in data
    assert f"{tmp_path / 'rapidocr_onnxruntime'}:rapidocr_onnxruntime" in data
    assert _pairs(cmd, "--hidden-import") == ["sklearn_crfsuite", "pycrfsuite"]
    assert "--onedir" not in cmd and cmd[-1] == "--yes"


def test_pack_command_uses_version_file(tmp_path):
    be = _load()
    v = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    cmd = be.pack_command(onedir=True, rapidocr_dir=tmp_path)
    assert _pairs(cmd, "--product-version") == [v]
    assert _pairs(cmd, "--file-version") == [f"{v}.0"]
    assert "--onedir" in cmd


def test_bundled_data_exists():
    be = _load()
    for src, _ in be.DATA_FILES:
        assert (ROOT / src).exists(), src


def test_no_machine_specific_paths_in_build_files():
    for name in ("scripts/build_exe.py", ".github/workflows/build.yml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "C:\\\\Users" not in text and "C:\\Users" not in text and "miniconda" not in text.lower()


def test_patch_makes_binary_open_ignore_encoding(tmp_path):
    """flet-cli 1.1.2 abre el .exe con open(path, "ab", encoding=...) en Windows."""
    import types

    be = _load()
    fake_win_utils = types.SimpleNamespace()
    assert be.patch_flet_cli_windows(fake_win_utils) is True

    target = tmp_path / "x.bin"
    with fake_win_utils.open(target, "ab", encoding="utf-8") as fh:
        fh.write(b"overlay")
    with fake_win_utils.open(target, "r", encoding="utf-8") as fh:
        assert fh.read() == "overlay"
