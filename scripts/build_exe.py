#!/usr/bin/env python3
"""
scripts/build_exe.py
====================
Empaqueta la app con `flet pack` (PyInstaller) en dist/. Es el mismo comando
que corre la CI en Windows para publicar OCR-UIF.exe, sin rutas de una máquina
en particular: la carpeta de rapidocr_onnxruntime se toma del entorno actual.

Requisitos (además de requirements.txt):
    pip install -r requirements-build.txt

Uso:
    python scripts/build_exe.py            # un solo archivo (OCR-UIF.exe)
    python scripts/build_exe.py --onedir   # carpeta: arranca más rápido
    python scripts/build_exe.py --print    # solo muestra el comando
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import os
import shlex
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_NAME = "OCR-UIF"

# Archivos de datos que la app lee en tiempo de ejecución (origen, destino)
DATA_FILES = [("VERSION", "."), ("models", "models"), ("ui/assets", "ui/assets")]
# Módulos que PyInstaller no detecta solo (el CRF se carga con joblib)
HIDDEN_IMPORTS = ["sklearn_crfsuite", "pycrfsuite"]


def package_dir(name: str) -> Path:
    """Carpeta instalada de un paquete (p. ej. rapidocr_onnxruntime)."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"No está instalado el paquete {name}; corre pip install -r requirements.txt")
    return Path(spec.origin).parent


def version() -> str:
    return (ROOT_DIR / "VERSION").read_text(encoding="utf-8").strip()


def pack_command(onedir: bool = False, rapidocr_dir: Path | None = None) -> List[str]:
    """Argumentos de `flet pack` para empaquetar la app."""
    rapidocr_dir = rapidocr_dir or package_dir("rapidocr_onnxruntime")
    v = version()
    cmd = [sys.executable, "-m", "flet_cli.cli", "pack", "ui/app.py", "--name", APP_NAME,
           "--product-name", APP_NAME, "--product-version", v, "--file-version", f"{v}.0"]
    # RapidOCR busca config.yaml y sus modelos junto a su propio paquete
    for src, dst in [*DATA_FILES, (str(rapidocr_dir), "rapidocr_onnxruntime")]:
        cmd += ["--add-data", f"{src}:{dst}"]
    for module in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", module]
    if onedir:
        cmd.append("--onedir")
    cmd.append("--yes")
    return cmd


def _binary_safe_open(file, mode="r", *args, **kwargs):
    """open() que ignora encoding en modo binario (como debería)."""
    if "b" in mode:
        kwargs.pop("encoding", None)
    return builtins.open(file, mode, *args, **kwargs)


def patch_flet_cli_windows(win_utils=None) -> bool:
    """flet-cli 1.1.2 en Windows abre el .exe con open(path, "ab", encoding=...)
    al reponer el overlay tras escribir la información de versión, y eso lanza
    ValueError. Se sustituye open() solo dentro de ese módulo. Devuelve True si
    se aplicó."""
    if win_utils is None:
        if sys.platform != "win32":
            return False
        import flet_cli.__pyinstaller.win_utils as win_utils
    win_utils.open = _binary_safe_open
    return True


def run_pack(cmd: List[str]) -> int:
    """Corre flet pack en este mismo proceso, con el parche de Windows."""
    from flet_cli import cli

    patch_flet_cli_windows()
    os.chdir(ROOT_DIR)
    sys.argv = ["flet", *cmd[cmd.index("pack"):]]
    try:
        cli.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Empaqueta OCR-UIF con flet pack")
    parser.add_argument("--onedir", action="store_true", help="Carpeta en vez de un solo archivo")
    parser.add_argument("--print", action="store_true", dest="only_print", help="Solo muestra el comando")
    args = parser.parse_args()

    cmd = pack_command(args.onedir)
    print(" ".join(shlex.quote(c) for c in cmd))
    if args.only_print:
        return 0
    return run_pack(cmd)


if __name__ == "__main__":
    sys.exit(main())
