#!/usr/bin/env python3
"""
scripts/tag_versions.py
=======================
Crea la etiqueta git vX.Y.Z de cada versión que pasó por el archivo VERSION y
todavía no la tiene. La corre la CI al fusionar en master (regla del proyecto:
cada versión lleva su etiqueta), así no depende de que alguien la cree a mano.

Cada versión vieja se etiqueta en el último commit que cambió VERSION a ese
valor; la versión actual, en la punta de la rama (el commit del merge).

Uso:
    python scripts/tag_versions.py [--ref HEAD] [--push] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _git(*args: str, cwd: Path = ROOT_DIR) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True).stdout


def version_commits(ref: str = "HEAD", cwd: Path = ROOT_DIR) -> Dict[str, str]:
    """{versión: commit}: para cada versión, el último commit de *ref* que cambió
    VERSION a ese valor; para la versión actual, la punta de *ref*."""
    commits = _git("rev-list", "--reverse", "--topo-order", ref, "--", "VERSION", cwd=cwd).split()
    tip = _git("rev-parse", ref, cwd=cwd).strip()
    found: Dict[str, str] = {}
    for sha in commits:
        version = _git("show", f"{sha}:VERSION", cwd=cwd).strip()
        if _SEMVER_RE.match(version):
            found[version] = sha
    current = _git("show", f"{tip}:VERSION", cwd=cwd).strip()
    if _SEMVER_RE.match(current):
        found[current] = tip
    return found


def missing_tags(ref: str = "HEAD", cwd: Path = ROOT_DIR) -> List[Tuple[str, str]]:
    """[(etiqueta, commit)] de las versiones sin etiqueta, de la más vieja a la más nueva."""
    existing = set(_git("tag", "--list", "v*", cwd=cwd).split())
    return [(f"v{v}", sha) for v, sha in version_commits(ref, cwd).items() if f"v{v}" not in existing]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--push", action="store_true", help="Sube las etiquetas creadas a origin")
    parser.add_argument("--dry-run", action="store_true", help="Solo muestra qué etiquetas faltan")
    args = parser.parse_args()

    todo = missing_tags(args.ref)
    if not todo:
        print("Todas las versiones ya tienen etiqueta.")
        return 0
    for tag, sha in todo:
        print(f"{tag} -> {sha[:7]}")
        if not args.dry_run:
            _git("tag", "-a", tag, sha, "-m", f"Release {tag}")
    if args.push and not args.dry_run:
        _git("push", "origin", *[tag for tag, _ in todo])
        print("Etiquetas subidas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
