"""
ui/version.py
=============
Automated Semantic Versioning (SemVer) module for OCR-UIF.

Resolves the project version from:
1. Bundled 'VERSION' file (in PyInstaller frozen executable or local root)
2. Git repository tags/describe (when running from source in a git tree)
3. Fallback default defined herein
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_VERSION = "2.2.0"

_SEMVER_REGEX = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?"
    r"(?:\+(?P<buildmetadata>[0-9A-Za-z.-]+))?$"
)


def _find_version_file() -> Path | None:
    """Locate the VERSION file across development and frozen environments."""
    # 1. PyInstaller extraction directory
    if hasattr(sys, "_MEIPASS"):
        meipass_p = Path(sys._MEIPASS) / "VERSION"
        if meipass_p.is_file():
            return meipass_p

    # 2. Executable parent directory (portable distribution)
    try:
        exe_dir = Path(sys.executable).parent / "VERSION"
        if exe_dir.is_file():
            return exe_dir
    except Exception:
        pass

    # 3. Project root (relative to ui/version.py)
    root_p = Path(__file__).resolve().parent.parent / "VERSION"
    if root_p.is_file():
        return root_p

    # 4. ui/ directory fallback
    ui_p = Path(__file__).resolve().parent / "VERSION"
    if ui_p.is_file():
        return ui_p

    return None


def _get_git_version() -> str | None:
    """Attempt to resolve the latest version tag from Git."""
    if getattr(sys, "frozen", False):
        return None
    try:
        root_dir = Path(__file__).resolve().parent.parent
        git_dir = root_dir / ".git"
        if not git_dir.exists():
            return None

        # Try exact tag or describe
        res = subprocess.run(
            ["git", "describe", "--tags", "--match", "v*"],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            raw = res.stdout.strip()
            # Strip leading 'v' if present (e.g., v2.2.0 -> 2.2.0)
            tag = raw.lstrip("v")
            # If format is 2.2.0-1-gabcdef, normalize or extract base
            base_match = re.match(r"^(\d+\.\d+\.\d+)", tag)
            if base_match:
                return base_match.group(1)
            return tag
    except Exception:
        pass
    return None


def resolve_semver() -> str:
    """Resolve the clean SemVer string (e.g. '2.2.0')."""
    # 1. Check VERSION file
    v_file = _find_version_file()
    if v_file:
        try:
            content = v_file.read_text(encoding="utf-8").strip()
            if content:
                # Extract first semantic version matching X.Y.Z
                m = re.search(r"\d+\.\d+\.\d+", content)
                if m:
                    return m.group(0)
                return content
        except Exception:
            pass

    # 2. Check Git tag
    git_v = _get_git_version()
    if git_v:
        m = re.search(r"\d+\.\d+\.\d+", git_v)
        if m:
            return m.group(0)

    return DEFAULT_VERSION


def parse_semver(v_str: str) -> Tuple[int, int, int]:
    """Parse 'X.Y.Z' into (major, minor, patch) integers."""
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", v_str)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (2, 2, 0)


def format_ui_version(semver_str: str) -> str:
    """
    Format standard SemVer for the OCR-UIF interface banner.
    e.g. '2.2.0' -> 'demo - v 2.2' (or 'demo - v 2.2.1' for patch releases).
    """
    major, minor, patch = parse_semver(semver_str)
    if patch == 0:
        return f"demo - v {major}.{minor}"
    return f"demo - v {major}.{minor}.{patch}"


# Global exports
__version__: str = resolve_semver()
APP_VERSION: str = format_ui_version(__version__)


def get_version_info() -> Dict[str, str | bool]:
    """Return comprehensive version and environment metadata."""
    major, minor, patch = parse_semver(__version__)
    return {
        "semver": __version__,
        "major": major,
        "minor": minor,
        "patch": patch,
        "app_version": APP_VERSION,
        "is_frozen": getattr(sys, "frozen", False),
    }
