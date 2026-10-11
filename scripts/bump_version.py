#!/usr/bin/env python3
"""
scripts/bump_version.py
=======================
Automated Semantic Versioning CLI for OCR-UIF.

Usage:
    python scripts/bump_version.py get
    python scripts/bump_version.py patch [--tag]
    python scripts/bump_version.py minor [--tag]
    python scripts/bump_version.py major [--tag]
    python scripts/bump_version.py set <new_version> [--tag]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT_DIR / "VERSION"


def get_current_version() -> str:
    if not VERSION_FILE.exists():
        return "0.0.0"
    content = VERSION_FILE.read_text(encoding="utf-8").strip()
    m = re.search(r"\d+\.\d+\.\d+", content)
    return m.group(0) if m else content


def parse_semver(v: str) -> tuple[int, int, int]:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", v)
    if not m:
        raise ValueError(f"Version '{v}' does not follow SemVer format 'MAJOR.MINOR.PATCH'")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def bump(action: str, current: str) -> str:
    major, minor, patch = parse_semver(current)
    if action == "major":
        return f"{major + 1}.0.0"
    elif action == "minor":
        return f"{major}.{minor + 1}.0"
    elif action == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Unknown action: {action}")


def write_version(new_v: str) -> None:
    # Ensure valid SemVer
    parse_semver(new_v)
    VERSION_FILE.write_text(f"{new_v}\n", encoding="utf-8")
    print(f"Updated {VERSION_FILE.name} to {new_v}")


def create_git_tag(new_v: str, message: str | None = None) -> bool:
    tag_name = f"v{new_v}"
    msg = message or f"Release {tag_name}"
    try:
        subprocess.run(["git", "tag", "-a", tag_name, "-m", msg], cwd=str(ROOT_DIR), check=True)
        print(f"Created Git tag: {tag_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not create Git tag '{tag_name}': {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR-UIF Semantic Versioning Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # get
    subparsers.add_parser("get", help="Print current SemVer")

    # bump: major, minor, patch
    p_patch = subparsers.add_parser("patch", help="Increment PATCH version (X.Y.Z+1)")
    p_patch.add_argument("--tag", action="store_true", help="Create git tag vX.Y.Z")

    p_minor = subparsers.add_parser("minor", help="Increment MINOR version (X.Y+1.0)")
    p_minor.add_argument("--tag", action="store_true", help="Create git tag vX.Y.Z")

    p_major = subparsers.add_parser("major", help="Increment MAJOR version (X+1.0.0)")
    p_major.add_argument("--tag", action="store_true", help="Create git tag vX.Y.Z")

    # set
    p_set = subparsers.add_parser("set", help="Explicitly set version")
    p_set.add_argument("version", help="New SemVer string (e.g. 2.2.0)")
    p_set.add_argument("--tag", action="store_true", help="Create git tag vX.Y.Z")

    args = parser.parse_args()
    curr_v = get_current_version()

    if args.command == "get":
        print(curr_v)
        return

    if args.command in ("patch", "minor", "major"):
        next_v = bump(args.command, curr_v)
    elif args.command == "set":
        next_v = args.version
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Current version : {curr_v}")
    print(f"New version     : {next_v}")
    write_version(next_v)

    if getattr(args, "tag", False):
        create_git_tag(next_v)


if __name__ == "__main__":
    main()
