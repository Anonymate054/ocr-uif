"""
run_test.py — CLI test runner for the OCR-UIF pipeline.
Runs the full pipeline on files/ → test/ without launching the Flet UI.
"""
import sys
import os
from pathlib import Path

# Set stdout to UTF-8 and replace characters for Windows compatibility
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Make sure the project root is on the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ui.pipeline import run_full_pipeline

INPUT_DIR  = str(ROOT / "files")
OUTPUT_DIR = str(ROOT / "test")


def progress(pct: float, msg: str) -> None:
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  [{bar}] {pct*100:5.1f}%  {msg}")
    sys.stdout.flush()


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  OCR-UIF Pipeline Test")
    print(f"  Input  : {INPUT_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    written = run_full_pipeline(INPUT_DIR, OUTPUT_DIR, progress)

    print(f"\n{'='*60}")
    print(f"  ✓ Done — {len(written)} CSV(s) written:")
    for p in written:
        print(f"    • {Path(p).name}")
    print(f"{'='*60}\n")
