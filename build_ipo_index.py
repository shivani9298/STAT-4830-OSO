#!/usr/bin/env python3
"""Compatibility wrapper for scripts/build_ipo_index.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "build_ipo_index.py"
    runpy.run_path(str(target), run_name="__main__")
