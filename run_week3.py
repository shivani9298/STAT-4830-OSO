#!/usr/bin/env python3
"""Compatibility wrapper for scripts/run_week3.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "run_week3.py"
    runpy.run_path(str(target), run_name="__main__")
