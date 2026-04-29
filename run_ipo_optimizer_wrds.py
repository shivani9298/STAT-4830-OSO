#!/usr/bin/env python3
"""Compatibility shim that re-exports scripts/run_ipo_optimizer_wrds.py."""

from scripts.run_ipo_optimizer_wrds import *  # noqa: F401,F403
from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "run_ipo_optimizer_wrds.py"
    runpy.run_path(str(target), run_name="__main__")
