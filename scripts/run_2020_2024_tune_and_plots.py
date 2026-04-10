#!/usr/bin/env python3
"""Backward-compatible wrapper: runs ``run_tune_experiment.py --preset 2020``."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_tune_experiment.py"), "--preset", "2020"],
        cwd=str(ROOT),
    )
    raise SystemExit(r.returncode)
