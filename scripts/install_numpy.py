#!/usr/bin/env python3
"""
Install NumPy into the current Python environment (matches requirements.txt pin).

Usage (from repo root):
  python scripts/install_numpy.py
"""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "-m", "pip", "install", "numpy>=1.26"]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
