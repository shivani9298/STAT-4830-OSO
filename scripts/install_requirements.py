#!/usr/bin/env python3
"""
Install all packages from requirements.txt into the **same** Python that runs this script.

Use this instead of only install_numpy.py so pandas, torch, wrds, etc. are available.

Usage (from repo root):
  python scripts/install_requirements.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    req = ROOT / "requirements.txt"
    if not req.is_file():
        print(f"Missing {req}", file=sys.stderr)
        return 1
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
