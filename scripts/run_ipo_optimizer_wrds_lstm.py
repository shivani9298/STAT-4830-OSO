#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_ipo_optimizer_wrds import main as run_main


if __name__ == "__main__":
    sys.argv.insert(1, "--model")
    sys.argv.insert(2, "lstm")
    raise SystemExit(run_main())

