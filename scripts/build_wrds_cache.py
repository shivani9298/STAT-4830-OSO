#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "run_ipo_optimizer_wrds.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build WRDS data cache without training.")
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--max-history", action="store_true")
    p.add_argument("--cache-dir", default=str(ROOT / "results" / "cache_wrds"))
    p.add_argument("--model", default="gru", choices=["gru", "lstm", "transformer"])
    p.add_argument("--ipo-index-method", default="fast", choices=["fast", "legacy"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(RUNNER),
        "--prepare-only",
        "--model",
        args.model,
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--cache-dir",
        args.cache_dir,
        "--ipo-index-method",
        args.ipo_index_method,
    ]
    if args.max_history:
        cmd.append("--max-history")
    print(f"[build-cache] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
