#!/usr/bin/env python3
"""
Run full IPO optimizer (2010–2024) with transformer + GICS sectors + local/ipo_optimizer_config.json,
then regenerate ablation figures and metrics tables.

Environment (set before import if needed):
  IPO_MODEL_TYPE=transformer
  IPO_SECTOR_SOURCE=compustat
  IPO_EXPORT_ATTENTION=1   (optional)

Usage:
  python scripts/TRANSFORMER_run_full_gics_and_reports.py

Requires ``local/ipo_optimizer_config.json`` (e.g. batch_size 32, hidden_size 64, lambda_vol 1.0).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main() -> int:
    os.chdir(ROOT)
    os.environ.setdefault("IPO_MODEL_TYPE", "transformer")
    os.environ.setdefault("IPO_SECTOR_SOURCE", "compustat")
    os.environ.setdefault("IPO_EXPORT_ATTENTION", "1")

    local_cfg = ROOT / "local" / "ipo_optimizer_config.json"
    if not local_cfg.is_file():
        print(
            f"Missing {local_cfg}. Create it with batch_size, hidden_size, lambda_vol overrides.",
            file=sys.stderr,
        )
        return 1

    print("[pipeline] Running run_ipo_optimizer_wrds.main() ...", flush=True)
    import run_ipo_optimizer_wrds as runner

    code = runner.main()
    if code != 0:
        return code

    py = sys.executable
    post = [
        [py, str(ROOT / "scripts" / "TRANSFORMER_plot_wrds_ablation_losses.py")],
        [py, str(ROOT / "scripts" / "TRANSFORMER_export_wrds_ablation_metrics_table.py")],
        [py, str(ROOT / "scripts" / "TRANSFORMER_plot_ablation_triplet_side_by_side.py")],
    ]
    for cmd in post:
        print(f"[pipeline] {' '.join(cmd)}", flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT))
    print("[pipeline] Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
