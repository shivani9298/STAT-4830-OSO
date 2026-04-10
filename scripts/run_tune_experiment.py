#!/usr/bin/env python3
"""
Run ``notebooks/tune_hyperparameters_wrds.py`` with a chosen calendar sample and split.

**Presets** (``--preset``):

- ``2015`` — data **2015-01-01**–**2024-12-31**, val **2019-01-01**, test **2022-01-01** (same anchors as ``run_ipo_optimizer_wrds.py``).
- ``2020`` — data **2020-01-01**–**2024-12-31**, val **2022-01-01**, test **2023-08-01**.

Sets ``IPO_TUNE_*`` env vars; see the tuning notebook docstring. Quick mode runs one grid config unless ``--full-grid``.

Outputs: ``results/ipo_optimizer_best_config.json``, loss PNG + JSON history, copy under ``figures/old diagrams/``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PRESETS: dict[str, tuple[str, str, str, str]] = {
    "2015": ("2015-01-01", "2024-12-31", "2019-01-01", "2022-01-01"),
    "2020": ("2020-01-01", "2024-12-31", "2022-01-01", "2023-08-01"),
}


def main() -> int:
    p = argparse.ArgumentParser(description="WRDS tune + loss plots for IPO optimizer")
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="2015",
        help="Calendar sample + split anchors (default: 2015 = 2015–2024)",
    )
    p.add_argument("--full-grid", action="store_true", help="All grid configs (slow); default is quick (1 config)")
    args = p.parse_args()

    ds, de, vs, ts = PRESETS[args.preset]
    os.environ["IPO_TUNE_DATA_START"] = ds
    os.environ["IPO_TUNE_DATA_END"] = de
    os.environ["IPO_TUNE_VAL_START"] = vs
    os.environ["IPO_TUNE_TEST_START"] = ts
    if args.full_grid:
        os.environ["IPO_TUNE_QUICK"] = "0"
    else:
        os.environ["IPO_TUNE_QUICK"] = "1"

    tune_script = ROOT / "notebooks" / "tune_hyperparameters_wrds.py"
    print(f"preset={args.preset}  data={ds}..{de}  val={vs}  test={ts}", flush=True)
    print(f"IPO_TUNE_QUICK={os.environ['IPO_TUNE_QUICK']}", flush=True)
    print(f"Running: {tune_script}", flush=True)

    r = subprocess.run([sys.executable, str(tune_script)], cwd=str(ROOT))
    return int(r.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
