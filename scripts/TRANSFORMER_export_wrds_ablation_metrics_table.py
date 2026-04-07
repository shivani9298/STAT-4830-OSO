#!/usr/bin/env python3
"""
Build a performance metrics table (CSV + Markdown) from wrds_transformer_ablation JSON.

Usage:
  python scripts/TRANSFORMER_export_wrds_ablation_metrics_table.py
  python scripts/TRANSFORMER_export_wrds_ablation_metrics_table.py --json results/wrds_transformer_ablation_2020_2024.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{100.0 * x:.2f}%"


def _fmt_float(x: float | None, nd: int = 4) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json",
        type=Path,
        default=ROOT / "results" / "wrds_transformer_ablation_2020_2024.json",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=ROOT / "results" / "wrds_ablation_metrics_table.csv",
    )
    p.add_argument(
        "--md-out",
        type=Path,
        default=ROOT / "results" / "wrds_ablation_metrics_table.md",
    )
    args = p.parse_args()

    if not args.json.is_file():
        print(f"Missing {args.json}", file=sys.stderr)
        return 1

    with open(args.json, encoding="utf-8") as f:
        payload = json.load(f)

    runs = payload.get("runs") or []
    split = payload.get("split") or {}
    rows: list[dict[str, str]] = []

    for r in runs:
        name = str(r.get("name", ""))
        tm = r.get("test_metrics") or {}
        overrides = r.get("overrides")
        ov_str = json.dumps(overrides, sort_keys=True) if overrides else ""

        best_val = r.get("best_val_loss")
        test_loss = r.get("test_loss")
        epochs = r.get("epochs_ran")

        sharpe = ann_ret = tot_ret = max_dd = ann_vol = avg_to = None
        if "mean_across_sectors" in tm:
            m = tm["mean_across_sectors"]
            sharpe = m.get("sharpe_annualized")
            ann_ret = m.get("return_annualized")
            tot_ret = m.get("total_return")
            max_dd = m.get("max_drawdown")
        elif "portfolio" in tm:
            m = tm["portfolio"]
            sharpe = m.get("sharpe_annualized")
            ann_ret = m.get("return_annualized")
            tot_ret = m.get("total_return")
            max_dd = m.get("max_drawdown")
            ann_vol = m.get("volatility_annualized")
            avg_to = m.get("avg_turnover")

        rows.append(
            {
                "run": name,
                "best_val_loss": _fmt_float(best_val, 6) if best_val is not None else "",
                "test_loss": _fmt_float(test_loss, 6) if test_loss is not None else "",
                "epochs": str(epochs) if epochs is not None else "",
                "test_Sharpe_ann": _fmt_float(sharpe, 4) if sharpe is not None else "",
                "test_ann_return": _fmt_pct(ann_ret) if ann_ret is not None else "",
                "test_total_return": _fmt_pct(tot_ret) if tot_ret is not None else "",
                "test_max_drawdown": _fmt_pct(max_dd) if max_dd is not None else "",
                "test_ann_vol": _fmt_pct(ann_vol) if ann_vol is not None else "",
                "test_avg_turnover": _fmt_float(avg_to, 6) if avg_to is not None else "",
                "overrides": ov_str,
            }
        )

    fieldnames = [
        "run",
        "best_val_loss",
        "test_loss",
        "epochs",
        "test_Sharpe_ann",
        "test_ann_return",
        "test_total_return",
        "test_max_drawdown",
        "test_ann_vol",
        "test_avg_turnover",
        "overrides",
    ]

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Markdown table
    lines = [
        "# WRDS Transformer ablation — performance metrics",
        "",
        f"Data: `{split.get('data_start')}–{split.get('data_end')}` · "
        f"val `{split.get('val_start')}` · test `{split.get('test_start')}` · "
        f"sector_portfolios={split.get('sector_portfolios')}",
        "",
        "Test metrics are **out-of-sample** on the test split, using the best-val checkpoint. "
        "Sector runs use **mean across sector heads** (same as `mean_across_sectors` in JSON).",
        "",
        "| run | best val loss | test loss | epochs | test Sharpe (ann.) | test ann. return | test total return | test max DD | ann. vol (2-asset only) | avg turnover (2-asset only) | overrides |",
        "|-----|---------------|-----------|--------|--------------------|------------------|-------------------|-------------|-------------------------|------------------------------|-----------|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run"].replace("|", "\\|"),
                    row["best_val_loss"],
                    row["test_loss"],
                    row["epochs"],
                    row["test_Sharpe_ann"],
                    row["test_ann_return"],
                    row["test_total_return"],
                    row["test_max_drawdown"],
                    row["test_ann_vol"],
                    row["test_avg_turnover"],
                    row["overrides"].replace("|", "\\|"),
                ]
            )
            + " |"
        )

    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
