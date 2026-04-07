#!/usr/bin/env python3
"""
Generate **only** the CCM sector cache CSV (no model training).

Workflow (see ``docs/SECTOR_CCM_WORKFLOW.md``):

1. Load SDC New Deals IPOs + CRSP daily prices for ``--start``/``--end`` (defaults
   **2010-01-01**–**2024-12-31**, aligned with ``run_ipo_optimizer_wrds``).
2. Keep IPOs whose **``ipo_date``** falls in ``[start, end]`` (offer-date filter).
3. Attach ``first_crsp_date`` = first date with a non-null CRSP price per ticker.
4. Call ``fetch_ticker_sectors_ccm_chain`` → ``enrich_ipo_with_gics``
   (``stocknames`` / ``dsenames`` → CCM → Compustat GICS).
5. Write ``results/ticker_sector_cache_ccm.csv``.

Use in training: ``set IPO_SECTOR_SOURCE=ccm`` then run ``run_ipo_optimizer_wrds.py``.

Usage (repo root, WRDS credentials in env or .env)::

  python scripts/generate_sector_cache_ccm.py
  python scripts/generate_sector_cache_ccm.py --start 2015-01-01 --end 2020-12-31
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.wrds_data import close_wrds_connection, get_connection, load_ipo_data_from_sdc_wrds, load_sdc_ipo_dates_wrds
from src.sector_ipo import fetch_ticker_sectors_ccm_chain

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build ticker_sector_cache_ccm.csv for IPOs with ipo_date in [start, end].",
    )
    p.add_argument(
        "--start",
        default=os.environ.get("IPO_START", START_DATE),
        help="Inclusive start of IPO offer-date window and WRDS data pull (default 2010-01-01).",
    )
    p.add_argument(
        "--end",
        default=os.environ.get("IPO_END", END_DATE),
        help="Inclusive end of IPO offer-date window and WRDS data pull (default 2024-12-31).",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore existing ticker_sector_cache_ccm.csv and re-label all tickers from WRDS.",
    )
    args = p.parse_args()

    cache = ROOT / "results" / "ticker_sector_cache_ccm.csv"
    print(
        f"[sectors] WRDS pull + IPO offer-date window: {args.start} – {args.end}",
        flush=True,
    )
    print("[sectors] Connecting to WRDS...", flush=True)
    conn = get_connection()

    print("[sectors] Loading SDC + CRSP prices (same as IPO optimizer)...", flush=True)
    ipo_csv = load_ipo_data_from_sdc_wrds(
        conn, start=args.start, end=args.end, library="sdc", price_source="crsp"
    )
    ipo_csv["datadate"] = pd.to_datetime(ipo_csv["datadate"])
    ipo_csv = ipo_csv.drop_duplicates(subset=["tic", "datadate"], keep="first")
    prices_ipo = ipo_csv.pivot_table(index="datadate", columns="tic", values="prccd")

    print("[sectors] Loading SDC IPO dates...", flush=True)
    ipo_dates = load_sdc_ipo_dates_wrds(conn, start=args.start, end=args.end, library="sdc")
    ipo_df = ipo_dates[ipo_dates["ticker"].isin(prices_ipo.columns)].copy()
    ipo_df["ipo_date"] = pd.to_datetime(ipo_df["ipo_date"]).dt.normalize()
    ts, te = pd.Timestamp(args.start), pd.Timestamp(args.end)
    ipo_df = ipo_df[(ipo_df["ipo_date"] >= ts) & (ipo_df["ipo_date"] <= te)].copy()
    ipo_df = ipo_df.sort_values("ipo_date").reset_index(drop=True)
    if ipo_df.empty:
        print("[sectors] No IPOs in offer-date window; exiting.", flush=True)
        close_wrds_connection(conn)
        return 1
    ipo_tickers = ipo_df["ticker"].tolist()
    _fd: dict[str, pd.Timestamp] = {}
    for _tic in ipo_df["ticker"]:
        if _tic in prices_ipo.columns:
            _s = prices_ipo[_tic].dropna()
            if len(_s) > 0:
                _fd[_tic] = pd.Timestamp(_s.index.min()).normalize()
    ipo_df["first_crsp_date"] = ipo_df["ticker"].map(_fd)
    print(f"[sectors] IPOs with prices: {len(ipo_df)} tickers", flush=True)

    print("[sectors] Running CCM GICS chain (WRDS queries; may take many minutes)...", flush=True)
    sec = fetch_ticker_sectors_ccm_chain(
        conn,
        ipo_df[["ticker", "ipo_date", "first_crsp_date"]],
        ipo_tickers,
        cache_path=cache,
        verbose=True,
        refresh=args.refresh,
    )
    close_wrds_connection(conn)

    print(f"[sectors] Wrote {cache}", flush=True)
    print(f"[sectors] Labeled {len(sec)} tickers; unique sectors: {sec.nunique()}", flush=True)
    print(sec.value_counts().head(15).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
