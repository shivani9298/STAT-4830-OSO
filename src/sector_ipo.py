"""
Map IPO tickers to Yahoo Finance sectors for grouping (healthcare, technology, etc.).

Results are cached on disk to avoid repeated metadata requests.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


def sanitize_sector_label(name: str) -> str:
    """File-safe column suffix from sector string."""
    s = (name or "Unknown").strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return s.strip("_") or "Unknown"


def fetch_ticker_sectors(
    tickers: list[str],
    *,
    cache_path: str | Path | None = None,
    throttle_sec: float = 0.05,
    verbose: bool = True,
) -> pd.Series:
    """
    Return Series index=ticker, value=sector name from yfinance ``info['sector']``.

    Unknown / errors -> \"Unknown\". Merges with existing cache if ``cache_path`` is set.
    """
    if yf is None:
        raise ImportError("yfinance is required for sector mapping: pip install yfinance")
    cache_path = Path(cache_path) if cache_path else None
    tickers = sorted(set(t.upper().replace(".", "-") for t in tickers if t))

    existing: dict[str, str] = {}
    if cache_path and cache_path.exists():
        prev = pd.read_csv(cache_path)
        if "ticker" in prev.columns and "sector" in prev.columns:
            for _, row in prev.iterrows():
                existing[str(row["ticker"]).upper()] = str(row["sector"])

    to_fetch = [t for t in tickers if t not in existing]
    if verbose and to_fetch:
        print(f"[IPO] Fetching Yahoo sector for {len(to_fetch)} tickers (cached: {len(existing)})...", flush=True)

    for i, tic in enumerate(to_fetch):
        sector = "Unknown"
        try:
            info = yf.Ticker(tic).info
            if isinstance(info, dict):
                s = info.get("sector")
                if s and isinstance(s, str) and s.strip():
                    sector = s.strip()
        except Exception:
            sector = "Unknown"
        existing[tic] = sector
        if throttle_sec > 0:
            time.sleep(throttle_sec)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [IPO] sector fetch {i + 1}/{len(to_fetch)}", flush=True)

    out = pd.Series({t: existing.get(t, "Unknown") for t in tickers}, name="sector")
    out.index.name = "ticker"

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": out.index.astype(str), "sector": out.values}).to_csv(
            cache_path, index=False
        )
    return out


def group_tickers_by_sector(
    tickers: list[str],
    sector_series: pd.Series,
    *,
    min_names: int,
) -> dict[str, list[str]]:
    """
    Coarse groups: one group per distinct sector label with count >= min_names.
    Tickers not in series get Unknown; Unknown is kept only if size >= min_names.
    """
    groups: dict[str, list[str]] = {}
    upper = {str(i).upper(): str(v) for i, v in sector_series.items()}
    for t in tickers:
        tu = str(t).upper().replace(".", "-")
        sec = upper.get(tu, "Unknown")
        groups.setdefault(sec, []).append(t)
    return {k: v for k, v in groups.items() if len(v) >= min_names}


def sector_column_name(sector_label: str) -> str:
    return f"ipo_sector_{sanitize_sector_label(sector_label)}"
