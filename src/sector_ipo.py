"""
Map IPO tickers to **sector groups** for multi-head portfolios.

- **Compustat GICS** (default when using WRDS): ``fetch_ticker_sectors_compustat`` reads
  ``comp.company.gsector`` via WRDS and maps to GICS sector names.
- **CCM chain** (recommended for identifier consistency): ``fetch_ticker_sectors_ccm_chain``
  uses ``ticker``, ``ipo_date``, optional ``first_crsp_date`` → ``match_date`` →
  ``stocknames`` / ``dsenames`` → CCM ``gvkey`` → Compustat GICS
  (see ``src/wrds_ipo_gics_enrichment.py``, ``docs/SECTOR_CCM_WORKFLOW.md``).
  Set ``IPO_SECTOR_SOURCE=ccm`` or ``wrds_chain``.
- **Yahoo Finance** (optional): ``fetch_ticker_sectors`` uses ``info['sector']``.

Set env ``IPO_SECTOR_SOURCE`` to ``compustat`` (default), ``ccm`` / ``wrds_chain``,
or ``yfinance``. Results are cached on disk to avoid repeated queries.
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


def fetch_ticker_sectors_compustat(
    conn,
    tickers: list[str],
    *,
    cache_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.Series:
    """
    Return Series index=ticker, value=GICS sector name from Compustat ``comp.company``.

    Uses ``load_gics_sectors_for_tickers_wrds``. Tickers without a Compustat row or
    with null ``gsector`` are labeled ``\"Unknown\"``. Merges with existing CSV cache
    like :func:`fetch_ticker_sectors`.
    """
    from src.wrds_data import load_gics_sectors_for_tickers_wrds

    cache_path = Path(cache_path) if cache_path else None
    tickers = sorted(set(str(t).upper().replace(".", "-") for t in tickers if t))

    existing: dict[str, str] = {}
    if cache_path and cache_path.exists():
        prev = pd.read_csv(cache_path)
        if "ticker" in prev.columns and "sector" in prev.columns:
            for _, row in prev.iterrows():
                existing[str(row["ticker"]).upper()] = str(row["sector"])

    to_fetch = [t for t in tickers if t not in existing]
    if verbose and to_fetch:
        print(
            f"[IPO] Compustat GICS lookup for {len(to_fetch)} tickers "
            f"(cached: {len(existing)})...",
            flush=True,
        )

    if to_fetch:
        df = load_gics_sectors_for_tickers_wrds(conn, to_fetch)
        found = set()
        if len(df) > 0:
            for _, row in df.iterrows():
                tic = str(row["ticker"]).upper()
                sec = str(row["sector"])
                existing[tic] = sec
                found.add(tic)
        for tic in to_fetch:
            if tic not in found:
                existing[tic] = "Unknown"

    out = pd.Series({t: existing.get(t, "Unknown") for t in tickers}, name="sector")
    out.index.name = "ticker"

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": out.index.astype(str), "sector": out.values}).to_csv(
            cache_path, index=False
        )
    return out


def fetch_ticker_sectors_ccm_chain(
    conn,
    ipo_dates_df: pd.DataFrame,
    tickers: list[str],
    *,
    cache_path: str | Path | None = None,
    verbose: bool = True,
    refresh: bool = False,
) -> pd.Series:
    """
    Map each IPO ticker to a GICS sector label using the date-valid WRDS chain:
    ``stocknames`` → ``ccmxpf_linktable`` → ``comp.company`` (see
    ``src.wrds_ipo_gics_enrichment``).

    Parameters
    ----------
    conn : wrds.Connection
    ipo_dates_df : DataFrame
        Columns ``ticker`` and ``ipo_date`` (one row per ticker in sample; IPO date
        defines name/link validity).
    tickers : list of tickers to label (typically all IPO names in the model).
    cache_path : optional CSV with columns ``ticker``, ``sector`` (and optional
        ``permno``, ``gvkey``, ``gsector_raw`` for audit).
    refresh : if True, ignore an existing cache file and re-query WRDS for all tickers.

    Returns
    -------
    Series index=ticker, value=sector label (same convention as
    ``fetch_ticker_sectors_compustat``).
    """
    from src.wrds_data import gics_sector_name_from_code
    from src.wrds_ipo_gics_enrichment import enrich_ipo_with_gics

    cache_path = Path(cache_path) if cache_path else None
    tickers_u = [str(t).upper().replace(".", "-") for t in tickers if t]
    tickers_u = sorted(set(tickers_u))

    existing: dict[str, str] = {}
    if not refresh and cache_path and cache_path.exists():
        prev = pd.read_csv(cache_path)
        if "ticker" in prev.columns and "sector" in prev.columns:
            for _, row in prev.iterrows():
                existing[str(row["ticker"]).upper().replace(".", "-")] = str(row["sector"])

    to_fetch = [t for t in tickers_u if t not in existing]
    if verbose and to_fetch:
        print(
            f"[IPO] CCM GICS chain for {len(to_fetch)} tickers "
            f"(cached: {len(existing)})...",
            flush=True,
        )

    extra_rows: list[dict] = []
    if to_fetch:
        idf = ipo_dates_df.copy()
        idf["ticker_norm"] = idf["ticker"].map(
            lambda x: str(x).upper().replace(".", "-") if pd.notna(x) else ""
        )
        _cols = ["ticker", "ipo_date"]
        if "first_crsp_date" in idf.columns:
            _cols.append("first_crsp_date")
        sub = idf[idf["ticker_norm"].isin(to_fetch)][_cols].drop_duplicates(subset=["ticker"], keep="first")
        if sub.empty:
            for t in to_fetch:
                existing[t] = "Unknown"
        else:
            enriched, diag = enrich_ipo_with_gics(sub, conn, include_compustat_name=False)
            for _, r in enriched.iterrows():
                t = str(r["ticker"]).upper().replace(".", "-")
                if pd.notna(r.get("gsector")):
                    lab = gics_sector_name_from_code(r["gsector"])
                else:
                    lab = "Unknown"
                existing[t] = lab
                extra_rows.append(
                    {
                        "ticker": t,
                        "permno": r.get("permno"),
                        "gvkey": r.get("gvkey"),
                        "gsector_raw": r.get("gsector"),
                    }
                )
            for t in to_fetch:
                if t not in existing:
                    existing[t] = "Unknown"

    out = pd.Series({t: existing.get(t, "Unknown") for t in tickers_u}, name="sector")
    out.index.name = "ticker"

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame({"ticker": out.index.astype(str), "sector": out.values})
        if extra_rows:
            ex = pd.DataFrame(extra_rows).drop_duplicates(subset=["ticker"])
            df_out = df_out.merge(ex, on="ticker", how="left")
        df_out.to_csv(cache_path, index=False)
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
