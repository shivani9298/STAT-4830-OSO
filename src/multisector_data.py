"""
Load market + sector-sleeve IPO returns for the global multi-sector allocator.

Builds one cap-weighted IPO index per GICS-style sector (from SDC when available).
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from .data_layer import add_optional_features
from .wrds_data import (
    load_ipo_data_from_sdc_wrds,
    load_market_returns_wrds,
    load_sdc_ipo_dates_wrds,
    load_sdc_new_deals_wrds,
    load_sp500_dow_market_returns_wrds,
    load_vix_wrds,
)


def _detect_ticker_column(df: pd.DataFrame) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for key in ("ticker", "ticker_symbol", "symbol", "tic", "iss_ticker", "primary_ticker"):
        if key in cols:
            return cols[key]
    return None


def _detect_gics_column(df: pd.DataFrame) -> Optional[str]:
    """Broad industry labels (not SIC descriptions — those explode into hundreds of sleeves)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for key in ("gics_sector", "gicssectordesc", "gics_sector_desc", "sector"):
        if key in cols_lower:
            return cols_lower[key]
    return None


def _detect_sic_code_column(df: pd.DataFrame) -> Optional[str]:
    """Numeric SIC (WRDS column names vary: sic, siccd, sic4, ...)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for key in ("sic", "sic_code", "siccd", "sic4", "compsic", "issuer_sic", "prim_sic"):
        if key in cols_lower:
            return cols_lower[key]
    for c in df.columns:
        cl = c.lower()
        if "desc" in cl or "name" in cl or "text" in cl:
            continue
        if cl in ("sic", "siccd") or (len(cl) <= 8 and "sic" in cl and "sicdesc" not in cl):
            return c
    return None


def _coarse_sector_from_sic(sic_raw: object) -> str:
    """Map 4-digit SIC to ~11 major groups (dense panels for training)."""
    if sic_raw is None or (isinstance(sic_raw, float) and np.isnan(sic_raw)):
        return "Unknown"
    try:
        s = int(float(str(sic_raw).strip()))
    except (TypeError, ValueError):
        return "Unknown"
    if s <= 0:
        return "Unknown"
    d = s // 100
    if d <= 9:
        return "Agriculture"
    if d <= 14:
        return "Mining"
    if d <= 17:
        return "Construction"
    if d <= 39:
        return "Manufacturing"
    if d <= 49:
        return "Transportation_Utilities"
    if d <= 51:
        return "Wholesale"
    if d <= 59:
        return "Retail"
    if d <= 67:
        return "Finance_RealEstate"
    if d <= 89:
        return "Services"
    return "Public_Admin"


def _clean_sector_label(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "Unknown"
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return "Unknown"
    s = re.sub(r"\s+", " ", s)
    return s[:80]


def _slug_for_column(label: str) -> str:
    """Safe fragment for CSV column names (match prior multisector outputs)."""
    t = re.sub(r"[^\w\s-]", "", label)
    t = re.sub(r"[\s_]+", "_", t).strip("_")
    return t or "Unknown"


def prepare_multisector_data(
    conn,
    start: str,
    end: str,
    holding_days: int = 180,
    min_tickers_per_sector: int = 1,
) -> dict:
    """
    Same WRDS load path as run_ipo_optimizer_wrds.prepare_data, plus one IPO sleeve per sector.

    Returns:
        df: features + market_return + sector_ret_* columns
        feature_cols: list of columns used as model inputs
        sector_labels: human-readable sector names (for export)
        sector_ret_cols: column names for sector return columns in df
        sector_portfolios: True
    """
    from scripts.run_ipo_optimizer_wrds import build_ipo_index_mcap  # local import avoids cycle at module load

    ipo_csv = load_ipo_data_from_sdc_wrds(
        conn, start=start, end=end, library="sdc", price_source="crsp"
    )
    print(f"IPO data from SDC + CRSP: {len(ipo_csv)} rows, {ipo_csv['tic'].nunique()} tickers")

    ipo_csv["datadate"] = pd.to_datetime(ipo_csv["datadate"])
    ipo_csv = ipo_csv.drop_duplicates(subset=["tic", "datadate"], keep="first")

    prices_ipo = ipo_csv.pivot_table(index="datadate", columns="tic", values="prccd")
    prices_ipo.index = pd.to_datetime(prices_ipo.index).normalize()

    ipo_dates = load_sdc_ipo_dates_wrds(conn, start=start, end=end, library="sdc")
    ipo_df = ipo_dates[ipo_dates["ticker"].isin(prices_ipo.columns)].copy()
    ipo_df = ipo_df.sort_values("ipo_date").reset_index(drop=True)

    start_d = prices_ipo.index.min().strftime("%Y-%m-%d")
    end_d = prices_ipo.index.max().strftime("%Y-%m-%d")
    print(f"IPO tickers: {len(ipo_df)}, Date range: {start_d} to {end_d}")

    market_end = max(end_d, end) if end_d else end
    market_ret = load_sp500_dow_market_returns_wrds(
        conn, start=start_d, end=market_end, w_sp500=0.82, w_dow=0.18
    )
    market_ret = market_ret.reindex(prices_ipo.index).dropna()
    if len(market_ret) < 50:
        market_ret = load_market_returns_wrds(conn, start=start_d, end=end_d)
        market_ret = market_ret.reindex(prices_ipo.index).dropna()
    if len(market_ret) < 50:
        raise RuntimeError("Insufficient market return data from CRSP.")

    ipo_tickers = ipo_df["ticker"].tolist()
    shares_outstanding: dict[str, float] = {}
    if "shrout" in ipo_csv.columns:
        last_shrout = ipo_csv.dropna(subset=["shrout"]).sort_values("datadate").groupby("tic")["shrout"].last()
        for tic, s in last_shrout.items():
            if s and s > 0:
                shares_outstanding[str(tic)] = float(s) * 1000
    elif "gvkey" in ipo_csv.columns:
        gvkeys = ipo_csv[["tic", "gvkey"]].drop_duplicates()
        gvkey_list = "','".join(gvkeys["gvkey"].astype(str).str.zfill(6).unique().tolist())
        shares_df = conn.raw_sql(
            f"""
            select gvkey, datadate, csho
            from comp.funda
            where gvkey in ('{gvkey_list}')
                and datadate >= '2020-01-01'
                and csho > 0
                and indfmt = 'INDL' and datafmt = 'STD'
            """,
            date_cols=["datadate"],
        )
        if len(shares_df) > 0:
            last_csho = shares_df.sort_values("datadate").groupby("gvkey")["csho"].last()
            gvkey_to_tic = dict(zip(gvkeys["gvkey"].astype(str).str.zfill(6), gvkeys["tic"]))
            for gvkey, csho in last_csho.items():
                t = gvkey_to_tic.get(str(gvkey).zfill(6))
                if t:
                    shares_outstanding[str(t)] = float(csho) * 1000

    for t in ipo_tickers:
        if t in prices_ipo.columns and t not in shares_outstanding:
            p = prices_ipo[t].dropna()
            if len(p) > 0 and p.iloc[-1] > 0:
                shares_outstanding[str(t)] = 1e6 / p.iloc[-1]

    prices = prices_ipo.copy().ffill().bfill()
    print(f"Market return days: {len(market_ret)}, Tickers with shares: {len(shares_outstanding)}")

    # Ticker -> coarse sector: prefer GICS-style column; else map numeric SIC to major group.
    ticker_to_sector: dict[str, str] = {}
    try:
        deals = load_sdc_new_deals_wrds(
            conn, start=start, end=end, library="sdc", ipodate_not_null=True, ipo_only=True
        )
        tcol = _detect_ticker_column(deals)
        gcol = _detect_gics_column(deals)
        siccol = _detect_sic_code_column(deals)
        if tcol:
            for _, row in deals.iterrows():
                tic = str(row[tcol]).strip().upper()
                if not tic or tic == "NAN":
                    continue
                label: Optional[str] = None
                if gcol is not None:
                    cg = _clean_sector_label(row[gcol])
                    if cg != "Unknown":
                        label = cg
                if label is None and siccol is not None:
                    label = _coarse_sector_from_sic(row[siccol])
                if label is None:
                    label = "Unknown"
                ticker_to_sector[tic] = label
            print(
                f"Sector mapping: GICS column={gcol!r}, SIC code column={siccol!r}, "
                f"{len(ticker_to_sector)} tickers"
            )
        else:
            print(f"Could not detect ticker column in SDC deals; columns: {list(deals.columns)[:20]}...")
    except Exception as e:
        print(f"SDC sector mapping skipped: {e}")

    # Count tickers per sector (among IPO names we trade)
    sector_counts: dict[str, list[str]] = {}
    for _, r in ipo_df.iterrows():
        t = str(r["ticker"]).strip().upper()
        lab = ticker_to_sector.get(t, "Unknown")
        sector_counts.setdefault(lab, []).append(t)

    ordered_sectors = sorted(sector_counts.keys(), key=lambda s: (-len(set(sector_counts[s])), s))
    sector_returns: dict[str, pd.Series] = {}

    for lab in ordered_sectors:
        names = list(dict.fromkeys(sector_counts[lab]))
        if len(names) < min_tickers_per_sector:
            continue
        sub = ipo_df[ipo_df["ticker"].isin(names)].copy()
        if len(sub) < 1:
            continue
        idx = build_ipo_index_mcap(prices, sub, shares_outstanding, holding_days=holding_days)
        ser = idx["ipo_ret"].rename(f"sector_{_slug_for_column(lab)}")
        sector_returns[lab] = ser
        print(f"  Sector {lab!r}: {len(names)} tickers, {ser.notna().sum()} valid return days")

    if not sector_returns:
        raise RuntimeError("No sector sleeves built; check SDC sector column and IPO filters.")

    # Assemble frame (concat once; fill missing sector days with 0 so dropna does not empty the panel)
    idx = market_ret.index
    sector_labels: list[str] = []
    sector_blocks: list[pd.Series] = []
    for lab, ser in sector_returns.items():
        slug = _slug_for_column(lab)
        col = f"sector_ret_{slug}"
        sector_labels.append(slug)
        sector_blocks.append(ser.reindex(idx).rename(col))

    df = pd.concat(
        [market_ret.reindex(idx).rename("market_return")] + sector_blocks,
        axis=1,
    )
    df["market_return"] = df["market_return"].clip(lower=-0.10, upper=0.10)
    sector_ret_cols = [c for c in df.columns if c.startswith("sector_ret_")]
    for c in sector_ret_cols:
        df[c] = df[c].clip(lower=-0.50, upper=0.50).fillna(0.0)
    df = df.sort_index().dropna(subset=["market_return"])

    vix_series = load_vix_wrds(conn, start=start_d, end=market_end)
    print(f"VIX data from CBOE: {len(vix_series)} days")
    df = add_optional_features(df, vix_series=vix_series)
    feature_cols = list(df.columns)

    return {
        "df": df,
        "feature_cols": feature_cols,
        "sector_labels": sector_labels,
        "sector_ret_cols": sector_ret_cols,
        "sector_portfolios": True,
    }
