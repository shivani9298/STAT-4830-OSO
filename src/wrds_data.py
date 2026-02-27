"""
Pull stock price and return data from WRDS (CRSP) and SDC New Deals.

Requires a WRDS account and, for remote access, two-factor authentication.
Set WRDS_USERNAME in the environment or pass it to get_connection().

SDC Platinum (LSEG) New Deals/New Issues: library is typically 'lseg';
table names vary by subscription (e.g. globalnewiss, new_issues, sdc_new_issues).
Use list_wrds_libraries() and list_wrds_tables(conn, 'lseg') to discover.
"""
from __future__ import annotations

import os
import pandas as pd
from typing import Optional

try:
    import wrds
except ImportError:
    wrds = None


def get_connection(
    wrds_username: Optional[str] = None,
    wrds_password: Optional[str] = None,
):
    """
    Connect to WRDS.

    Args:
        wrds_username: WRDS username. If None, uses env WRDS_USERNAME or prompts.
        wrds_password: WRDS password. If None, uses env WRDS_PASSWORD or prompts.

    Returns:
        wrds.Connection

    Raises:
        ImportError: if the wrds package is not installed.
    """
    if wrds is None:
        raise ImportError("wrds is required. Install with: uv add wrds  or  pip install wrds")
    username = wrds_username or os.environ.get("WRDS_USERNAME")
    password = wrds_password or os.environ.get("WRDS_PASSWORD")
    if username and password:
        return wrds.Connection(wrds_username=username, wrds_password=password)
    return wrds.Connection(wrds_username=username)


def load_market_returns_wrds(
    conn,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    index: str = "vwretd",
) -> pd.Series:
    """
    Load daily market returns from CRSP index file (crsp.dsi).

    Args:
        conn: wrds.Connection from get_connection().
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD', or None for latest.
        index: Return column: 'vwretd' (value-weighted), 'ewretd' (equal-weighted), etc.

    Returns:
        Series of daily returns with datetime index, name 'market_return'.
    """
    end_clause = f"AND date <= '{end}'" if end else ""
    sql = f"""
        SELECT date, {index} AS ret
        FROM crsp.dsi
        WHERE date >= '{start}' {end_clause}
        ORDER BY date
    """
    df = conn.raw_sql(sql, date_cols=["date"])
    out = df.set_index("date")["ret"].sort_index()
    out.name = "market_return"
    return out


def load_sp500_dow_market_returns_wrds(
    conn,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    w_sp500: float = 0.82,
    w_dow: float = 0.18,
) -> pd.Series:
    """
    Load market returns as a market-cap weighted portfolio of S&P 500 and Dow Jones.

    Uses SPY (S&P 500 proxy) and DIA (Dow Jones proxy) from CRSP for split-adjusted returns.
    Default weights: 82% S&P 500, 18% Dow (approximate index market-cap ratio).

    Args:
        conn: wrds.Connection from get_connection().
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD', or None for latest.
        w_sp500: Weight for S&P 500 (SPY).
        w_dow: Weight for Dow Jones (DIA).

    Returns:
        Series of daily returns with datetime index, name 'market_return'.
    """
    df = load_stock_returns_wrds(
        conn, start=start, end=end, tickers=["SPY", "DIA"]
    )
    if df.empty or df.shape[1] < 2:
        return pd.Series(dtype=float)
    # Normalize weights
    wt = w_sp500 + w_dow
    w_sp500, w_dow = w_sp500 / wt, w_dow / wt
    spy_col = "SPY" if "SPY" in df.columns else df.columns[0]
    dia_col = "DIA" if "DIA" in df.columns else df.columns[1]
    mask = df[spy_col].notna() | df[dia_col].notna()
    ret = df[spy_col].fillna(0) * w_sp500 + df[dia_col].fillna(0) * w_dow
    ret = ret.where(mask).dropna()
    ret.name = "market_return"
    return ret.sort_index()


def load_stock_prices_wrds(
    conn,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    permnos: Optional[list[int]] = None,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load daily stock prices (and returns) from CRSP daily stock file (crsp.dsf).

    Provide either permnos (CRSP permanent IDs) or tickers. If tickers are given,
    they are mapped to permno via crsp.dsenames (most recent name per ticker).

    Args:
        conn: wrds.Connection from get_connection().
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD', or None for latest.
        permnos: List of CRSP permno identifiers.
        tickers: List of ticker symbols (e.g. ['AAPL','MSFT']). Used only if permnos is None.

    Returns:
        DataFrame with columns: date, permno, prc, ret, vol, (ticker if tickers given).
        Index is date; one row per (date, permno) or long format with date index.
    """
    end_clause = f"AND a.date <= '{end}'" if end else ""
    if permnos is not None:
        permno_list = ",".join(str(p) for p in permnos)
        sql = f"""
            SELECT a.date, a.permno, a.prc, a.ret, a.vol
            FROM crsp.dsf AS a
            WHERE a.permno IN ({permno_list})
              AND a.date >= '{start}' {end_clause}
            ORDER BY a.date, a.permno
        """
        df = conn.raw_sql(sql, date_cols=["date"])
        return df
    if tickers is not None:
        # Resolve tickers to permnos via crsp.dsenames (ticker, namedt, nameendt)
        ticker_list = ",".join(f"'{t}'" for t in tickers)
        end_val = end or "2099-12-31"
        sql_permno = f"""
            SELECT permno, ticker
            FROM crsp.dsenames
            WHERE ticker IN ({ticker_list})
              AND namedt <= '{end_val}'
              AND (nameendt >= '{start}' OR nameendt IS NULL)
        """
        try:
            names = conn.raw_sql(sql_permno, date_cols=[])
        except Exception:
            # Some WRDS/CRSP use nameenddt instead of nameendt
            try:
                sql2 = f"""SELECT permno, ticker FROM crsp.dsenames
                    WHERE ticker IN ({ticker_list}) AND namedt <= '{end_val}'
                    AND (nameenddt >= '{start}' OR nameenddt IS NULL)"""
                names = conn.raw_sql(sql2, date_cols=[])
            except Exception:
                names = conn.raw_sql(
                    f"SELECT permno, ticker FROM crsp.dsenames WHERE ticker IN ({ticker_list})",
                    date_cols=[],
                )
        if names.empty:
            return pd.DataFrame()
        permnos = names["permno"].drop_duplicates().tolist()
        df = load_stock_prices_wrds(conn, start=start, end=end, permnos=permnos)
        # Merge back ticker
        df = df.merge(names[["permno", "ticker"]], on="permno", how="left")
        return df
    return pd.DataFrame()


def load_stock_returns_wrds(
    conn,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    permnos: Optional[list[int]] = None,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load daily returns from CRSP (crsp.dsf) as a wide DataFrame.

    Returns:
        DataFrame with date index and one column per permno/ticker (column names = permno or ticker).
        Values are daily returns. Drops rows where all returns are missing.
    """
    df = load_stock_prices_wrds(
        conn, start=start, end=end, permnos=permnos, tickers=tickers
    )
    if df.empty:
        return df
    if "ticker" in df.columns:
        pivot = df.pivot_table(index="date", columns="ticker", values="ret")
    else:
        pivot = df.pivot_table(index="date", columns="permno", values="ret")
    return pivot.dropna(how="all").sort_index()


def load_portfolio_returns_compustat_wrds(
    conn,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    tickers: Optional[list[str]] = None,
    value_weighted: bool = True,
) -> pd.Series:
    """
    Load daily portfolio returns from Compustat (comp.sec_dprc + comp.funda).

    Uses prccd (close price) and csho (shares, from funda) for value weighting.
    Falls back to equal weight if market-cap data is insufficient.

    Args:
        conn: wrds.Connection from get_connection().
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD', or None for latest.
        tickers: List of ticker symbols.
        value_weighted: If True, weight by market cap; else equal weight.

    Returns:
        Series of daily portfolio returns with datetime index, name 'market_return'.
    """
    if not tickers:
        return pd.Series(dtype=float)
    prices_df = load_compustat_daily_prices_wrds(
        conn, tickers=tickers, start=start, end=end
    )
    if prices_df.empty or prices_df["tic"].nunique() < 2:
        return pd.Series(dtype=float)
    pivot = prices_df.pivot_table(index="datadate", columns="tic", values="prccd")
    ret = pivot.pct_change()
    # Clip extreme daily returns (likely data errors); ±25% for large-cap names
    ret = ret.clip(lower=-0.25, upper=0.25)
    ret = ret.dropna(how="all")
    if ret.empty:
        return pd.Series(dtype=float)
    if not value_weighted:
        port = ret.mean(axis=1)
        port.name = "market_return"
        return port.sort_index()
    gvkeys = prices_df[["tic", "gvkey"]].drop_duplicates()
    gvkey_list = "','".join(gvkeys["gvkey"].astype(str).str.zfill(6).unique().tolist())
    csho_df = conn.raw_sql(
        f"""
        SELECT gvkey, datadate, csho
        FROM comp.funda
        WHERE gvkey IN ('{gvkey_list}')
          AND datadate >= '2018-01-01'
          AND csho > 0
          AND indfmt = 'INDL' AND datafmt = 'STD'
        """,
        date_cols=["datadate"],
    )
    if csho_df.empty:
        port = ret.mean(axis=1)
        port.name = "market_return"
        return port.sort_index()
    last_csho = csho_df.sort_values("datadate").groupby("gvkey")["csho"].last()
    gvkey_to_tic = dict(zip(gvkeys["gvkey"].astype(str).str.zfill(6), gvkeys["tic"]))
    mcap_dict = {
        gvkey_to_tic.get(str(g).zfill(6)): float(c) * 1000
        for g, c in last_csho.items()
        if gvkey_to_tic.get(str(g).zfill(6))
    }
    weights_list = []
    for d in ret.index:
        prcs = pivot.loc[d].dropna()
        if prcs.empty:
            continue
        mcaps = {}
        for t, p in prcs.items():
            csho = mcap_dict.get(t)
            if csho and p and p > 0:
                mcaps[t] = abs(float(p)) * csho
        total = sum(mcaps.values())
        if total > 0 and len(mcaps) >= 2:
            w = {t: m / total for t, m in mcaps.items()}
            weights_list.append((d, w))
    if len(weights_list) < 50:
        port = ret.mean(axis=1)
        port.name = "market_return"
        return port.sort_index()
    out = []
    for d, w in weights_list:
        r = ret.loc[d].fillna(0)
        wr = sum(w.get(t, 0) * r.get(t, 0) for t in w)
        out.append((d, wr))
    s = pd.Series(dict(out)).sort_index()
    s.name = "market_return"
    return s


def load_portfolio_returns_value_weighted_wrds(
    conn,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    tickers: Optional[list[str]] = None,
) -> pd.Series:
    """
    Load daily value-weighted (market-cap) portfolio returns from CRSP (crsp.dsf).

    Weights each stock by market cap (|prc| * shrout; CRSP shrout in thousands).
    Falls back to equal weight if market-cap data is missing.

    Args:
        conn: wrds.Connection from get_connection().
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD', or None for latest.
        tickers: List of ticker symbols.

    Returns:
        Series of daily value-weighted returns with datetime index, name 'market_return'.
    """
    if not tickers:
        return pd.Series(dtype=float)
    end_clause = f"AND a.date <= '{end}'" if end else ""
    ticker_list = ",".join(f"'{t}'" for t in tickers)
    end_val = end or "2099-12-31"
    sql_permno = f"""
        SELECT permno, ticker
        FROM crsp.dsenames
        WHERE ticker IN ({ticker_list})
          AND namedt <= '{end_val}'
          AND (nameendt >= '{start}' OR nameendt IS NULL)
    """
    try:
        names = conn.raw_sql(sql_permno, date_cols=[])
    except Exception:
        try:
            sql2 = f"""SELECT permno, ticker FROM crsp.dsenames
                WHERE ticker IN ({ticker_list}) AND namedt <= '{end_val}'
                AND (nameenddt >= '{start}' OR nameenddt IS NULL)"""
            names = conn.raw_sql(sql2, date_cols=[])
        except Exception:
            names = conn.raw_sql(
                f"SELECT permno, ticker FROM crsp.dsenames WHERE ticker IN ({ticker_list})",
                date_cols=[],
            )
    if names.empty:
        return pd.Series(dtype=float)
    permno_list = ",".join(str(p) for p in names["permno"].drop_duplicates().tolist())
    sql = f"""
        SELECT a.date, a.permno, a.prc, a.ret, a.shrout
        FROM crsp.dsf AS a
        WHERE a.permno IN ({permno_list})
          AND a.date >= '{start}' {end_clause}
          AND a.prc IS NOT NULL AND a.ret IS NOT NULL
        ORDER BY a.date
    """
    df = conn.raw_sql(sql, date_cols=["date"])
    if df.empty:
        return pd.Series(dtype=float)
    df = df.merge(names[["permno", "ticker"]], on="permno", how="left")
    df["mcap"] = df["prc"].abs() * df["shrout"].fillna(0)
    df = df[df["mcap"] > 0]
    if df.empty:
        return pd.Series(dtype=float)
    total_mcap = df.groupby("date")["mcap"].sum()
    df = df.merge(total_mcap.rename("total_mcap"), left_on="date", right_index=True)
    df["weight"] = df["mcap"] / df["total_mcap"]
    df["weighted_ret"] = df["weight"] * df["ret"]
    vw = df.groupby("date")["weighted_ret"].sum()
    vw.name = "market_return"
    return vw.sort_index()


def list_wrds_libraries(conn) -> list[str]:
    """
    List WRDS libraries (schemas) available to your account.

    Args:
        conn: wrds.Connection from get_connection().

    Returns:
        List of library names. Use list_wrds_tables(conn, lib) to inspect tables.
    """
    return conn.list_libraries()


def list_wrds_tables(conn, library: str) -> list[str]:
    """
    List tables in a WRDS library (schema).

    Args:
        conn: wrds.Connection from get_connection().
        library: Schema name (e.g. 'crsp', 'comp', 'lseg').

    Returns:
        List of table names.
    """
    return conn.list_tables(library=library)


def load_sdc_new_deals_wrds(
    conn,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    library: str = "sdc",
    table: Optional[str] = None,
    date_column: str = "ipodate",
    ipodate_not_null: bool = True,
    ipo_only: bool = False,
) -> pd.DataFrame:
    """
    Load SDC Platinum New Deals / New Issues (IPO) data from WRDS.

    SDC data lives in the 'sdc' schema (wrds_ni_details). Includes all rows
    where ipodate is not null by default.

    Args:
        conn: wrds.Connection from get_connection().
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD', or None for latest.
        library: WRDS library (schema); 'sdc' for WRDS SDC New Issues.
        table: Table name. If None, tries wrds_ni_details, then legacy names.
        date_column: Column to filter dates (ipodate for wrds_ni_details).
        ipodate_not_null: If True, exclude rows where ipodate is null.
        ipo_only: If True, also filter to ipo = 'Yes' (wrds_ni_details).

    Returns:
        DataFrame of SDC new deals. Column names depend on the table.
    """
    end_clause = f"AND {date_column} <= '{end}'" if end else ""
    not_null_clause = f"AND {date_column} IS NOT NULL" if ipodate_not_null else ""
    ipo_clause = "AND ipo = 'Yes'" if ipo_only else ""
    tables_to_try = (
        [table]
        if table
        else ["wrds_ni_details", "globalnewiss", "new_issues", "sdc_new_issues", "newiss"]
    )
    last_err = None
    for tbl in tables_to_try:
        try:
            sql = f"""
                SELECT *
                FROM {library}.{tbl}
                WHERE {date_column} >= '{start}' {end_clause} {not_null_clause} {ipo_clause}
                LIMIT 50000
            """
            df = conn.raw_sql(sql)
            if df.empty:
                continue
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not load SDC new deals from {library}. Tried tables: {tables_to_try}. "
        f"Use list_wrds_tables(conn, '{library}') to see available tables. Last error: {last_err}"
    ) from last_err


def load_sdc_ipo_dates_wrds(
    conn,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    library: str = "sdc",
    table: Optional[str] = None,
    date_column: str = "ipodate",
    ticker_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load (ticker, ipodate) from SDC New Deals for IPOs only (ipo='Yes').

    Returns one row per ticker with the earliest ipodate. Drops duplicates.

    Args:
        conn: wrds.Connection from get_connection().
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD', or None for latest.
        library: WRDS library (typically 'sdc').
        table: SDC table name. If None, auto-discovers.
        date_column: Date column (ipodate).
        ticker_column: Ticker column name. If None, auto-detects.

    Returns:
        DataFrame with columns ticker, ipodate.
    """
    df = load_sdc_new_deals_wrds(
        conn, start=start, end=end, library=library,
        table=table, date_column=date_column,
        ipodate_not_null=True, ipo_only=True
    )
    if df.empty:
        return pd.DataFrame(columns=["ticker", "ipo_date"])
    cols = [c.lower() for c in df.columns]
    ticker_col = ticker_column or next(
        (df.columns[cols.index(c)] for c in ["ticker", "ticker_symbol", "symbol", "tic", "iss_ticker", "primary_ticker"]
        if c in cols),
        None,
    )
    date_col = date_column if date_column in df.columns else next(
        (c for c in ["ipodate", "issue_date", "offer_date"] if c in df.columns),
        None,
    )
    if not ticker_col or not date_col:
        return pd.DataFrame(columns=["ticker", "ipo_date"])
    out = df[[ticker_col, date_col]].copy()
    out.columns = ["ticker", "ipo_date"]
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out = out[out["ticker"].str.match(r"^[A-Z0-9\.\-]+$", na=False)]
    out["ipo_date"] = pd.to_datetime(out["ipo_date"])
    out = out.dropna()
    out = out.sort_values("ipo_date").drop_duplicates(subset=["ticker"], keep="first")
    return out[["ticker", "ipo_date"]].reset_index(drop=True)


def load_sdc_ipo_tickers_wrds(
    conn,
    start: str = "2024-01-01",
    end: Optional[str] = None,
    library: str = "sdc",
    table: Optional[str] = None,
    date_column: str = "ipodate",
    ticker_column: Optional[str] = None,
) -> list[str]:
    """
    Extract unique IPO ticker symbols from SDC New Deals (2024-2025).

    Args:
        conn: wrds.Connection from get_connection().
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD', or None for latest.
        library: WRDS library (typically 'sdc' for wrds_ni_details).
        table: SDC table name. If None, tries wrds_ni_details first.
        date_column: Date column for filtering (ipodate for wrds_ni_details).
        ticker_column: Ticker column name. If None, tries: ticker, ticker_symbol,
            symbol, tic, iss_ticker, primary_ticker.

    Returns:
        Sorted list of unique ticker strings (financial tickers from SDC).
    """
    df = load_sdc_new_deals_wrds(
        conn, start=start, end=end, library=library,
        table=table, date_column=date_column,
        ipodate_not_null=True, ipo_only=True
    )
    if df.empty:
        return []
    cols = [c.lower() for c in df.columns]
    candidates = (
        ticker_column
        or next(
            (df.columns[cols.index(c)] for c in ["ticker", "ticker_symbol", "symbol", "tic", "iss_ticker", "primary_ticker"]
            if c in cols),
            None,
        )
    )
    if candidates is None:
        raise ValueError(
            f"No ticker column found in SDC. Columns: {list(df.columns)}. "
            "Pass ticker_column= explicitly."
        )
    tickers = df[candidates].dropna().astype(str).str.strip().str.upper()
    tickers = tickers[tickers.str.match(r"^[A-Z0-9\.\-]+$", na=False)].unique().tolist()
    return sorted(set(tickers))


def load_crsp_daily_prices_wrds(
    conn,
    tickers: list[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load daily close prices and shares from CRSP (crsp.dsf) for given tickers.

    Uses split-adjusted prices. Returns DataFrame with tic, datadate, prccd, prcod, shrout
    for compatibility with IPO optimizer. shrout is shares outstanding (thousands).
    No gvkey; use shrout for market-cap weighting instead of comp.funda.

    Args:
        conn: wrds.Connection from get_connection().
        tickers: List of ticker symbols.
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD', or None for latest.

    Returns:
        DataFrame with tic, datadate, prccd, prcod, shrout.
    """
    if not tickers:
        return pd.DataFrame(columns=["tic", "datadate", "prccd", "prcod", "shrout"])
    ticker_list = ",".join(f"'{t}'" for t in tickers)
    end_val = end or "2099-12-31"
    end_clause = f"AND date <= '{end}'" if end else ""
    sql_permno = f"""
        SELECT permno, ticker
        FROM crsp.dsenames
        WHERE ticker IN ({ticker_list})
          AND namedt <= '{end_val}'
          AND (nameendt >= '{start}' OR nameendt IS NULL)
    """
    try:
        names = conn.raw_sql(sql_permno, date_cols=[])
    except Exception:
        try:
            sql2 = f"""SELECT permno, ticker FROM crsp.dsenames
                WHERE ticker IN ({ticker_list}) AND namedt <= '{end_val}'
                AND (nameenddt >= '{start}' OR nameenddt IS NULL)"""
            names = conn.raw_sql(sql2, date_cols=[])
        except Exception:
            names = conn.raw_sql(
                f"SELECT permno, ticker FROM crsp.dsenames WHERE ticker IN ({ticker_list})",
                date_cols=[],
            )
    if names.empty:
        return pd.DataFrame(columns=["tic", "datadate", "prccd", "prcod", "shrout"])
    permno_list = ",".join(str(p) for p in names["permno"].drop_duplicates().tolist())
    sql = f"""
        SELECT date, permno, prc, shrout
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date >= '{start}' {end_clause}
          AND prc IS NOT NULL
        ORDER BY date, permno
    """
    df = conn.raw_sql(sql, date_cols=["date"])
    if df.empty:
        return pd.DataFrame(columns=["tic", "datadate", "prccd", "prcod", "shrout"])
    df = df.merge(names[["permno", "ticker"]], on="permno", how="left")
    df = df.rename(columns={"date": "datadate", "ticker": "tic"})
    df["prccd"] = df["prc"].abs()
    df["prcod"] = df["prccd"]
    df = df.drop_duplicates(subset=["tic", "datadate"], keep="first")
    return df[["tic", "datadate", "prccd", "prcod", "shrout"]]


def load_compustat_daily_prices_wrds(
    conn,
    tickers: list[str],
    start: str = "2024-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load daily open (prcod) and close (prccd) prices from Compustat for given tickers.

    Returns DataFrame with columns: tic, datadate, prccd, prcod, gvkey (and cusip if available)
    to match the structure expected by the IPO optimizer (like 2025iposdata.csv).

    Args:
        conn: wrds.Connection from get_connection().
        tickers: List of ticker symbols.
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD', or None for latest.

    Returns:
        DataFrame with tic, datadate, prccd, prcod, gvkey. prcod may equal prccd if open unavailable.
    """
    if not tickers:
        return pd.DataFrame(columns=["tic", "datadate", "prccd", "prcod", "gvkey"])
    ticker_list = ",".join(f"'{t}'" for t in tickers)
    end_clause = f"AND s.datadate <= '{end}'" if end else ""
    sql = f"""
        SELECT f.tic, s.datadate, s.prccd, s.prcod, s.gvkey
        FROM comp.sec_dprc s
        INNER JOIN (
            SELECT tic, gvkey, ROW_NUMBER() OVER (PARTITION BY tic ORDER BY datadate DESC) AS rn
            FROM comp.funda
            WHERE tic IN ({ticker_list})
              AND datadate >= '2018-01-01'
              AND csho > 0
              AND indfmt = 'INDL' AND datafmt = 'STD'
        ) f ON f.gvkey = s.gvkey AND f.rn = 1
        WHERE s.datadate >= '{start}' {end_clause}
          AND s.prccd IS NOT NULL
        ORDER BY s.datadate, f.tic
    """
    try:
        df = conn.raw_sql(sql, date_cols=["datadate"])
    except Exception:
        sql_no_prcod = f"""
            SELECT f.tic, s.datadate, s.prccd, s.prccd AS prcod, s.gvkey
            FROM comp.sec_dprc s
            INNER JOIN (
                SELECT tic, gvkey, ROW_NUMBER() OVER (PARTITION BY tic ORDER BY datadate DESC) AS rn
                FROM comp.funda
                WHERE tic IN ({ticker_list})
                  AND datadate >= '2018-01-01'
                  AND csho > 0
                  AND indfmt = 'INDL' AND datafmt = 'STD'
            ) f ON f.gvkey = s.gvkey AND f.rn = 1
            WHERE s.datadate >= '{start}' {end_clause}
              AND s.prccd IS NOT NULL
            ORDER BY s.datadate, f.tic
        """
        df = conn.raw_sql(sql_no_prcod, date_cols=["datadate"])
    if df.empty:
        return pd.DataFrame(columns=["tic", "datadate", "prccd", "prcod", "gvkey"])
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df = df.drop_duplicates(subset=["tic", "datadate"], keep="first")
    return df


def load_ipo_data_from_sdc_wrds(
    conn,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    library: str = "sdc",
    table: Optional[str] = None,
    date_column: str = "ipodate",
    price_source: str = "crsp",
) -> pd.DataFrame:
    """
    Build IPO data from SDC New Deals + daily prices from CRSP or Compustat.

    Flow: (1) Get IPO tickers from SDC, (2) Pull daily prices from CRSP (default)
    or Compustat. CRSP gives split-adjusted prices and shrout for shares.
    Compustat gives gvkey for comp.funda shares lookup.

    Args:
        conn: wrds.Connection from get_connection().
        start: Start date for SDC and prices.
        end: End date for SDC and prices.
        library: WRDS library for SDC (typically 'sdc').
        table: SDC table name. If None, auto-discovers.
        date_column: SDC date column for filtering deals.
        price_source: 'crsp' (split-adjusted, with shrout) or 'compustat'.

    Returns:
        DataFrame with tic, datadate, prccd, prcod, and either shrout (CRSP)
        or gvkey (Compustat).
    """
    tickers = load_sdc_ipo_tickers_wrds(
        conn, start=start, end=end, library=library,
        table=table, date_column=date_column
    )
    if not tickers:
        if price_source == "crsp":
            return pd.DataFrame(columns=["tic", "datadate", "prccd", "prcod", "shrout"])
        return pd.DataFrame(columns=["tic", "datadate", "prccd", "prcod", "gvkey"])
    if price_source == "crsp":
        prices = load_crsp_daily_prices_wrds(conn, tickers=tickers, start=start, end=end)
        if prices.empty:
            return pd.DataFrame(columns=["tic", "datadate", "prccd", "prcod", "shrout"])
        prices["datadate"] = pd.to_datetime(prices["datadate"]).dt.normalize()
        return prices[["tic", "datadate", "prccd", "prcod", "shrout"]].copy()
    prices = load_compustat_daily_prices_wrds(conn, tickers=tickers, start=start, end=end)
    if prices.empty:
        return pd.DataFrame(columns=["tic", "datadate", "prccd", "prcod", "gvkey"])
    prices["datadate"] = pd.to_datetime(prices["datadate"]).dt.normalize()
    return prices[["tic", "datadate", "prccd", "prcod", "gvkey"]].copy()
