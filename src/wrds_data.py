"""
Pull stock price and return data from WRDS (CRSP).

Requires a WRDS account and, for remote access, two-factor authentication.
Set WRDS_USERNAME in the environment or pass it to get_connection().
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
