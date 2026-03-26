"""
Data layer for IPO portfolio optimizer.

Load/construct market returns and IPO index returns, align by date,
build rolling windows (T, F), and provide train/validation split by time.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import yfinance as yf
except ImportError:
    yf = None


def load_market_returns(
    start: str = "2010-01-01",
    end: Optional[str] = None,
    ticker: str = "SPY",
) -> pd.Series:
    """
    Load daily market returns (e.g. S&P 500 proxy).

    Uses yfinance to download adjusted close and compute pct_change().
    """
    if yf is None:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            close = data["Adj Close"].squeeze()
        else:
            close = data["Close"].squeeze()
    else:
        close = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    returns = close.pct_change().dropna()
    returns.name = "market_return"
    return returns


def load_ipo_index_from_csv(path: str | Path) -> pd.Series:
    """
    Load IPO index returns from a CSV with columns: date, ipo_return (or similar).
    """
    path = Path(path)
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    if "ipo_return" in df.columns:
        ret_col = "ipo_return"
    else:
        candidates = [c for c in df.columns if "return" in c.lower() or c == "ret"]
        ret_col = candidates[0] if candidates else df.columns[1] if len(df.columns) > 1 else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    out = df.set_index(date_col)[ret_col].sort_index()
    out.name = "ipo_return"
    return out


def build_ipo_index_from_tickers(
    ipo_list: list[tuple[str, str]],
    start: str = "2010-01-01",
    end: Optional[str] = None,
    holding_days: int = 252,
    min_names_per_date: int = 5,
) -> pd.Series:
    """
    Build equal-weight IPO index from a list of (ticker, ipo_date) using yfinance.

    For each calendar date, average daily return across all tickers that are
    in their first `holding_days` trading days after IPO.
    """
    if yf is None:
        raise ImportError("yfinance is required")
    ipo_dates = pd.to_datetime([d for _, d in ipo_list])
    tickers = [t for t, _ in ipo_list]
    # Build panel: for each ticker get first holding_days of returns
    rows = []
    for (ticker, ipo_str), ipo_d in zip(ipo_list, ipo_dates):
        try:
            hist = yf.download(ticker, start=ipo_d, end=end, progress=False, auto_adjust=True)
            if hist.empty or len(hist) < 2:
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                close = hist["Close"].squeeze()
            else:
                close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
            ret = close.pct_change().dropna()
            ret = ret.head(holding_days)
            for d, r in ret.items():
                rows.append({"date": d, "ticker": ticker, "ret": r, "ipo_date": ipo_d})
        except Exception:
            continue
    if not rows:
        # Return a series of zeros aligned to market later
        return pd.Series(dtype=float)
    panel = pd.DataFrame(rows)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ipo_date"] = pd.to_datetime(panel["ipo_date"])
    panel["age"] = panel.groupby("ticker")["date"].transform(lambda x: (x - x.min()).dt.days)
    panel = panel[(panel["age"] >= 0) & (panel["age"] < holding_days)]
    agg = panel.groupby("date")["ret"].agg(["mean", "count"])
    agg.loc[agg["count"] < min_names_per_date, "mean"] = np.nan
    out = agg["mean"].sort_index()
    out.name = "ipo_return"
    return out


def align_returns(
    market: pd.Series,
    ipo: pd.Series,
    drop_na: bool = True,
    clip_market: tuple[float, float] = (-0.10, 0.10),
    clip_ipo: tuple[float, float] = (-0.50, 0.50),
) -> pd.DataFrame:
    """
    Align market and IPO return series to a common date index.
    Clips extreme returns: market ±10% (diversified rarely >10%/day), IPO ±50%.
    """
    df = pd.DataFrame({"market_return": market, "ipo_return": ipo})
    df["market_return"] = df["market_return"].clip(lower=clip_market[0], upper=clip_market[1])
    df["ipo_return"] = df["ipo_return"].clip(lower=clip_ipo[0], upper=clip_ipo[1])
    df = df.sort_index()
    if drop_na:
        df = df.dropna()
    return df


def add_optional_features(df: pd.DataFrame, include_vix: bool = False) -> pd.DataFrame:
    """
    Add optional features: rolling volatility (21d), optionally VIX.
    """
    df = df.copy()
    df["rolling_vol"] = df["market_return"].rolling(21, min_periods=5).std()
    df["rolling_vol"] = df["rolling_vol"].bfill().fillna(0.01)
    if include_vix and yf is not None:
        try:
            vix = yf.download("^VIX", start=df.index.min(), end=df.index.max(), progress=False)
            if not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                    vix_close = vix["Close"].squeeze()
                else:
                    vix_close = vix["Close"]
                df["vix"] = vix_close.reindex(df.index).ffill().fillna(20.0)
            else:
                df["vix"] = 20.0
        except Exception:
            df["vix"] = 20.0
    else:
        df["vix"] = 20.0
    return df


def build_rolling_windows(
    df: pd.DataFrame,
    window_len: int = 252,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build rolling-window dataset for training.

    Returns:
        X: (N, T, F) array of past returns/features
        R: (N, n_assets) realized returns at output date (for loss)
        dates: (N,) index of output dates
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c in ("market_return", "ipo_return", "rolling_vol", "vix")]
    arr = df[feature_cols].values.astype(np.float32)
    n = len(df)
    if n <= window_len:
        return np.zeros((0, window_len, len(feature_cols)), np.float32), np.zeros((0, 2)), np.array([], dtype=object)
    X_list = []
    R_list = []
    dates_list = []
    for i in range(window_len, n):
        X_list.append(arr[i - window_len : i])
        # Realized return at date i: market_return, ipo_return (first two cols for portfolio)
        r = arr[i, :2]
        R_list.append(r)
        dates_list.append(df.index[i])
    X = np.stack(X_list, axis=0)
    R = np.stack(R_list, axis=0)
    dates = np.array(dates_list)
    return X, R, dates


def build_rolling_windows_sector_heads(
    df: pd.DataFrame,
    window_len: int,
    feature_cols: list[str],
    sector_ret_cols: list[str],
    market_col: str = "market_return",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling windows for a shared encoder with **per-sector portfolios** (each head: market vs that sector's IPO basket).

    Returns:
        X: (N, T, F)
        R: (N, G, 2) — for each sector g, [:, g, 0] = market return, [:, g, 1] = sector IPO basket return
        dates: (N,) prediction dates
    """
    arr = df[feature_cols].values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mkt = df[market_col].values.astype(np.float32)
    mkt = np.nan_to_num(mkt, nan=0.0, posinf=0.0, neginf=0.0)
    sec = np.stack([df[c].values.astype(np.float32) for c in sector_ret_cols], axis=1)
    sec = np.nan_to_num(sec, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(df)
    g = len(sector_ret_cols)
    if n <= window_len or g == 0:
        return (
            np.zeros((0, window_len, len(feature_cols)), np.float32),
            np.zeros((0, g, 2), np.float32),
            np.array([], dtype=object),
        )
    X_list = []
    R_list = []
    dates_list = []
    for i in range(window_len, n):
        X_list.append(arr[i - window_len : i])
        m_row = np.full((g,), mkt[i], dtype=np.float32)
        r_g = np.stack([m_row, sec[i, :]], axis=1)
        R_list.append(r_g)
        dates_list.append(df.index[i])
    X = np.stack(X_list, axis=0)
    R = np.stack(R_list, axis=0)
    dates = np.array(dates_list)
    return X, R, dates


def train_val_split(
    X: np.ndarray,
    R: np.ndarray,
    dates: np.ndarray,
    val_start: Optional[str] = None,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into train and validation by time.
    If val_start is None, use last val_frac of dates as validation.
    """
    n = len(dates)
    if n == 0:
        return X, R, dates, X[:0], R[:0], dates[:0]
    if val_start is not None:
        val_start = pd.Timestamp(val_start)
        val_mask = dates >= val_start
    else:
        n_val = max(1, int(n * val_frac))
        val_mask = np.zeros(n, dtype=bool)
        val_mask[-n_val:] = True
    train_mask = ~val_mask
    X_train, R_train, d_train = X[train_mask], R[train_mask], dates[train_mask]
    X_val, R_val, d_val = X[val_mask], R[val_mask], dates[val_mask]
    return X_train, R_train, d_train, X_val, R_val, d_val


def train_val_test_split(
    X: np.ndarray,
    R: np.ndarray,
    dates: np.ndarray,
    val_start: str,
    test_start: str,
    *,
    df_index: Optional[pd.DatetimeIndex] = None,
    window_len: Optional[int] = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split rolling windows by prediction date into train, validation, and test.

    If ``df_index`` and ``window_len`` are provided, applies an **embargo** so that
    the calendar span of inputs for one split does not overlap the prediction dates
    of the next split (no shared rows between a window's past and another split's
    label period). Concretely:

    - **Train** only includes windows whose prediction date is strictly before
      ``df_index[val_pos - window_len]`` (``val_pos`` = first index with date >= val_start).
    - **Validation** includes prediction dates in ``[val_start, val_end_exclusive)``
      where ``val_end_exclusive = min(test_start, df_index[test_pos - window_len])``.
    - **Test** includes prediction dates ``>= test_start``.

    Rows whose prediction date falls only in an embargo gap (between val and test)
    are excluded from val and test; they are not used for training either.

    If ``df_index`` or ``window_len`` is omitted, falls back to label-only boundaries
    (train < val_start; val in [val_start, test_start); test >= test_start) without embargo.
    """
    n = len(dates)
    if n == 0:
        empty = X[:0], R[:0], dates[:0]
        return (*empty, *empty, *empty)
    val_ts = pd.Timestamp(val_start)
    test_ts = pd.Timestamp(test_start)
    if not (val_ts < test_ts):
        raise ValueError("val_start must be strictly before test_start")

    d = pd.to_datetime(dates)

    if df_index is not None and window_len is not None and window_len > 0:
        idx = pd.DatetimeIndex(df_index).sort_values()
        val_pos = int(idx.searchsorted(val_ts))
        test_pos = int(idx.searchsorted(test_ts))
        if val_pos < window_len:
            raise ValueError(
                f"Not enough history before val_start for window_len={window_len}: "
                f"need val_pos >= window_len (val_pos={val_pos})."
            )
        if test_pos < window_len:
            raise ValueError(
                f"Not enough history before test_start for window_len={window_len}: "
                f"need test_pos >= window_len (test_pos={test_pos})."
            )
        if test_pos <= val_pos:
            raise ValueError("test_start must be after val_start on the data index.")

        train_end_exclusive = idx[val_pos - window_len]
        embargo_before_test = idx[test_pos - window_len]
        val_end_exclusive = min(test_ts, embargo_before_test)
        if val_end_exclusive <= val_ts:
            raise ValueError(
                f"No validation window after embargo: val_end_exclusive={val_end_exclusive} "
                f"<= val_start={val_ts}. Space out val_start and test_start or reduce window_len."
            )

        train_mask = d < train_end_exclusive
        val_mask = (d >= val_ts) & (d < val_end_exclusive)
        test_mask = d >= test_ts
    else:
        train_mask = d < val_ts
        val_mask = (d >= val_ts) & (d < test_ts)
        test_mask = d >= test_ts

    X_train, R_train, d_train = X[train_mask], R[train_mask], dates[train_mask]
    X_val, R_val, d_val = X[val_mask], R[val_mask], dates[val_mask]
    X_test, R_test, d_test = X[test_mask], R[test_mask], dates[test_mask]
    return X_train, R_train, d_train, X_val, R_val, d_val, X_test, R_test, d_test


def get_data(
    start: str = "2010-01-01",
    end: Optional[str] = None,
    ipo_csv_path: Optional[str] = None,
    ipo_list: Optional[list[tuple[str, str]]] = None,
    window_len: int = 252,
    val_frac: float = 0.15,
    include_optional_features: bool = True,
    market_source: str = "yfinance",
    wrds_conn=None,
) -> dict:
    """
    Full pipeline: load market + IPO returns, align, add features, build windows, split.

    market_source: "yfinance" (default) or "wrds". If "wrds", pass wrds_conn from wrds_data.get_connection().
    """
    if market_source == "wrds" and wrds_conn is not None:
        from .wrds_data import load_market_returns_wrds
        market = load_market_returns_wrds(wrds_conn, start=start, end=end)
    else:
        market = load_market_returns(start=start, end=end)
    if ipo_csv_path and Path(ipo_csv_path).exists():
        ipo = load_ipo_index_from_csv(ipo_csv_path)
    elif ipo_list:
        ipo = build_ipo_index_from_tickers(ipo_list, start=start, end=end)
    else:
        # Synthetic IPO returns for testing: correlated with market + noise
        np.random.seed(42)
        ipo = market * 1.2 + np.random.randn(len(market)).astype(np.float64) * 0.01
        ipo = pd.Series(ipo.values, index=market.index, name="ipo_return")
    df = align_returns(market, ipo)
    if include_optional_features:
        df = add_optional_features(df, include_vix=False)
    feature_cols = list(df.columns)
    X, R, dates = build_rolling_windows(df, window_len=window_len, feature_cols=feature_cols)
    X_train, R_train, d_train, X_val, R_val, d_val = train_val_split(X, R, dates, val_frac=val_frac)
    return {
        "X_train": X_train,
        "R_train": R_train,
        "dates_train": d_train,
        "X_val": X_val,
        "R_val": R_val,
        "dates_val": d_val,
        "feature_cols": feature_cols,
        "df": df,
        "n_assets": 2,
        "window_len": window_len,
    }
