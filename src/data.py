"""
Data module - Stock price loading via yfinance and episode building.
Supports single stock vs S&P 500 (SPY) comparison.
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np


@dataclass
class Episode:
    """
    Trading episode for a single stock.

    Attributes:
        ticker: Stock ticker symbol
        start_date: Start date of the episode
        df: DataFrame with columns: date, close, volume, spy_close (benchmark)
        day0_index: Index in df where the episode starts
        N: Number of days in the episode window
        meta: Optional metadata dict
    """
    ticker: str
    ipo_date: Union[datetime, date]  # kept as ipo_date for compatibility
    df: pd.DataFrame
    day0_index: int
    N: int
    meta: Optional[Dict] = None

    def __post_init__(self):
        """Validate episode structure."""
        required_cols = ['date', 'close']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"Episode df must have columns: {required_cols}")

        if not self.df['date'].is_monotonic_increasing:
            self.df = self.df.sort_values('date').reset_index(drop=True)

        if not (0 <= self.day0_index < len(self.df)):
            raise ValueError(f"day0_index {self.day0_index} out of bounds for df length {len(self.df)}")


def get_sp500_tickers() -> List[str]:
    """
    Fetch current S&P 500 constituent tickers from Wikipedia.
    Returns list of ticker symbols.
    """
    import urllib.request
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            html = response.read()
        tables = pd.read_html(html)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch S&P 500 list from Wikipedia: {e}")

    df = tables[0]
    if "Symbol" not in df.columns:
        sym_col = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()]
        if not sym_col:
            raise ValueError(f"S&P 500 table has no Symbol column: {df.columns.tolist()}")
        df = df.rename(columns={sym_col[0]: "Symbol"})
    tickers = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
    return [t for t in tickers if t and t.upper() != "NAN"]


def fetch_stock_data(
    ticker: str,
    period: str = "1y",
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Fetch daily price data for a single ticker from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        start: Start date (overrides period if both start and end provided)
        end: End date

    Returns:
        DataFrame with columns: date, close, volume
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    try:
        if start and end:
            data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        else:
            data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {e}")

    if data is None or len(data) == 0:
        raise ValueError(f"No data returned for {ticker}")

    df = data.reset_index()
    df["date"] = pd.to_datetime(df["Date"])
    if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    # Handle both single-ticker and multi-ticker column formats
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df["close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    out = df[["date", "close", "volume"]].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return out


def fetch_stock_vs_benchmark(
    ticker: str,
    benchmark: str = "SPY",
    period: str = "1y",
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Fetch daily price data for a stock and its benchmark (e.g., SPY for S&P 500).

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        benchmark: Benchmark ticker (default: 'SPY' for S&P 500)
        period: Time period if start/end not provided
        start: Start date
        end: End date

    Returns:
        DataFrame with columns: date, close, volume, benchmark_close,
                               stock_return, benchmark_return, excess_return
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    tickers_to_fetch = [ticker, benchmark]

    try:
        if start and end:
            data = yf.download(tickers_to_fetch, start=start, end=end, auto_adjust=True, progress=True)
        else:
            data = yf.download(tickers_to_fetch, period=period, auto_adjust=True, progress=True)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data: {e}")

    if data is None or len(data) == 0:
        raise ValueError(f"No data returned for {ticker} and {benchmark}")

    # Build combined dataframe
    df = pd.DataFrame({
        "date": data.index,
        "close": data["Close"][ticker].values,
        "volume": data["Volume"][ticker].values if "Volume" in data.columns else 0,
        "benchmark_close": data["Close"][benchmark].values,
    })

    df["date"] = pd.to_datetime(df["date"])
    if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    # Calculate returns
    df["stock_return"] = df["close"].pct_change()
    df["benchmark_return"] = df["benchmark_close"].pct_change()
    df["excess_return"] = df["stock_return"] - df["benchmark_return"]

    # Drop rows with NaN
    df = df.dropna(subset=["close", "benchmark_close"]).reset_index(drop=True)

    return df


def build_episodes_from_stock(
    ticker: str,
    benchmark: str = "SPY",
    period: str = "1y",
    N: int = 21,
    step: int = 5,
) -> List[Episode]:
    """
    Build rolling episodes from a single stock vs benchmark.

    Creates overlapping episodes of N days, stepping by `step` days.
    Each episode contains stock prices and benchmark prices for comparison.

    Args:
        ticker: Stock ticker symbol
        benchmark: Benchmark ticker (default: 'SPY')
        period: Historical period to fetch
        N: Number of trading days per episode
        step: Days to step between episode start dates

    Returns:
        List of Episode objects
    """
    df = fetch_stock_vs_benchmark(ticker, benchmark, period)

    if len(df) < N:
        raise ValueError(f"Not enough data: got {len(df)} days, need {N}")

    episodes = []

    for start_idx in range(0, len(df) - N + 1, step):
        episode_df = df.iloc[start_idx:start_idx + N].copy().reset_index(drop=True)
        start_date = episode_df["date"].iloc[0]

        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()

        episode = Episode(
            ticker=ticker,
            ipo_date=start_date,
            df=episode_df,
            day0_index=0,
            N=N,
            meta={"benchmark": benchmark, "start_idx": start_idx},
        )
        episodes.append(episode)

    return episodes


def load_prices_from_yfinance(
    meta_df: pd.DataFrame,
    N: int = 21,
    buffer_days: int = 5,
    delay_seconds: float = 0.2,
    fetch_days: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily price data from Yahoo Finance for tickers in meta_df using batch download.
    Returns prices_map: ticker -> DataFrame with columns date, close, volume.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    tickers = meta_df["ticker"].astype(str).str.strip().tolist()
    tickers = [t for t in tickers if t and t.upper() != "NAN"]

    if not tickers:
        return {}

    start_date = meta_df["ipo_date"].min()
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    elif hasattr(start_date, "date"):
        start_date = start_date.date()

    end_date = date.today()

    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=True,
            threads=True,
        )
    except Exception as e:
        print(f"yfinance download failed: {e}")
        return {}

    if data is None or len(data) == 0:
        return {}

    prices_map = {}

    if len(tickers) == 1:
        ticker = tickers[0]
        df = data.reset_index()
        df["date"] = pd.to_datetime(df["Date"])
        if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df["close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        out = df[["date", "close", "volume"]].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        if len(out) >= 2:
            prices_map[ticker] = out
    else:
        for ticker in tickers:
            try:
                if ticker not in data["Close"].columns:
                    continue
                df = pd.DataFrame({
                    "date": data.index,
                    "close": data["Close"][ticker].values,
                    "volume": data["Volume"][ticker].values if "Volume" in data.columns else 0,
                })
                df["date"] = pd.to_datetime(df["date"])
                if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
                    df["date"] = df["date"].dt.tz_localize(None)
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
                out = df[["date", "close", "volume"]].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                if len(out) >= 2:
                    prices_map[ticker] = out
            except Exception:
                continue

    return prices_map


def build_episodes(
    meta_df: pd.DataFrame,
    prices_map: Dict[str, pd.DataFrame],
    N: int,
    short_mode: str = "skip"
) -> List[Episode]:
    """
    Build trading episodes from metadata and price data.
    """
    if short_mode not in ["skip", "truncate"]:
        raise ValueError(f"short_mode must be 'skip' or 'truncate', got: {short_mode}")

    episodes = []
    meta_sorted = meta_df.sort_values('ipo_date').reset_index(drop=True)

    for _, row in meta_sorted.iterrows():
        ticker = row['ticker']
        ipo_date = row['ipo_date']

        if ticker not in prices_map:
            continue

        price_df = prices_map[ticker].copy()

        if isinstance(ipo_date, date):
            ipo_datetime = pd.Timestamp(ipo_date)
        else:
            ipo_datetime = pd.Timestamp(ipo_date)

        day0_mask = price_df['date'] >= ipo_datetime
        if not day0_mask.any():
            continue

        first_match_idx = day0_mask.idxmax()
        if first_match_idx is None:
            continue

        if isinstance(price_df.index, pd.RangeIndex):
            day0_pos_int = first_match_idx
        else:
            day0_pos_int = price_df.index.get_loc(first_match_idx)

        end_pos = day0_pos_int + N

        if end_pos > len(price_df):
            if short_mode == "skip":
                continue
            else:
                episode_df = price_df.iloc[day0_pos_int:].copy()
                actual_N = len(episode_df)
        else:
            episode_df = price_df.iloc[day0_pos_int:end_pos].copy()
            actual_N = N

        episode_df = episode_df.reset_index(drop=True)

        meta_dict = None
        if hasattr(row, "to_dict"):
            meta_dict = row.to_dict()
            meta_dict = {k: (None if pd.isna(v) else v) for k, v in meta_dict.items()}

        episode = Episode(
            ticker=ticker,
            ipo_date=ipo_date,
            df=episode_df,
            day0_index=0,
            N=actual_N,
            meta=meta_dict,
        )

        episodes.append(episode)

    return episodes


def generate_synthetic_prices(
    ticker: str,
    ipo_date: Union[date, datetime],
    N: int,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """Generate synthetic daily price data for testing."""
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(ipo_date, datetime):
        ipo_date = ipo_date.date()

    dates = [ipo_date + timedelta(days=i) for i in range(N)]

    returns = rng.normal(0, volatility, N)
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    volumes = rng.lognormal(15, 0.5, N).astype(int)

    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'close': prices,
        'volume': volumes
    })

    return df
