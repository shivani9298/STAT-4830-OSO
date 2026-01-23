"""
Data module - IPO metadata loading, price data loading, and episode building.
Owned by Person A.
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np


@dataclass
class IPOInfo:
    """IPO metadata: ticker and IPO date."""
    ticker: str
    ipo_date: Union[datetime, date]


@dataclass
class Episode:
    """
    Trading episode for a single IPO.
    
    Attributes:
        ticker: Stock ticker symbol
        ipo_date: IPO date
        df: DataFrame with columns: date, close, (volume?) sorted ascending by date
        day0_index: Index in df where day0 (IPO day) occurs
        N: Number of days in the episode window
    """
    ticker: str
    ipo_date: Union[datetime, date]
    df: pd.DataFrame
    day0_index: int
    N: int
    
    def __post_init__(self):
        """Validate episode structure."""
        required_cols = ['date', 'close']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"Episode df must have columns: {required_cols}")
        
        # Ensure df is sorted by date ascending
        if not self.df['date'].is_monotonic_increasing:
            self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Validate day0_index is within bounds
        if not (0 <= self.day0_index < len(self.df)):
            raise ValueError(f"day0_index {self.day0_index} out of bounds for df length {len(self.df)}")


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required: List of required column names
        
    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_ipo_meta(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load IPO metadata CSV.
    
    Required columns: ticker, ipo_date
    Parses ipo_date as datetime/date.
    
    Also supports CSV files with 'Symbol' and 'Date Priced' columns (auto-converts).
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with columns: ticker, ipo_date (parsed as datetime)
        
    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IPO metadata file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Handle different column name formats
    if 'Symbol' in df.columns and 'ticker' not in df.columns:
        df['ticker'] = df['Symbol']
    if 'Date Priced' in df.columns and 'ipo_date' not in df.columns:
        df['ipo_date'] = df['Date Priced']
    
    # Validate required columns
    validate_columns(df, ['ticker', 'ipo_date'])
    
    # Parse ipo_date (handle mixed formats)
    df['ipo_date'] = pd.to_datetime(df['ipo_date'], format='mixed', errors='coerce').dt.date
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['ipo_date', 'ticker'])
    
    return df[['ticker', 'ipo_date']].copy()


def generate_synthetic_prices(
    ticker: str,
    ipo_date: Union[date, datetime],
    N: int,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """
    Generate synthetic daily price data for an IPO episode.
    
    Creates N days of price data starting from ipo_date with random walk.
    
    Args:
        ticker: Stock ticker
        ipo_date: IPO date (day 0)
        N: Number of days
        initial_price: Starting price
        volatility: Daily volatility (std of returns)
        rng: Random number generator (for reproducibility)
        
    Returns:
        DataFrame with columns: date, close, volume
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if isinstance(ipo_date, datetime):
        ipo_date = ipo_date.date()
    
    dates = [ipo_date + timedelta(days=i) for i in range(N)]
    
    # Generate random walk prices
    returns = rng.normal(0, volatility, N)
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate synthetic volume
    volumes = rng.lognormal(15, 0.5, N).astype(int)
    
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'close': prices,
        'volume': volumes
    })
    
    return df


def load_prices_dir(prices_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load price data from directory of CSV files.
    
    Each CSV file should be named {ticker}.csv and contain columns:
    - date (will be parsed as datetime)
    - close (required)
    - volume (optional)
    
    Data is sorted by date ascending and non-monotonic dates are fixed by sorting.
    
    Args:
        prices_dir: Directory containing price CSV files
        
    Returns:
        Dictionary mapping ticker -> DataFrame with columns: date, close, (volume?)
        DataFrames are sorted by date ascending
    """
    prices_dir = Path(prices_dir)
    if not prices_dir.exists():
        raise FileNotFoundError(f"Prices directory not found: {prices_dir}")
    
    prices_map = {}
    
    for csv_file in prices_dir.glob("*.csv"):
        ticker = csv_file.stem  # filename without .csv extension
        
        df = pd.read_csv(csv_file)
        
        # Validate required columns
        validate_columns(df, ['date', 'close'])
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date ascending (fixes non-monotonic issues)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Ensure close is numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # Handle volume if present (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        prices_map[ticker] = df
    
    return prices_map


def build_episodes(
    meta_df: pd.DataFrame,
    prices_map: Dict[str, pd.DataFrame],
    N: int,
    short_mode: str = "skip"
) -> List[Episode]:
    """
    Build trading episodes from IPO metadata and price data.
    
    For each IPO:
    1. Find the IPO date in the price data
    2. Extract N days starting from day0 (IPO day)
    3. Handle cases where price history is too short
    
    Args:
        meta_df: DataFrame with columns: ticker, ipo_date
        prices_map: Dictionary mapping ticker -> price DataFrame (date, close, volume?)
        N: Number of days in episode window (including day0)
        short_mode: How to handle short histories:
            - "skip": Skip IPOs with insufficient data
            - "truncate": Use available data (may result in episodes with < N days)
    
    Returns:
        List of Episode objects, sorted by IPO date
        
    Note:
        - day0_index is the index in the episode df where the IPO date occurs
        - Episode df contains exactly N days (or fewer if truncated) starting from day0
        - Episodes are deterministic: sorted by IPO date
    """
    if short_mode not in ["skip", "truncate"]:
        raise ValueError(f"short_mode must be 'skip' or 'truncate', got: {short_mode}")
    
    episodes = []
    
    # Sort by IPO date for deterministic ordering
    meta_sorted = meta_df.sort_values('ipo_date').reset_index(drop=True)
    
    for _, row in meta_sorted.iterrows():
        ticker = row['ticker']
        ipo_date = row['ipo_date']
        
        # Skip if no price data available
        if ticker not in prices_map:
            if short_mode == "skip":
                continue
            else:  # truncate mode - can't truncate if no data at all
                continue
        
        price_df = prices_map[ticker].copy()
        
        # Find IPO date in price data
        # Convert ipo_date to datetime for comparison if needed
        if isinstance(ipo_date, date):
            ipo_datetime = pd.Timestamp(ipo_date)
        else:
            ipo_datetime = pd.Timestamp(ipo_date)
        
        # Find the row where date matches IPO date (or closest after)
        day0_mask = price_df['date'] >= ipo_datetime
        if not day0_mask.any():
            # IPO date is after all available price data
            if short_mode == "skip":
                continue
            else:
                continue
        
        # Get the integer position of day0 in the original price_df
        # Use argmax to find first True (or use where + first_valid_index)
        first_match_idx = day0_mask.idxmax() if day0_mask.any() else None
        if first_match_idx is None:
            if short_mode == "skip":
                continue
            else:
                continue
        
        # Convert index label to integer position
        if isinstance(price_df.index, pd.RangeIndex):
            day0_pos_int = first_match_idx
        else:
            day0_pos_int = price_df.index.get_loc(first_match_idx)
        
        # Extract N days starting from day0
        end_pos = day0_pos_int + N
        
        if end_pos > len(price_df):
            # Not enough data
            if short_mode == "skip":
                continue
            else:  # truncate
                episode_df = price_df.iloc[day0_pos_int:].copy()
                actual_N = len(episode_df)
        else:
            episode_df = price_df.iloc[day0_pos_int:end_pos].copy()
            actual_N = N
        
        # Reset index for clean episode df
        episode_df = episode_df.reset_index(drop=True)
        
        # day0_index in the episode df is always 0 (first row)
        episode = Episode(
            ticker=ticker,
            ipo_date=ipo_date,
            df=episode_df,
            day0_index=0,
            N=actual_N
        )
        
        episodes.append(episode)
    
    return episodes
