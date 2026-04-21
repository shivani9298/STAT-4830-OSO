"""
Tests for data schema (IPOInfo, Episode dataclasses).
"""

import pytest
from datetime import date, datetime
import pandas as pd
import numpy as np

from src.data import IPOInfo, Episode


def test_ipo_info_basic():
    """Test IPOInfo dataclass creation."""
    ipo = IPOInfo(ticker="AAPL", ipo_date=date(2020, 1, 15))
    assert ipo.ticker == "AAPL"
    assert ipo.ipo_date == date(2020, 1, 15)
    
    # Test with datetime
    ipo2 = IPOInfo(ticker="MSFT", ipo_date=datetime(2020, 2, 1))
    assert ipo2.ticker == "MSFT"
    assert isinstance(ipo2.ipo_date, (date, datetime))


def test_episode_basic():
    """Test Episode dataclass creation with valid data."""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-15', periods=10, freq='D'),
        'close': [100.0 + i for i in range(10)],
        'volume': [1000000] * 10
    })
    
    episode = Episode(
        ticker="AAPL",
        ipo_date=date(2020, 1, 15),
        df=df,
        day0_index=0,
        N=10
    )
    
    assert episode.ticker == "AAPL"
    assert episode.ipo_date == date(2020, 1, 15)
    assert len(episode.df) == 10
    assert episode.day0_index == 0
    assert episode.N == 10
    assert 'date' in episode.df.columns
    assert 'close' in episode.df.columns


def test_episode_missing_required_columns():
    """Test Episode validation fails with missing required columns."""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-15', periods=10, freq='D'),
        # Missing 'close' column
    })
    
    with pytest.raises(ValueError, match="Episode df must have columns"):
        Episode(
            ticker="AAPL",
            ipo_date=date(2020, 1, 15),
            df=df,
            day0_index=0,
            N=10
        )


def test_episode_auto_sorts_by_date():
    """Test Episode automatically sorts df by date if not monotonic."""
    # Create unsorted dates
    dates = pd.date_range('2020-01-15', periods=5, freq='D')
    df = pd.DataFrame({
        'date': dates[::-1],  # Reverse order
        'close': [100.0 + i for i in range(5)]
    })
    
    episode = Episode(
        ticker="AAPL",
        ipo_date=date(2020, 1, 11),  # Before first date
        df=df,
        day0_index=0,
        N=5
    )
    
    # Should be sorted ascending
    assert episode.df['date'].is_monotonic_increasing
    assert episode.df.iloc[0]['date'] < episode.df.iloc[-1]['date']


def test_episode_day0_index_validation():
    """Test Episode validates day0_index is within bounds."""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-15', periods=10, freq='D'),
        'close': [100.0] * 10
    })
    
    # Valid index
    episode = Episode(
        ticker="AAPL",
        ipo_date=date(2020, 1, 15),
        df=df,
        day0_index=5,
        N=10
    )
    assert episode.day0_index == 5
    
    # Invalid index (out of bounds)
    with pytest.raises(ValueError, match="day0_index.*out of bounds"):
        Episode(
            ticker="AAPL",
            ipo_date=date(2020, 1, 15),
            df=df,
            day0_index=20,  # Out of bounds
            N=10
        )


def test_episode_optional_volume():
    """Test Episode works with or without volume column."""
    # With volume
    df_with_vol = pd.DataFrame({
        'date': pd.date_range('2020-01-15', periods=5, freq='D'),
        'close': [100.0] * 5,
        'volume': [1000000] * 5
    })
    
    episode1 = Episode(
        ticker="AAPL",
        ipo_date=date(2020, 1, 15),
        df=df_with_vol,
        day0_index=0,
        N=5
    )
    assert 'volume' in episode1.df.columns
    
    # Without volume
    df_no_vol = pd.DataFrame({
        'date': pd.date_range('2020-01-15', periods=5, freq='D'),
        'close': [100.0] * 5
    })
    
    episode2 = Episode(
        ticker="MSFT",
        ipo_date=date(2020, 1, 15),
        df=df_no_vol,
        day0_index=0,
        N=5
    )
    assert 'volume' not in episode2.df.columns
