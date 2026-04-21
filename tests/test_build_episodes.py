"""
Tests for episode building (build_episodes).
"""

import pytest
import pandas as pd
from datetime import date
from pathlib import Path

from src.data import build_episodes, Episode


def test_build_episodes_basic():
    """Test building episodes with sufficient data."""
    # Create metadata
    meta_df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'ipo_date': [date(2020, 1, 15), date(2020, 2, 1)]
    })
    
    # Create price data
    prices_map = {
        'AAPL': pd.DataFrame({
            'date': pd.date_range('2020-01-15', periods=20, freq='D'),
            'close': [100.0 + i for i in range(20)]
        }),
        'MSFT': pd.DataFrame({
            'date': pd.date_range('2020-02-01', periods=20, freq='D'),
            'close': [200.0 + i for i in range(20)]
        })
    }
    
    episodes = build_episodes(meta_df, prices_map, N=10, short_mode="skip")
    
    assert len(episodes) == 2
    assert all(isinstance(ep, Episode) for ep in episodes)
    
    # Check first episode (AAPL)
    aapl_ep = episodes[0]
    assert aapl_ep.ticker == 'AAPL'
    assert aapl_ep.ipo_date == date(2020, 1, 15)
    assert len(aapl_ep.df) == 10
    assert aapl_ep.day0_index == 0
    assert aapl_ep.N == 10
    assert aapl_ep.df.iloc[0]['date'] == pd.Timestamp('2020-01-15')


def test_build_episodes_short_mode_skip():
    """Test build_episodes skips IPOs with insufficient data when short_mode='skip'."""
    meta_df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'ipo_date': [date(2020, 1, 15), date(2020, 2, 1), date(2020, 3, 1)]
    })
    
    prices_map = {
        'AAPL': pd.DataFrame({
            'date': pd.date_range('2020-01-15', periods=20, freq='D'),
            'close': [100.0] * 20
        }),
        'MSFT': pd.DataFrame({
            'date': pd.date_range('2020-02-01', periods=5, freq='D'),  # Only 5 days, need 10
            'close': [200.0] * 5
        }),
        'GOOGL': pd.DataFrame({
            'date': pd.date_range('2020-03-01', periods=20, freq='D'),
            'close': [300.0] * 20
        })
    }
    
    episodes = build_episodes(meta_df, prices_map, N=10, short_mode="skip")
    
    # Should skip MSFT (insufficient data)
    assert len(episodes) == 2
    tickers = [ep.ticker for ep in episodes]
    assert 'AAPL' in tickers
    assert 'GOOGL' in tickers
    assert 'MSFT' not in tickers


def test_build_episodes_short_mode_truncate():
    """Test build_episodes truncates when short_mode='truncate'."""
    meta_df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'ipo_date': [date(2020, 1, 15), date(2020, 2, 1)]
    })
    
    prices_map = {
        'AAPL': pd.DataFrame({
            'date': pd.date_range('2020-01-15', periods=20, freq='D'),
            'close': [100.0] * 20
        }),
        'MSFT': pd.DataFrame({
            'date': pd.date_range('2020-02-01', periods=5, freq='D'),  # Only 5 days, need 10
            'close': [200.0] * 5
        })
    }
    
    episodes = build_episodes(meta_df, prices_map, N=10, short_mode="truncate")
    
    assert len(episodes) == 2
    
    # AAPL should have full 10 days
    aapl_ep = [ep for ep in episodes if ep.ticker == 'AAPL'][0]
    assert aapl_ep.N == 10
    assert len(aapl_ep.df) == 10
    
    # MSFT should be truncated to 5 days
    msft_ep = [ep for ep in episodes if ep.ticker == 'MSFT'][0]
    assert msft_ep.N == 5
    assert len(msft_ep.df) == 5


def test_build_episodes_missing_ticker():
    """Test build_episodes handles missing tickers in prices_map."""
    meta_df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'ipo_date': [date(2020, 1, 15), date(2020, 2, 1), date(2020, 3, 1)]
    })
    
    prices_map = {
        'AAPL': pd.DataFrame({
            'date': pd.date_range('2020-01-15', periods=20, freq='D'),
            'close': [100.0] * 20
        }),
        # MSFT missing
        'GOOGL': pd.DataFrame({
            'date': pd.date_range('2020-03-01', periods=20, freq='D'),
            'close': [300.0] * 20
        })
    }
    
    episodes = build_episodes(meta_df, prices_map, N=10, short_mode="skip")
    
    # Should skip MSFT
    assert len(episodes) == 2
    tickers = [ep.ticker for ep in episodes]
    assert 'AAPL' in tickers
    assert 'GOOGL' in tickers
    assert 'MSFT' not in tickers


def test_build_episodes_ipo_date_after_data():
    """Test build_episodes handles IPO date after all available price data."""
    meta_df = pd.DataFrame({
        'ticker': ['AAPL'],
        'ipo_date': [date(2020, 2, 1)]  # IPO date
    })
    
    prices_map = {
        'AAPL': pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='D'),  # Ends before IPO
            'close': [100.0] * 10
        })
    }
    
    episodes = build_episodes(meta_df, prices_map, N=10, short_mode="skip")
    
    # Should skip (IPO date after all data)
    assert len(episodes) == 0


def test_build_episodes_deterministic_ordering():
    """Test build_episodes returns episodes in deterministic order (by IPO date)."""
    meta_df = pd.DataFrame({
        'ticker': ['MSFT', 'AAPL', 'GOOGL'],  # Not sorted
        'ipo_date': [date(2020, 2, 1), date(2020, 1, 15), date(2020, 3, 1)]
    })
    
    prices_map = {
        'AAPL': pd.DataFrame({
            'date': pd.date_range('2020-01-15', periods=20, freq='D'),
            'close': [100.0] * 20
        }),
        'MSFT': pd.DataFrame({
            'date': pd.date_range('2020-02-01', periods=20, freq='D'),
            'close': [200.0] * 20
        }),
        'GOOGL': pd.DataFrame({
            'date': pd.date_range('2020-03-01', periods=20, freq='D'),
            'close': [300.0] * 20
        })
    }
    
    episodes = build_episodes(meta_df, prices_map, N=10, short_mode="skip")
    
    # Should be sorted by IPO date
    assert len(episodes) == 3
    assert episodes[0].ticker == 'AAPL'  # 2020-01-15
    assert episodes[1].ticker == 'MSFT'  # 2020-02-01
    assert episodes[2].ticker == 'GOOGL'  # 2020-03-01


def test_build_episodes_day0_index_is_zero():
    """Test that day0_index in episode is always 0 (first row of episode df)."""
    meta_df = pd.DataFrame({
        'ticker': ['AAPL'],
        'ipo_date': [date(2020, 1, 20)]  # Day 5 in the price data
    })
    
    prices_map = {
        'AAPL': pd.DataFrame({
            'date': pd.date_range('2020-01-15', periods=20, freq='D'),
            'close': [100.0 + i for i in range(20)]
        })
    }
    
    episodes = build_episodes(meta_df, prices_map, N=10, short_mode="skip")
    
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.day0_index == 0  # Always 0 in episode df
    assert ep.df.iloc[0]['date'] == pd.Timestamp('2020-01-20')  # IPO date


def test_build_episodes_invalid_short_mode():
    """Test build_episodes raises error for invalid short_mode."""
    meta_df = pd.DataFrame({
        'ticker': ['AAPL'],
        'ipo_date': [date(2020, 1, 15)]
    })
    
    prices_map = {
        'AAPL': pd.DataFrame({
            'date': pd.date_range('2020-01-15', periods=20, freq='D'),
            'close': [100.0] * 20
        })
    }
    
    with pytest.raises(ValueError, match="short_mode must be"):
        build_episodes(meta_df, prices_map, N=10, short_mode="invalid")


def test_build_episodes_with_volume():
    """Test build_episodes preserves volume column if present."""
    meta_df = pd.DataFrame({
        'ticker': ['AAPL'],
        'ipo_date': [date(2020, 1, 15)]
    })
    
    prices_map = {
        'AAPL': pd.DataFrame({
            'date': pd.date_range('2020-01-15', periods=20, freq='D'),
            'close': [100.0] * 20,
            'volume': [1000000] * 20
        })
    }
    
    episodes = build_episodes(meta_df, prices_map, N=10, short_mode="skip")
    
    assert len(episodes) == 1
    assert 'volume' in episodes[0].df.columns
