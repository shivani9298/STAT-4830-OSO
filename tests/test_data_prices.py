"""
Tests for price data loading (load_prices_dir).
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.data import load_prices_dir, validate_columns


def test_load_prices_dir_basic(tmp_path):
    """Test loading price data from directory."""
    # Create test CSV files
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    
    # AAPL prices
    aapl_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10, freq='D'),
        'close': [100.0 + i for i in range(10)],
        'volume': [1000000] * 10
    })
    aapl_df.to_csv(prices_dir / "AAPL.csv", index=False)
    
    # MSFT prices
    msft_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10, freq='D'),
        'close': [200.0 + i for i in range(10)],
        'volume': [2000000] * 10
    })
    msft_df.to_csv(prices_dir / "MSFT.csv", index=False)
    
    # Load prices
    prices_map = load_prices_dir(prices_dir)
    
    assert len(prices_map) == 2
    assert 'AAPL' in prices_map
    assert 'MSFT' in prices_map
    
    # Check AAPL data
    aapl = prices_map['AAPL']
    assert len(aapl) == 10
    assert 'date' in aapl.columns
    assert 'close' in aapl.columns
    assert 'volume' in aapl.columns
    assert aapl['date'].is_monotonic_increasing


def test_load_prices_dir_missing_required_columns(tmp_path):
    """Test load_prices_dir raises error when required columns are missing."""
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    
    # Missing 'close' column
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='D'),
        # Missing 'close'
    })
    df.to_csv(prices_dir / "AAPL.csv", index=False)
    
    with pytest.raises(ValueError, match="Missing required columns"):
        load_prices_dir(prices_dir)


def test_load_prices_dir_optional_volume(tmp_path):
    """Test load_prices_dir works with or without volume column."""
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    
    # With volume
    df_with_vol = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='D'),
        'close': [100.0] * 5,
        'volume': [1000000] * 5
    })
    df_with_vol.to_csv(prices_dir / "AAPL.csv", index=False)
    
    # Without volume
    df_no_vol = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='D'),
        'close': [200.0] * 5
    })
    df_no_vol.to_csv(prices_dir / "MSFT.csv", index=False)
    
    prices_map = load_prices_dir(prices_dir)
    
    assert 'volume' in prices_map['AAPL'].columns
    assert 'volume' not in prices_map['MSFT'].columns


def test_load_prices_dir_sorts_by_date(tmp_path):
    """Test load_prices_dir sorts data by date ascending."""
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    
    # Create unsorted dates
    dates = pd.date_range('2020-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'date': dates[::-1],  # Reverse order
        'close': [100.0 + i for i in range(5)]
    })
    df.to_csv(prices_dir / "AAPL.csv", index=False)
    
    prices_map = load_prices_dir(prices_dir)
    
    # Should be sorted ascending
    aapl = prices_map['AAPL']
    assert aapl['date'].is_monotonic_increasing
    assert aapl.iloc[0]['date'] < aapl.iloc[-1]['date']


def test_load_prices_dir_fixes_non_monotonic(tmp_path):
    """Test load_prices_dir fixes non-monotonic dates by sorting."""
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    
    # Create data with non-monotonic dates
    df = pd.DataFrame({
        'date': ['2020-01-05', '2020-01-03', '2020-01-01', '2020-01-04', '2020-01-02'],
        'close': [100.0, 102.0, 104.0, 101.0, 103.0]
    })
    df.to_csv(prices_dir / "AAPL.csv", index=False)
    
    prices_map = load_prices_dir(prices_dir)
    
    aapl = prices_map['AAPL']
    # Should be sorted
    assert aapl['date'].is_monotonic_increasing
    assert aapl.iloc[0]['date'] == pd.Timestamp('2020-01-01')
    assert aapl.iloc[-1]['date'] == pd.Timestamp('2020-01-05')


def test_load_prices_dir_directory_not_found():
    """Test load_prices_dir raises error when directory doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_prices_dir("nonexistent_directory")


def test_load_prices_dir_parses_dates(tmp_path):
    """Test load_prices_dir correctly parses date column."""
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    
    df = pd.DataFrame({
        'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'close': [100.0, 101.0, 102.0]
    })
    df.to_csv(prices_dir / "AAPL.csv", index=False)
    
    prices_map = load_prices_dir(prices_dir)
    
    aapl = prices_map['AAPL']
    # Dates should be datetime type
    assert pd.api.types.is_datetime64_any_dtype(aapl['date'])


def test_load_prices_dir_handles_numeric_close(tmp_path):
    """Test load_prices_dir handles numeric close values correctly."""
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=3, freq='D'),
        'close': ['100.5', '101.25', '102.75']  # String numbers
    })
    df.to_csv(prices_dir / "AAPL.csv", index=False)
    
    prices_map = load_prices_dir(prices_dir)
    
    aapl = prices_map['AAPL']
    # Should convert to numeric
    assert pd.api.types.is_numeric_dtype(aapl['close'])
