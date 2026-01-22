"""
Tests for IPO metadata loading (load_ipo_meta).
"""

import pytest
import pandas as pd
from datetime import date
from pathlib import Path
import tempfile
import os

from src.data import load_ipo_meta, validate_columns


def test_validate_columns_success():
    """Test validate_columns with all required columns present."""
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'ipo_date': ['2020-01-15', '2020-02-01'],
        'other_col': [1, 2]
    })
    
    # Should not raise
    validate_columns(df, ['ticker', 'ipo_date'])


def test_validate_columns_missing():
    """Test validate_columns raises error when columns are missing."""
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        # Missing 'ipo_date'
    })
    
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(df, ['ticker', 'ipo_date'])


def test_load_ipo_meta_basic(tmp_path):
    """Test loading IPO metadata from CSV."""
    # Create test CSV
    csv_path = tmp_path / "ipo_meta.csv"
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'ipo_date': ['2020-01-15', '2020-02-01', '2020-03-10']
    })
    df.to_csv(csv_path, index=False)
    
    # Load it
    result = load_ipo_meta(csv_path)
    
    assert len(result) == 3
    assert 'ticker' in result.columns
    assert 'ipo_date' in result.columns
    assert result['ticker'].tolist() == ['AAPL', 'MSFT', 'GOOGL']
    
    # Check dates are parsed
    assert all(isinstance(d, date) for d in result['ipo_date'])


def test_load_ipo_meta_missing_columns(tmp_path):
    """Test load_ipo_meta raises error when required columns are missing."""
    csv_path = tmp_path / "ipo_meta.csv"
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        # Missing 'ipo_date'
    })
    df.to_csv(csv_path, index=False)
    
    with pytest.raises(ValueError, match="Missing required columns"):
        load_ipo_meta(csv_path)


def test_load_ipo_meta_file_not_found():
    """Test load_ipo_meta raises error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_ipo_meta("nonexistent_file.csv")


def test_load_ipo_meta_date_parsing(tmp_path):
    """Test load_ipo_meta correctly parses various date formats."""
    csv_path = tmp_path / "ipo_meta.csv"
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'ipo_date': ['2020-01-15', '01/02/2020']  # Different formats
    })
    df.to_csv(csv_path, index=False)
    
    result = load_ipo_meta(csv_path)
    
    # Should parse both formats
    assert all(isinstance(d, date) for d in result['ipo_date'])


def test_load_ipo_meta_preserves_other_columns(tmp_path):
    """Test load_ipo_meta preserves additional columns if present."""
    csv_path = tmp_path / "ipo_meta.csv"
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'ipo_date': ['2020-01-15', '2020-02-01'],
        'sector': ['Tech', 'Tech'],
        'market_cap': [1000, 2000]
    })
    df.to_csv(csv_path, index=False)
    
    result = load_ipo_meta(csv_path)
    
    # Should have all columns
    assert 'ticker' in result.columns
    assert 'ipo_date' in result.columns
    assert 'sector' in result.columns
    assert 'market_cap' in result.columns
