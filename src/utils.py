"""
Utility functions for Online Portfolio Optimization.

This module provides helper functions for:
- Data fetching from Yahoo Finance
- IPO index construction
- Performance metrics calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def fetch_price_and_shares(
    tickers: List[str],
    start_date: str,
    end_date: str,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fetch price data and shares outstanding from Yahoo Finance.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    verbose : bool
        Print progress messages

    Returns
    -------
    prices : pd.DataFrame
        DataFrame with Date index and ticker columns (adjusted close)
    shares : Dict[str, float]
        Dictionary mapping ticker -> shares outstanding
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required. Install with: pip install yfinance")

    price_data = {}
    shares_data = {}
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=True)

            if not hist.empty and len(hist) > 10:
                price_data[ticker] = hist['Close']

                info = stock.info
                shares = info.get('sharesOutstanding',
                                  info.get('impliedSharesOutstanding'))
                if shares:
                    shares_data[ticker] = shares
                else:
                    mkt_cap = info.get('marketCap')
                    if mkt_cap and len(hist) > 0:
                        shares_data[ticker] = mkt_cap / hist['Close'].iloc[-1]
            else:
                failed.append(ticker)
        except Exception:
            failed.append(ticker)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(tickers)} tickers...")

    if verbose and failed:
        print(f"Failed: {failed}")

    prices = pd.DataFrame(price_data)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    return prices, shares_data


def build_ipo_index(
    prices: pd.DataFrame,
    ipo_dates: pd.DataFrame,
    shares: Dict[str, float],
    holding_days: int = 180,
    min_constituents: int = 1,
    weighting: str = "market_cap"
) -> pd.DataFrame:
    """
    Build IPO index with specified holding period.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with Date index and ticker columns
    ipo_dates : pd.DataFrame
        DataFrame with 'ticker' and 'ipo_date' columns
    shares : Dict[str, float]
        Shares outstanding per ticker
    holding_days : int
        Days to hold each IPO stock (default: 180)
    min_constituents : int
        Minimum stocks required for valid return
    weighting : str
        'market_cap' or 'equal'

    Returns
    -------
    pd.DataFrame
        Index with columns: ipo_ret, num_stocks, total_market_cap, constituents
    """
    ipo_lookup = dict(zip(ipo_dates['ticker'], ipo_dates['ipo_date']))
    returns = prices.pct_change()

    # Get trading days per ticker
    trading_days = {}
    for ticker in prices.columns:
        if ticker not in ipo_lookup:
            continue
        valid_days = prices[ticker].dropna().index.tolist()
        if valid_days:
            trading_days[ticker] = valid_days

    index_data = []

    for date in prices.index:
        eligible = []
        mcaps = {}

        for ticker, ipo_date in ipo_lookup.items():
            if ticker not in trading_days:
                continue
            if ticker not in shares and weighting == "market_cap":
                continue

            ticker_days = trading_days[ticker]

            # Find first trading day
            first_idx = None
            for i, d in enumerate(ticker_days):
                if d >= ipo_date:
                    first_idx = i
                    break

            if first_idx is None or date not in ticker_days:
                continue

            current_idx = ticker_days.index(date)
            days_since = current_idx - first_idx

            if 0 <= days_since < holding_days:
                try:
                    price = prices.loc[date, ticker]
                    if pd.notna(price) and price > 0:
                        eligible.append(ticker)
                        if weighting == "market_cap" and ticker in shares:
                            mcaps[ticker] = price * shares[ticker]
                except Exception:
                    pass

        # Calculate return
        if len(eligible) >= min_constituents:
            if weighting == "market_cap" and mcaps:
                total = sum(mcaps.values())
                if total > 0:
                    ret = sum(
                        mcaps.get(t, 0) / total * returns.loc[date, t]
                        for t in eligible
                        if pd.notna(returns.loc[date, t])
                    )
                else:
                    ret = np.nan
            else:
                rets = [returns.loc[date, t] for t in eligible
                        if pd.notna(returns.loc[date, t])]
                ret = np.mean(rets) if rets else np.nan
        else:
            ret = np.nan

        index_data.append({
            'date': date,
            'ipo_ret': ret,
            'num_stocks': len(eligible),
            'total_market_cap': sum(mcaps.values()) if mcaps else 0,
            'constituents': eligible
        })

    return pd.DataFrame(index_data).set_index('date')


def calculate_metrics(returns: pd.Series, name: str = "Portfolio") -> Dict:
    """
    Calculate comprehensive performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily return series
    name : str
        Strategy name

    Returns
    -------
    Dict with performance metrics
    """
    rets = returns.dropna()
    if len(rets) == 0:
        return {'Name': name, 'Error': 'No returns'}

    # Basic stats
    total_ret = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total_ret) ** (1/n_years) - 1
    ann_vol = rets.std() * np.sqrt(252)

    # Sharpe
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Sortino
    downside = rets[rets < 0]
    down_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = ann_ret / down_std if down_std > 0 else 0

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        'Name': name,
        'Total Return': f"{total_ret:.1%}",
        'Ann. Return': f"{ann_ret:.1%}",
        'Ann. Volatility': f"{ann_vol:.1%}",
        'Sharpe': f"{sharpe:.2f}",
        'Sortino': f"{sortino:.2f}",
        'Max Drawdown': f"{max_dd:.1%}",
        'Calmar': f"{calmar:.2f}"
    }


def run_backtest(
    returns: pd.DataFrame,
    allocator,
    window: int = 126
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run walk-forward backtest.

    Parameters
    ----------
    returns : pd.DataFrame
        Return matrix with Date index and asset columns
    allocator : OnlineOGDAllocator
        Initialized allocator instance
    window : int
        Lookback window

    Returns
    -------
    portfolio_returns : pd.Series
        Daily portfolio returns
    weights : pd.DataFrame
        Daily portfolio weights
    """
    weights_list = []
    dates = []

    for i in range(window, len(returns)):
        window_data = returns.iloc[i - window:i].values
        w = allocator.step(window_data)
        weights_list.append(w)
        dates.append(returns.index[i])

    weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)
    aligned = returns.loc[weights_df.index]
    port_ret = (aligned.values * weights_df.values).sum(axis=1)
    port_ret = pd.Series(port_ret, index=weights_df.index)

    return port_ret, weights_df
