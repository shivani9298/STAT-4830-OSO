#!/usr/bin/env python3
"""
Build a market-cap weighted IPO index from dailyhistorical_21-26.csv.
Fetches 180 days of data for each IPO using yfinance.
"""

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import torch
    from src.gru_portfolio import GRUPolicy, train_gru_on_returns
    _GRU_AVAILABLE = True
except ImportError:
    _GRU_AVAILABLE = False


def load_ipo_dates(csv_path: str) -> pd.DataFrame:
    """Load IPO data and extract first trading date for each ticker."""
    df = pd.read_csv(csv_path)
    df['datadate'] = pd.to_datetime(df['datadate'])

    # Get first trading date (IPO date) for each ticker
    ipo_dates = df.groupby('tic').agg({
        'datadate': 'min',
        'prccd': 'first',  # First day closing price
        'gvkey': 'first',
    }).reset_index()

    ipo_dates.columns = ['ticker', 'ipo_date', 'ipo_price', 'gvkey']
    ipo_dates = ipo_dates.sort_values('ipo_date').reset_index(drop=True)

    return ipo_dates


def fetch_ipo_data(ticker: str, ipo_date: pd.Timestamp, days: int = 180) -> dict:
    """
    Fetch price data and shares outstanding for an IPO.

    Returns dict with:
        - prices: DataFrame with date, close, volume
        - shares_outstanding: number of shares
        - success: bool
    """
    end_date = ipo_date + timedelta(days=days + 30)  # Buffer for trading days

    try:
        stock = yf.Ticker(ticker)

        # Get historical data
        hist = stock.history(start=ipo_date, end=end_date, auto_adjust=True)

        if hist is None or len(hist) == 0:
            return {'success': False, 'reason': 'No price data'}

        # Get shares outstanding
        info = stock.info
        shares_outstanding = info.get('sharesOutstanding', None)

        if shares_outstanding is None:
            # Try alternative fields
            shares_outstanding = info.get('impliedSharesOutstanding', None)

        if shares_outstanding is None:
            return {'success': False, 'reason': 'No shares outstanding data'}

        # Process price data
        hist = hist.reset_index()
        hist['date'] = pd.to_datetime(hist['Date'])
        if hist['date'].dt.tz is not None:
            hist['date'] = hist['date'].dt.tz_localize(None)

        prices = hist[['date', 'Close', 'Volume']].copy()
        prices.columns = ['date', 'close', 'volume']
        prices = prices.dropna(subset=['close']).head(days)  # Limit to 180 trading days

        if len(prices) < 10:  # Need at least 10 days
            return {'success': False, 'reason': f'Only {len(prices)} days of data'}

        return {
            'success': True,
            'prices': prices,
            'shares_outstanding': shares_outstanding,
            'ticker': ticker,
            'ipo_date': ipo_date,
        }

    except Exception as e:
        return {'success': False, 'reason': str(e)}


def calculate_metrics(returns_window: np.ndarray, risk_free: float = 0.04) -> tuple:
    """
    Calculate Sharpe, annualized return, and max drawdown over a return window.

    Returns:
        tuple: (sharpe_ratio, annualized_return, max_drawdown)
    """
    if len(returns_window) == 0:
        return 0.0, 0.0, 0.0
    ann_return = np.mean(returns_window) * 252
    vol = np.std(returns_window) * np.sqrt(252)
    sharpe = (ann_return - risk_free) / vol if vol > 1e-12 else 0.0
    cum = np.cumprod(1 + returns_window)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_dd = np.min(drawdowns)  # negative number
    return sharpe, ann_return, max_dd


def project_onto_simplex(weights: np.ndarray) -> np.ndarray:
    """Project weights onto simplex (non-negative, sum to 1)."""
    w = np.clip(weights, 0, None)
    s = np.sum(w)
    if s <= 0:
        return np.ones_like(weights) / len(weights)
    return w / s


class OnlinePortfolioOptimizer:
    """
    Online portfolio optimizer using gradient ascent on a risk-adjusted objective.
    Updates weights each period using a gradient that combines return, volatility, and drawdown.
    """

    def __init__(self, n_assets: int, learning_rate: float = 0.01, window_size: int = 30):
        self.n_assets = n_assets
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.weights = np.ones(n_assets) / n_assets
        self.returns_history = []

    def calculate_gradient(self, daily_returns: np.ndarray) -> np.ndarray:
        """Compute gradient for weight updates (return + volatility + simplified drawdown)."""
        portfolio_return = np.sum(self.weights * daily_returns)
        self.returns_history.append(portfolio_return)

        if len(self.returns_history) < self.window_size:
            return daily_returns - np.mean(daily_returns)

        returns_window = np.array(self.returns_history[-self.window_size:])
        sharpe, ann_ret, drawdown = calculate_metrics(returns_window)

        return_gradient = daily_returns
        volatility_gradient = -2 * (daily_returns - np.mean(returns_window))
        gradient = 0.4 * return_gradient + 0.3 * volatility_gradient + 0.3 * return_gradient
        return gradient

    def update(self, daily_returns: np.ndarray) -> np.ndarray:
        """Update portfolio weights using gradient ascent and project onto simplex."""
        gradient = self.calculate_gradient(daily_returns)
        self.weights = self.weights + self.learning_rate * gradient
        self.weights = project_onto_simplex(self.weights)
        return self.weights.copy()


def _get_daily_constituents(ipo_data: list, current_date: pd.Timestamp, days_window: int = 180) -> list:
    """Return list of constituent dicts (ticker, price, market_cap, daily_return, ...) for one date."""
    daily_constituents = []
    for ipo in ipo_data:
        ipo_date = ipo['ipo_date']
        prices_df = ipo['prices']
        shares = ipo['shares_outstanding']
        days_since_ipo = (current_date - ipo_date).days
        if days_since_ipo < 0 or days_since_ipo > days_window:
            continue
        price_row = prices_df[prices_df['date'].dt.date == current_date.date()]
        if len(price_row) == 0:
            continue
        close_price = price_row['close'].iloc[0]
        market_cap = shares * close_price
        prev_row = pd.DataFrame()
        for lookback in range(1, 6):
            prev_date = current_date - timedelta(days=lookback)
            prev_row = prices_df[prices_df['date'].dt.date == prev_date.date()]
            if len(prev_row) > 0:
                break
        daily_return = (close_price / prev_row['close'].iloc[0] - 1) if len(prev_row) > 0 else 0.0
        daily_constituents.append({
            'ticker': ipo['ticker'],
            'price': close_price,
            'market_cap': market_cap,
            'shares': shares,
            'daily_return': daily_return,
            'days_since_ipo': days_since_ipo,
        })
    return daily_constituents


def build_ipo_index(
    ipo_data: list,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    days_window: int = 180,
    use_optimizer: bool = False,
    optimizer_kwargs: dict = None,
    use_gru: bool = False,
    gru_kwargs: dict = None,
) -> tuple:
    """
    Build IPO index (market-cap, OnlinePortfolioOptimizer, or GRU weighted).

    Each day: include IPOs within first `days_window` days; weight by market cap,
    OnlinePortfolioOptimizer (use_optimizer=True), or GRUPolicy (use_gru=True).

    Returns:
        tuple: (index_df, weights_df)
    """
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    index_data = []
    weights_data = []
    optimizer_kwargs = optimizer_kwargs or {}
    opt_lr = optimizer_kwargs.get('learning_rate', 0.01)
    opt_window = optimizer_kwargs.get('window_size', 30)
    gru_kwargs = gru_kwargs or {}
    gru_window = gru_kwargs.get('window_size', 21)
    gru_epochs = gru_kwargs.get('n_epochs', 30)
    gru_hidden = gru_kwargs.get('hidden_size', 128)

    gru_model = None
    if use_gru:
        if not _GRU_AVAILABLE:
            raise RuntimeError("GRU requires PyTorch and src.gru_portfolio; install torch and run from repo root.")
        # Collect all tickers and build return matrix [T, N] + list of (date, constituents)
        all_tickers = []
        for current_date in all_dates:
            daily_constituents = _get_daily_constituents(ipo_data, current_date, days_window)
            for c in daily_constituents:
                if c['ticker'] not in all_tickers:
                    all_tickers.append(c['ticker'])
        n_assets = len(all_tickers)
        if n_assets == 0:
            return pd.DataFrame(index_data), pd.DataFrame(weights_data)
        ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}

        index_dates = []
        constituents_per_date = []
        return_rows = []
        for current_date in all_dates:
            daily_constituents = _get_daily_constituents(ipo_data, current_date, days_window)
            if len(daily_constituents) == 0:
                continue
            row = np.zeros(n_assets, dtype=np.float32)
            for c in daily_constituents:
                row[ticker_to_idx[c['ticker']]] = c['daily_return']
            index_dates.append(current_date)
            constituents_per_date.append(daily_constituents)
            return_rows.append(row)

        return_matrix = np.array(return_rows)
        if len(return_matrix) < gru_window + 1:
            use_gru = False  # fallback to market-cap for the rest
        else:
            gru_model, _ = train_gru_on_returns(
                return_matrix,
                window=gru_window,
                n_epochs=gru_epochs,
                hidden_size=gru_hidden,
                lr=gru_kwargs.get('lr', 1e-3),
                lam_var=gru_kwargs.get('lam_var', 0.1),
                lam_tc=gru_kwargs.get('lam_tc', 1e-3),
                seed=gru_kwargs.get('seed', 0),
            )
            gru_model.eval()

    if use_optimizer and not use_gru:
        # First pass: collect all tickers (order of first appearance)
        all_tickers = []
        for current_date in all_dates:
            daily_constituents = _get_daily_constituents(ipo_data, current_date, days_window)
            for c in daily_constituents:
                if c['ticker'] not in all_tickers:
                    all_tickers.append(c['ticker'])
        n_assets = len(all_tickers)
        if n_assets == 0:
            return pd.DataFrame(index_data), pd.DataFrame(weights_data)
        optimizer = OnlinePortfolioOptimizer(n_assets, learning_rate=opt_lr, window_size=opt_window)
        ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}

    if use_gru and gru_model is not None:
        # GRU path: iterate over index_dates
        device = next(gru_model.parameters()).device
        for i, current_date in enumerate(index_dates):
            daily_constituents = constituents_per_date[i]
            total_market_cap = sum(c['market_cap'] for c in daily_constituents)

            if i < gru_window:
                # Not enough history: market-cap weight
                w_sum = total_market_cap
                weighted_return = sum(c['daily_return'] * (c['market_cap'] / w_sum) for c in daily_constituents)
                for c in daily_constituents:
                    weights_data.append({
                        'date': current_date,
                        'ticker': c['ticker'],
                        'price': c['price'],
                        'shares': c['shares'],
                        'market_cap': c['market_cap'],
                        'weight': c['market_cap'] / w_sum,
                        'daily_return': c['daily_return'],
                        'days_since_ipo': c['days_since_ipo'],
                    })
            else:
                x = torch.tensor(
                    return_matrix[i - gru_window:i][None],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    w = gru_model(x).cpu().numpy()[0]
                active_weights = np.array([w[ticker_to_idx[c['ticker']]] for c in daily_constituents])
                if active_weights.sum() <= 0:
                    active_weights = np.ones(len(daily_constituents)) / len(daily_constituents)
                else:
                    active_weights = active_weights / active_weights.sum()
                weighted_return = sum(c['daily_return'] * w for c, w in zip(daily_constituents, active_weights))
                for c, w in zip(daily_constituents, active_weights):
                    weights_data.append({
                        'date': current_date,
                        'ticker': c['ticker'],
                        'price': c['price'],
                        'shares': c['shares'],
                        'market_cap': c['market_cap'],
                        'weight': float(w),
                        'daily_return': c['daily_return'],
                        'days_since_ipo': c['days_since_ipo'],
                    })

            index_data.append({
                'date': current_date,
                'num_constituents': len(daily_constituents),
                'total_market_cap': total_market_cap,
                'daily_return': weighted_return,
            })
        return pd.DataFrame(index_data), pd.DataFrame(weights_data)

    for current_date in all_dates:
        daily_constituents = _get_daily_constituents(ipo_data, current_date, days_window)
        if len(daily_constituents) == 0:
            continue

        if use_optimizer:
            daily_returns = np.zeros(n_assets)
            for c in daily_constituents:
                daily_returns[ticker_to_idx[c['ticker']]] = c['daily_return']
            optimizer.update(daily_returns)
            # Use optimizer weights only for today's constituents; renormalize
            active_weights = np.array([optimizer.weights[ticker_to_idx[c['ticker']]] for c in daily_constituents])
            if active_weights.sum() <= 0:
                active_weights = np.ones(len(daily_constituents)) / len(daily_constituents)
            else:
                active_weights = active_weights / active_weights.sum()
            weighted_return = sum(c['daily_return'] * w for c, w in zip(daily_constituents, active_weights))
            total_market_cap = sum(c['market_cap'] for c in daily_constituents)
            for c, w in zip(daily_constituents, active_weights):
                weights_data.append({
                    'date': current_date,
                    'ticker': c['ticker'],
                    'price': c['price'],
                    'shares': c['shares'],
                    'market_cap': c['market_cap'],
                    'weight': w,
                    'daily_return': c['daily_return'],
                    'days_since_ipo': c['days_since_ipo'],
                })
        else:
            total_market_cap = sum(c['market_cap'] for c in daily_constituents)
            weighted_return = sum(
                c['daily_return'] * (c['market_cap'] / total_market_cap)
                for c in daily_constituents
            )
            for c in daily_constituents:
                weight = c['market_cap'] / total_market_cap
                weights_data.append({
                    'date': current_date,
                    'ticker': c['ticker'],
                    'price': c['price'],
                    'shares': c['shares'],
                    'market_cap': c['market_cap'],
                    'weight': weight,
                    'daily_return': c['daily_return'],
                    'days_since_ipo': c['days_since_ipo'],
                })

        index_data.append({
            'date': current_date,
            'num_constituents': len(daily_constituents),
            'total_market_cap': total_market_cap,
            'daily_return': weighted_return,
        })

    return pd.DataFrame(index_data), pd.DataFrame(weights_data)


def fetch_spy_data(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Fetch SPY data for comparison."""
    spy = yf.Ticker('SPY')
    hist = spy.history(start=start_date, end=end_date, auto_adjust=True)
    hist = hist.reset_index()
    hist['date'] = pd.to_datetime(hist['Date'])
    if hist['date'].dt.tz is not None:
        hist['date'] = hist['date'].dt.tz_localize(None)
    hist['daily_return'] = hist['Close'].pct_change()
    return hist[['date', 'Close', 'daily_return']].rename(columns={'Close': 'spy_close'})


def main():
    parser = argparse.ArgumentParser(description="Build Market-Cap Weighted IPO Index")
    parser.add_argument("--csv", type=str, default=".pytest_cache/dailyhistorical_21-26.csv",
                        help="Path to dailyhistorical CSV")
    parser.add_argument("--max_ipos", type=int, default=50,
                        help="Maximum number of IPOs to fetch (default: 50)")
    parser.add_argument("--min_year", type=int, default=2023,
                        help="Minimum IPO year to include (default: 2023)")
    parser.add_argument("--days", type=int, default=180,
                        help="Days after IPO to include in index (default: 180)")
    parser.add_argument("--output", type=str, default="results/ipo_index.csv",
                        help="Output CSV path")
    parser.add_argument("--use-optimizer", action="store_true",
                        help="Use OnlinePortfolioOptimizer (gradient-based) instead of market-cap weights")
    parser.add_argument("--opt-lr", type=float, default=0.01,
                        help="Optimizer learning rate (default: 0.01)")
    parser.add_argument("--opt-window", type=int, default=30,
                        help="Optimizer rolling window size (default: 30)")
    parser.add_argument("--use-gru", action="store_true",
                        help="Use GRU policy (PyTorch) to set weights from return windows; trains on index history")
    parser.add_argument("--gru-window", type=int, default=21,
                        help="GRU lookback window in days (default: 21)")
    parser.add_argument("--gru-epochs", type=int, default=30,
                        help="GRU training epochs (default: 30)")
    parser.add_argument("--gru-hidden", type=int, default=128,
                        help="GRU hidden size (default: 128)")

    args = parser.parse_args()

    if args.use_optimizer and args.use_gru:
        print("Error: use only one of --use-optimizer or --use-gru")
        sys.exit(1)

    print(f"\n{'='*60}")
    mode = " (GRU)" if args.use_gru else (" (Online Optimizer)" if args.use_optimizer else " (Market-Cap Weighted)")
    print("  Building IPO Index" + mode)
    print(f"{'='*60}\n")

    # Load IPO dates
    print(f"Loading IPO data from {args.csv}...")
    ipo_dates = load_ipo_dates(args.csv)
    print(f"Found {len(ipo_dates)} unique IPOs")

    # Filter by year
    ipo_dates = ipo_dates[ipo_dates['ipo_date'].dt.year >= args.min_year]
    print(f"After filtering for year >= {args.min_year}: {len(ipo_dates)} IPOs")

    # Limit number of IPOs
    ipo_dates = ipo_dates.head(args.max_ipos)
    print(f"Processing {len(ipo_dates)} IPOs...\n")

    # Fetch data for each IPO
    successful_ipos = []
    failed_ipos = []

    for i, row in ipo_dates.iterrows():
        ticker = row['ticker']
        ipo_date = row['ipo_date']

        print(f"  [{len(successful_ipos)+1}/{len(ipo_dates)}] Fetching {ticker} (IPO: {ipo_date.date()})...", end=" ")

        result = fetch_ipo_data(ticker, ipo_date, days=args.days)

        if result['success']:
            successful_ipos.append(result)
            print(f"OK ({len(result['prices'])} days, {result['shares_outstanding']:,.0f} shares)")
        else:
            failed_ipos.append({'ticker': ticker, 'reason': result['reason']})
            print(f"FAILED: {result['reason']}")

    print(f"\nSuccessfully fetched: {len(successful_ipos)}/{len(ipo_dates)} IPOs")

    if len(successful_ipos) == 0:
        print("Error: No IPO data fetched successfully")
        sys.exit(1)

    # Determine date range
    min_date = min(ipo['ipo_date'] for ipo in successful_ipos)
    max_date = max(ipo['prices']['date'].max() for ipo in successful_ipos)

    print(f"\nBuilding index from {min_date.date()} to {max_date.date()}...")

    # Build IPO index
    index_df, weights_df = build_ipo_index(
        successful_ipos, min_date, max_date,
        days_window=args.days,
        use_optimizer=args.use_optimizer,
        optimizer_kwargs={'learning_rate': args.opt_lr, 'window_size': args.opt_window},
        use_gru=args.use_gru,
        gru_kwargs={
            'window_size': args.gru_window,
            'n_epochs': args.gru_epochs,
            'hidden_size': args.gru_hidden,
        },
    )
    print(f"Index built with {len(index_df)} trading days")
    print(f"Weights data: {len(weights_df)} records")

    # Calculate cumulative returns
    index_df['cumulative_return'] = (1 + index_df['daily_return']).cumprod()
    index_df['index_value'] = 100 * index_df['cumulative_return']

    # Fetch SPY for comparison
    print("\nFetching SPY for comparison...")
    spy_df = fetch_spy_data(min_date, max_date)
    spy_df['spy_cumulative'] = (1 + spy_df['daily_return']).cumprod()
    spy_df['spy_index'] = 100 * spy_df['spy_cumulative']

    # Merge with SPY
    index_df['date_only'] = index_df['date'].dt.date
    spy_df['date_only'] = spy_df['date'].dt.date
    merged = index_df.merge(spy_df[['date_only', 'spy_close', 'spy_index']], on='date_only', how='left')

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    # Save daily weights
    weights_path = output_path.parent / "ipo_weights.csv"
    weights_df.to_csv(weights_path, index=False)

    # Create pivot table of weights (date x ticker)
    weights_pivot = weights_df.pivot(index='date', columns='ticker', values='weight').fillna(0)
    weights_pivot_path = output_path.parent / "ipo_weights_pivot.csv"
    weights_pivot.to_csv(weights_pivot_path)

    print(f"\nResults saved to:")
    print(f"  - {output_path} (daily index)")
    print(f"  - {weights_path} (daily weights long format)")
    print(f"  - {weights_pivot_path} (daily weights pivot table)")

    # Print summary
    print(f"\n{'='*60}")
    print("  IPO INDEX SUMMARY")
    print(f"{'='*60}")

    first_valid = merged.dropna(subset=['spy_index']).iloc[0]
    last_valid = merged.dropna(subset=['spy_index']).iloc[-1]

    ipo_return = (last_valid['index_value'] / 100) - 1
    spy_return = (last_valid['spy_index'] / 100) - 1
    excess_return = ipo_return - spy_return

    print(f"\nDate Range: {first_valid['date'].date()} to {last_valid['date'].date()}")
    print(f"Constituents: {int(merged['num_constituents'].mean()):.0f} avg, {int(merged['num_constituents'].max())} max")

    print(f"\n{'─'*60}")
    print("RETURNS")
    print(f"{'─'*60}")
    print(f"  IPO Index Return:       {ipo_return*100:+8.2f}%")
    print(f"  S&P 500 Return:         {spy_return*100:+8.2f}%")
    print(f"  {'─'*40}")
    print(f"  Excess Return:          {excess_return*100:+8.2f}%")

    if excess_return > 0:
        print(f"\n  Result: IPO Index BEAT the S&P 500 by {excess_return*100:.2f}%")
    else:
        print(f"\n  Result: IPO Index UNDERPERFORMED the S&P 500 by {abs(excess_return)*100:.2f}%")

    # Calculate fitness
    daily_returns = merged['daily_return'].dropna().values
    spy_daily = spy_df['daily_return'].dropna().values

    # IPO Index fitness
    ipo_annual = np.mean(daily_returns) * 252
    ipo_vol = np.std(daily_returns) * np.sqrt(252)
    ipo_sharpe = (ipo_annual - 0.04) / ipo_vol if ipo_vol > 0 else 0
    ipo_cumret = np.cumprod(1 + daily_returns)
    ipo_maxdd = np.min(ipo_cumret / np.maximum.accumulate(ipo_cumret) - 1)
    ipo_fitness = 0.4 * ipo_sharpe + 0.3 * ipo_annual + 0.3 * (1 + ipo_maxdd)

    # SPY fitness
    spy_annual = np.mean(spy_daily) * 252
    spy_vol = np.std(spy_daily) * np.sqrt(252)
    spy_sharpe = (spy_annual - 0.04) / spy_vol if spy_vol > 0 else 0
    spy_cumret = np.cumprod(1 + spy_daily)
    spy_maxdd = np.min(spy_cumret / np.maximum.accumulate(spy_cumret) - 1)
    spy_fitness = 0.4 * spy_sharpe + 0.3 * spy_annual + 0.3 * (1 + spy_maxdd)

    print(f"\n{'─'*60}")
    print("RISK METRICS")
    print(f"{'─'*60}")
    print(f"  IPO Index Volatility:   {ipo_vol*100:8.2f}%")
    print(f"  S&P 500 Volatility:     {spy_vol*100:8.2f}%")
    print(f"  IPO Index Max DD:       {ipo_maxdd*100:8.2f}%")
    print(f"  S&P 500 Max DD:         {spy_maxdd*100:8.2f}%")

    print(f"\n{'─'*60}")
    print("FITNESS SCORE")
    print("  Formula: 0.4*Sharpe + 0.3*AnnualReturn + 0.3*(1+MaxDD)")
    print(f"{'─'*60}")
    print(f"  IPO Index Fitness:      {ipo_fitness:8.3f}")
    print(f"  S&P 500 Fitness:        {spy_fitness:8.3f}")

    if ipo_fitness > spy_fitness:
        print(f"\n  Result: IPO Index has HIGHER fitness than S&P 500")
    else:
        print(f"\n  Result: IPO Index has LOWER fitness than S&P 500")

    # Show latest day weights
    print(f"\n{'─'*60}")
    print("LATEST DAY WEIGHTS (Top 10)")
    print(f"{'─'*60}")

    latest_date = weights_df['date'].max()
    latest_weights = weights_df[weights_df['date'] == latest_date].sort_values('weight', ascending=False)

    print(f"\n  Date: {latest_date.date()}")
    print(f"  {'Ticker':<10} {'Weight':>10} {'Price':>10} {'Market Cap':>15}")
    print(f"  {'-'*50}")

    for _, row in latest_weights.head(10).iterrows():
        print(f"  {row['ticker']:<10} {row['weight']*100:>9.2f}% ${row['price']:>8.2f} ${row['market_cap']/1e9:>12.2f}B")

    print(f"\n  Total constituents: {len(latest_weights)}")
    print(f"  Top 10 weight: {latest_weights.head(10)['weight'].sum()*100:.1f}%")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
