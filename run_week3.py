#!/usr/bin/env python3
"""
Simple stock vs S&P 500 comparison.
Buy-and-hold analysis: Did the stock beat the market?
"""

import argparse
import sys
import numpy as np
import pandas as pd

from src.data import fetch_stock_vs_benchmark


def calculate_fitness(returns, risk_free_rate=0.04):
    """
    Calculate the fitness score for a portfolio.

    Fitness = 0.4 * Sharpe + 0.3 * Annual Return + 0.3 * (1 + MaxDrawdown)

    Parameters:
        returns (np.array): Array of daily portfolio returns
        risk_free_rate (float): Annual risk-free rate (default 4%)

    Returns:
        dict: Fitness score and components
    """
    if len(returns) == 0:
        return {'fitness': 0, 'sharpe': 0, 'annual_return': 0, 'max_drawdown': 0}

    # Annualized return
    portfolio_return = np.mean(returns) * 252

    # Annualized volatility
    portfolio_volatility = np.std(returns) * np.sqrt(252)

    # Sharpe ratio
    if portfolio_volatility > 0:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    else:
        sharpe_ratio = 0

    # Maximum drawdown calculation
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    # Combined fitness score
    fitness = (0.4 * sharpe_ratio +
               0.3 * portfolio_return +
               0.3 * (1 + max_drawdown))  # Add 1 to make max_drawdown positive

    return {
        'fitness': fitness,
        'sharpe': sharpe_ratio,
        'annual_return': portfolio_return,
        'max_drawdown': max_drawdown,
        'volatility': portfolio_volatility,
    }


def calculate_metrics(df: pd.DataFrame, ticker: str, benchmark: str = "SPY") -> dict:
    """Calculate performance metrics for stock vs benchmark."""

    # Total returns
    stock_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    bench_return = (df['benchmark_close'].iloc[-1] / df['benchmark_close'].iloc[0]) - 1
    excess_return = stock_return - bench_return

    # Daily returns
    daily_stock = df['close'].pct_change().dropna()
    daily_bench = df['benchmark_close'].pct_change().dropna()
    daily_excess = daily_stock - daily_bench

    # Volatility (annualized)
    stock_vol = daily_stock.std() * np.sqrt(252)
    bench_vol = daily_bench.std() * np.sqrt(252)
    excess_vol = daily_excess.std() * np.sqrt(252)

    # Sharpe ratio (assuming 0 risk-free rate for simplicity)
    trading_days = len(df)
    annual_factor = 252 / trading_days
    stock_sharpe = (stock_return * annual_factor) / stock_vol if stock_vol > 0 else 0
    bench_sharpe = (bench_return * annual_factor) / bench_vol if bench_vol > 0 else 0

    # Information ratio (excess return / tracking error)
    info_ratio = (excess_return * annual_factor) / excess_vol if excess_vol > 0 else 0

    # Max drawdown
    stock_cummax = (1 + daily_stock).cumprod().cummax()
    stock_drawdown = ((1 + daily_stock).cumprod() / stock_cummax) - 1
    stock_max_dd = stock_drawdown.min()

    bench_cummax = (1 + daily_bench).cumprod().cummax()
    bench_drawdown = ((1 + daily_bench).cumprod() / bench_cummax) - 1
    bench_max_dd = bench_drawdown.min()

    # Win rate (days stock beat benchmark)
    win_days = (daily_excess > 0).sum()
    total_days = len(daily_excess)
    win_rate = win_days / total_days if total_days > 0 else 0

    # Beta (stock vs benchmark)
    covariance = daily_stock.cov(daily_bench)
    bench_variance = daily_bench.var()
    beta = covariance / bench_variance if bench_variance > 0 else 1

    # Alpha (annualized)
    alpha = (stock_return - beta * bench_return) * annual_factor

    # Calculate fitness scores
    stock_fitness = calculate_fitness(daily_stock.values)
    bench_fitness = calculate_fitness(daily_bench.values)
    excess_fitness = calculate_fitness(daily_excess.values)

    return {
        'ticker': ticker,
        'benchmark': benchmark,
        'trading_days': trading_days,
        'stock_return': stock_return,
        'bench_return': bench_return,
        'excess_return': excess_return,
        'stock_vol': stock_vol,
        'bench_vol': bench_vol,
        'stock_sharpe': stock_sharpe,
        'bench_sharpe': bench_sharpe,
        'info_ratio': info_ratio,
        'stock_max_dd': stock_max_dd,
        'bench_max_dd': bench_max_dd,
        'win_rate': win_rate,
        'beta': beta,
        'alpha': alpha,
        'stock_fitness': stock_fitness,
        'bench_fitness': bench_fitness,
        'excess_fitness': excess_fitness,
    }


def main():
    parser = argparse.ArgumentParser(description="Stock vs S&P 500 Buy-and-Hold Comparison")

    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Stock ticker to analyze (e.g., AAPL, GOOGL, MSFT)")
    parser.add_argument("--benchmark", type=str, default="SPY",
                        help="Benchmark ticker (default: SPY for S&P 500)")
    parser.add_argument("--period", type=str, default="1y",
                        help="Historical period: 1mo, 3mo, 6mo, 1y, 2y, 5y, max")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  {args.ticker} vs {args.benchmark} (S&P 500)")
    print(f"  Period: {args.period}")
    print(f"{'='*60}\n")

    try:
        df = fetch_stock_vs_benchmark(args.ticker, args.benchmark, args.period)
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

    metrics = calculate_metrics(df, args.ticker, args.benchmark)

    # Print results
    print(f"Date Range: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
    print(f"Trading Days: {metrics['trading_days']}")

    print(f"\n{'─'*60}")
    print("RETURNS")
    print(f"{'─'*60}")
    print(f"  {args.ticker:8} Total Return:    {metrics['stock_return']*100:+8.2f}%")
    print(f"  {args.benchmark:8} Total Return:    {metrics['bench_return']*100:+8.2f}%")
    print(f"  {'─'*40}")
    print(f"  Excess Return:          {metrics['excess_return']*100:+8.2f}%")

    if metrics['excess_return'] > 0:
        print(f"\n  Result: {args.ticker} BEAT the S&P 500 by {metrics['excess_return']*100:.2f}%")
    elif metrics['excess_return'] < 0:
        print(f"\n  Result: {args.ticker} UNDERPERFORMED the S&P 500 by {abs(metrics['excess_return'])*100:.2f}%")
    else:
        print(f"\n  Result: {args.ticker} MATCHED the S&P 500")

    print(f"\n{'─'*60}")
    print("RISK METRICS")
    print(f"{'─'*60}")
    print(f"  {args.ticker:8} Volatility (ann): {metrics['stock_vol']*100:8.2f}%")
    print(f"  {args.benchmark:8} Volatility (ann): {metrics['bench_vol']*100:8.2f}%")
    print(f"  {args.ticker:8} Max Drawdown:     {metrics['stock_max_dd']*100:8.2f}%")
    print(f"  {args.benchmark:8} Max Drawdown:     {metrics['bench_max_dd']*100:8.2f}%")

    print(f"\n{'─'*60}")
    print("RISK-ADJUSTED METRICS")
    print(f"{'─'*60}")
    print(f"  {args.ticker:8} Sharpe Ratio:    {metrics['stock_sharpe']:8.2f}")
    print(f"  {args.benchmark:8} Sharpe Ratio:    {metrics['bench_sharpe']:8.2f}")
    print(f"  Information Ratio:      {metrics['info_ratio']:8.2f}")
    print(f"  Beta:                   {metrics['beta']:8.2f}")
    print(f"  Alpha (annualized):     {metrics['alpha']*100:+8.2f}%")

    print(f"\n{'─'*60}")
    print("CONSISTENCY")
    print(f"{'─'*60}")
    print(f"  Daily Win Rate:         {metrics['win_rate']*100:8.1f}%")
    print(f"  (Days {args.ticker} beat {args.benchmark})")

    print(f"\n{'─'*60}")
    print("FITNESS SCORE")
    print("  Formula: 0.4*Sharpe + 0.3*AnnualReturn + 0.3*(1+MaxDD)")
    print(f"{'─'*60}")
    sf = metrics['stock_fitness']
    bf = metrics['bench_fitness']
    ef = metrics['excess_fitness']
    print(f"  {args.ticker:8} Fitness:         {sf['fitness']:8.3f}")
    print(f"  {args.benchmark:8} Fitness:         {bf['fitness']:8.3f}")
    print(f"  {'─'*40}")
    print(f"  Excess Fitness:         {ef['fitness']:8.3f}")

    if sf['fitness'] > bf['fitness']:
        print(f"\n  Result: {args.ticker} has HIGHER fitness than {args.benchmark}")
    elif sf['fitness'] < bf['fitness']:
        print(f"\n  Result: {args.ticker} has LOWER fitness than {args.benchmark}")
    else:
        print(f"\n  Result: {args.ticker} and {args.benchmark} have EQUAL fitness")

    print(f"\n{'='*60}\n")

    # Price chart using ASCII
    print("PRICE PERFORMANCE (normalized to 100)")
    print(f"{'─'*60}")

    # Normalize prices to start at 100
    stock_norm = 100 * df['close'] / df['close'].iloc[0]
    bench_norm = 100 * df['benchmark_close'] / df['benchmark_close'].iloc[0]

    # Sample ~20 points for ASCII chart
    n_points = min(20, len(df))
    indices = np.linspace(0, len(df)-1, n_points, dtype=int)

    all_vals = list(stock_norm.iloc[indices]) + list(bench_norm.iloc[indices])
    min_val, max_val = min(all_vals), max(all_vals)
    chart_width = 40

    def val_to_pos(val):
        if max_val == min_val:
            return chart_width // 2
        return int((val - min_val) / (max_val - min_val) * (chart_width - 1))

    print(f"\n  {'Start':<10} {'─'*chart_width} End")

    for i, idx in enumerate(indices):
        date_str = df['date'].iloc[idx].strftime('%m/%d')
        s_val = stock_norm.iloc[idx]
        b_val = bench_norm.iloc[idx]
        s_pos = val_to_pos(s_val)
        b_pos = val_to_pos(b_val)

        line = [' '] * chart_width
        # Place benchmark first, then stock (stock overwrites if same position)
        line[b_pos] = 'S'  # SPY
        line[s_pos] = '*'  # Stock
        if s_pos == b_pos:
            line[s_pos] = 'X'  # Overlap

        print(f"  {date_str:6} |{''.join(line)}| {s_val:.0f} vs {b_val:.0f}")

    print(f"\n  Legend: * = {args.ticker}, S = {args.benchmark}, X = overlap")
    print()


if __name__ == "__main__":
    main()
