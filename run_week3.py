#!/usr/bin/env python3
"""
CLI runner for IPO trading strategy optimization.
Owned by Person D.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import load_ipo_meta, load_prices_dir, build_episodes, generate_synthetic_prices
from src.optimize import random_search, baseline_always_skip, baseline_always_participate, baseline_fixed_hold_k


def create_synthetic_data(n_episodes: int = 50, N: int = 10, seed: int = 0):
    """Create synthetic IPO episodes for testing."""
    rng = np.random.default_rng(seed)
    
    episodes = []
    base_date = pd.Timestamp('2020-01-01').date()
    
    for i in range(n_episodes):
        ticker = f"SYNTH{i:03d}"
        ipo_date = base_date + pd.Timedelta(days=i * 7)  # Weekly IPOs
        
        price_df = generate_synthetic_prices(
            ticker=ticker,
            ipo_date=ipo_date,
            N=N,
            initial_price=rng.uniform(10, 100),
            volatility=rng.uniform(0.01, 0.05),
            rng=rng
        )
        
        from src.data import Episode
        episode = Episode(
            ticker=ticker,
            ipo_date=ipo_date,
            df=price_df,
            day0_index=0,
            N=N
        )
        episodes.append(episode)
    
    return episodes


def main():
    parser = argparse.ArgumentParser(description="IPO Trading Strategy Optimization")
    
    # Data arguments
    parser.add_argument("--data", type=str, default="synth", choices=["synth", "path"],
                        help="Data source: 'synth' for synthetic, 'path' for file path")
    parser.add_argument("--prices_dir", type=str, default=None,
                        help="Directory containing price CSV files")
    parser.add_argument("--meta_csv", type=str, default=None,
                        help="Path to IPO metadata CSV file")
    parser.add_argument("--N", type=int, default=10,
                        help="Number of days in episode window")
    
    # Optimization arguments
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of optimization trials")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    # Objective function arguments
    parser.add_argument("--lam", type=float, default=1.0,
                        help="CVaR penalty weight (λ)")
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="CVaR confidence level (α)")
    parser.add_argument("--kappa", type=float, default=1.0,
                        help="Cost penalty weight (κ)")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="MDD penalty weight (μ)")
    parser.add_argument("--cost_bps", type=float, default=10.0,
                        help="Transaction cost in basis points")
    
    # Output arguments
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load or generate episodes
    if args.data == "synth":
        print("Generating synthetic data...")
        episodes = create_synthetic_data(n_episodes=50, N=args.N, seed=args.seed)
        print(f"Created {len(episodes)} synthetic episodes")
    else:
        if args.meta_csv is None:
            # Try to use archive-3 data
            meta_path = Path("archive-3/ipo_clean_2010_2018.csv")
            if not meta_path.exists():
                print(f"Error: --meta_csv required when --data=path")
                sys.exit(1)
        else:
            meta_path = Path(args.meta_csv)
        
        if args.prices_dir is None:
            # Generate synthetic prices for each IPO
            print(f"Loading IPO metadata from {meta_path}...")
            meta_df = load_ipo_meta(meta_path)
            print(f"Loaded {len(meta_df)} IPOs")
            
            # Generate synthetic prices
            print("Generating synthetic price data...")
            rng = np.random.default_rng(args.seed)
            prices_map = {}
            for _, row in meta_df.iterrows():
                ticker = row['ticker']
                ipo_date = row['ipo_date']
                price_df = generate_synthetic_prices(
                    ticker=ticker,
                    ipo_date=ipo_date,
                    N=args.N + 5,  # Extra days for safety
                    initial_price=rng.uniform(10, 100),
                    volatility=rng.uniform(0.01, 0.05),
                    rng=rng
                )
                prices_map[ticker] = price_df
        else:
            prices_dir = Path(args.prices_dir)
            meta_df = load_ipo_meta(meta_path)
            prices_map = load_prices_dir(prices_dir)
        
        print("Building episodes...")
        episodes = build_episodes(meta_df, prices_map, N=args.N, short_mode="skip")
        print(f"Built {len(episodes)} episodes")
    
    if len(episodes) == 0:
        print("Error: No episodes available for optimization")
        sys.exit(1)
    
    # Prepare objective kwargs
    objective_kwargs = {
        "lam": args.lam,
        "alpha": args.alpha,
        "kappa": args.kappa,
        "mu": args.mu,
        "cost_bps": args.cost_bps
    }
    
    # Run optimization
    print(f"\nRunning random search with {args.trials} trials...")
    out_dir = Path(args.out_dir)
    results = random_search(
        episodes=episodes,
        n_trials=args.trials,
        seed=args.seed,
        objective_kwargs=objective_kwargs,
        out_dir=out_dir
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nBest Score: {results['best_score']:.6f}")
    print(f"\nBest Metrics:")
    for key, value in results['best_metrics'].items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nBest Parameters:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")
    
    print(f"\nTop 5 Trials:")
    for i, record in enumerate(results['leaderboard'], 1):
        print(f"\n  {i}. Score: {record['score']:.6f}")
        print(f"     E[R]: {record['metrics']['E[R]']:.6f}, "
              f"CVaR: {record['metrics']['CVaR']:.6f}, "
              f"MDD: {record['metrics']['MDD']:.6f}")
    
    # Run baselines
    print("\n" + "="*60)
    print("BASELINE COMPARISONS")
    print("="*60)
    
    baselines = [
        baseline_always_skip(episodes),
        baseline_always_participate(episodes, weight=0.1),
        baseline_fixed_hold_k(episodes, hold_k=1, weight=0.1),
        baseline_fixed_hold_k(episodes, hold_k=5, weight=0.1),
    ]
    
    for baseline in baselines:
        print(f"\n{baseline['name']}:")
        print(f"  Score: {baseline['score']:.6f}")
        print(f"  E[R]: {baseline['metrics']['E[R]']:.6f}, "
              f"CVaR: {baseline['metrics']['CVaR']:.6f}, "
              f"MDD: {baseline['metrics']['MDD']:.6f}")
    
    print(f"\n\nResults saved to: {out_dir}/")
    print(f"  - best_params.json")
    print(f"  - trials.jsonl")
    print(f"  - results.csv")


if __name__ == "__main__":
    main()
