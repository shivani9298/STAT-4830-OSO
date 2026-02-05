#!/usr/bin/env python3
"""
Run PyTorch policy training (REINFORCE + Adam). Course: SGD, Adam, validation.
"""

import argparse
from datetime import date, timedelta
from pathlib import Path
import numpy as np

from src.data import (
    load_ipo_meta,
    load_prices_dir,
    load_prices_from_yfinance,
    build_episodes,
    build_episodes_from_rich_csv,
    generate_synthetic_prices,
    Episode,
)
from src.train_policy import train_reinforce


def create_synthetic_episodes(n: int = 100, N: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    episodes = []
    base_date = date(2020, 1, 1)
    for i in range(n):
        ticker = f"SYNTH{i:03d}"
        ipo_date = base_date + timedelta(days=i * 7)
        price_df = generate_synthetic_prices(
            ticker=ticker, ipo_date=ipo_date, N=N,
            initial_price=float(rng.uniform(10, 100)),
            volatility=float(rng.uniform(0.01, 0.05)), rng=rng,
        )
        ep = Episode(ticker=ticker, ipo_date=ipo_date, df=price_df, day0_index=0, N=N)
        episodes.append(ep)
    return episodes


def main():
    p = argparse.ArgumentParser(description="Train PyTorch IPO policy (REINFORCE + Adam)")
    p.add_argument("--data", choices=["synth", "path", "yfinance"], default="synth")
    p.add_argument("--meta_csv", type=str, default=None)
    p.add_argument("--rich_csv", type=str, default=None,
                   help="Path to rich IPO CSV (firstday_*, inweek_*, sector, etc.). If set with --data=path, uses this instead of meta+prices.")
    p.add_argument("--ritter_csv", type=str, default=None, help="Optional Ritter CSV (not used with --data yfinance; yfinance uses S&P 500)")
    p.add_argument("--max_tickers", type=int, default=None, help="Max tickers with --data yfinance (default: all S&P 500)")
    p.add_argument("--lookback_days", type=int, default=252, help="Days of history for --data yfinance")
    p.add_argument("--prices_dir", type=str, default=None)
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_schedule", choices=["constant", "cosine", "step"], default="constant")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--cost_bps", type=float, default=10.0)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--reward_type", choices=["net_ret", "score"], default="net_ret",
                   help="REINFORCE reward: net_ret = per-episode return; score = full fitness (E[R]-λ·CVaR-κ·Cost-μ·MDD) per batch")
    args = p.parse_args()

    if args.data == "synth":
        episodes = create_synthetic_episodes(n=100, N=args.N, seed=args.seed)
    elif args.data == "yfinance":
        import pandas as pd
        from datetime import date, timedelta
        from src.data import get_sp500_tickers
        print("Fetching S&P 500 constituent list...")
        tickers = get_sp500_tickers()
        if args.max_tickers is not None:
            tickers = tickers[: args.max_tickers]
        end_date = date.today()
        start_date = end_date - timedelta(days=args.lookback_days)
        meta_df = pd.DataFrame({"ticker": tickers, "ipo_date": [start_date] * len(tickers)})
        print(f"Fetching prices from Yahoo Finance for {len(meta_df)} S&P 500 tickers...")
        prices_map = load_prices_from_yfinance(
            meta_df, N=args.N, buffer_days=5, fetch_days=args.lookback_days
        )
        print(f"Fetched {len(prices_map)} tickers")
        episodes = build_episodes(meta_df, prices_map, N=args.N, short_mode="skip")
    else:
        rich_path = Path(args.rich_csv) if args.rich_csv else Path("archive-3/ipo_stock_2010_2018_v2.csv")
        if rich_path.exists():
            print(f"Using rich IPO CSV: {rich_path}")
            episodes = build_episodes_from_rich_csv(rich_path)
        else:
            meta_path = Path(args.meta_csv or "archive-3/ipo_clean_2010_2018.csv")
            if not meta_path.exists():
                raise SystemExit("When --data=path, provide --rich_csv or --meta_csv (and optionally --prices_dir)")
            meta_df = load_ipo_meta(meta_path)
            if args.prices_dir:
                prices_map = load_prices_dir(args.prices_dir)
            else:
                rng = np.random.default_rng(args.seed)
                prices_map = {}
                for _, row in meta_df.iterrows():
                    t, d = row["ticker"], row["ipo_date"]
                    prices_map[t] = generate_synthetic_prices(
                        t, d, args.N + 5, rng.uniform(10, 100), rng.uniform(0.01, 0.05), rng
                    )
            episodes = build_episodes(meta_df, prices_map, N=args.N, short_mode="skip")

    n = len(episodes)
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val
    perm = np.random.RandomState(args.seed).permutation(n)
    train_ep = [episodes[i] for i in perm[:n_train]]
    val_ep = [episodes[i] for i in perm[n_train:]]

    print(f"Train {n_train} episodes, val {n_val} episodes")
    result = train_reinforce(
        train_ep,
        val_episodes=val_ep,
        n_epochs=args.n_epochs,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        cost_bps=args.cost_bps,
        lam=args.lam,
        alpha=args.alpha,
        kappa=args.kappa,
        mu=args.mu,
        batch_size=min(args.batch_size, n_train),
        seed=args.seed,
        out_dir=Path(args.out_dir),
        reward_type=args.reward_type,
    )
    print("Done. Policy saved to", Path(args.out_dir) / "policy_network.pt")


if __name__ == "__main__":
    main()
