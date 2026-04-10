#!/usr/bin/env python3
"""
5-Model Comparison on Real WRDS Data with Train/Val/Test Splits.

Compares 5 model architectures on real IPO data:
  1. Market-Cap Weighted         (baseline, no learning)
  2. OnlinePortfolioOptimizer    (gradient-based online learning)
  3. GRUPolicy                   (recurrent NN, per-split in-sample)
  4. IPOPolicyNetwork / REINFORCE (MLP, trained on train episodes)
  5. PolicyParams                 (rule-based heuristic, no training)

Usage:
    python3 scripts/run_comparison.py
    python3 scripts/run_comparison.py --max_index_ipos 15 --no_gru
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ─────────────────────────────────────────────────────────────
# Config defaults
# ─────────────────────────────────────────────────────────────
WRDS_CSV        = str(_REPO_ROOT / "src/dailyhistorical_21-26.csv")
MAX_INDEX_IPOS  = 20          # IPOs fetched with shares_outstanding (Models 1-3)
MAX_EPISODE_IPOS = 50         # IPOs used for episode-based models (4-5)
N_EPISODE       = 21          # episode length (trading days after IPO)
GRU_EPOCHS      = 30
GRU_HIDDEN      = 64
GRU_WINDOW      = 21
REINFORCE_EPOCHS = 30
REINFORCE_BATCH  = 16
RISK_FREE        = 0.04
RESULTS_DIR      = _REPO_ROOT / "results"

SPLITS = {
    "train": ("2021-01-01", "2021-12-31"),
    "val":   ("2022-01-01", "2023-12-31"),
    "test":  ("2024-01-01", "2025-12-31"),
}


# ─────────────────────────────────────────────────────────────
# 1. Data helpers
# ─────────────────────────────────────────────────────────────

def load_wrds_ipo_dates(csv_path: str) -> pd.DataFrame:
    """Load WRDS CSV and return one row per ticker with first trading date."""
    df = pd.read_csv(csv_path)
    df["datadate"] = pd.to_datetime(df["datadate"])
    ipo_dates = (
        df.groupby("tic")
        .agg(
            ipo_date=("datadate", "min"),
            ipo_price=("prccd", "first"),
            gvkey=("gvkey", "first"),
        )
        .reset_index()
        .rename(columns={"tic": "ticker"})
    )
    ipo_dates = ipo_dates.sort_values("ipo_date").reset_index(drop=True)
    ipo_dates["ipo_year"] = ipo_dates["ipo_date"].dt.year
    return ipo_dates


def split_by_year(ipo_dates: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split IPO tickers into train (≤2021) / val (2022-2023) / test (2024-2025)."""
    train = ipo_dates[ipo_dates["ipo_year"] <= 2021].copy()
    val   = ipo_dates[(ipo_dates["ipo_year"] >= 2022) & (ipo_dates["ipo_year"] <= 2023)].copy()
    test  = ipo_dates[ipo_dates["ipo_year"] >= 2024].copy()
    return {"train": train, "val": val, "test": test}


def fetch_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch daily returns for a benchmark ticker from yfinance."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]
    rets = close.pct_change().dropna()
    rets.index = pd.to_datetime(rets.index).tz_localize(None)
    rets.name = ticker
    return rets


# ─────────────────────────────────────────────────────────────
# 2. Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    returns: np.ndarray,
    spy_returns: Optional[np.ndarray] = None,
    dji_returns: Optional[np.ndarray] = None,
    risk_free: float = RISK_FREE,
    returns_are_daily: bool = True,
) -> dict:
    """
    Compute portfolio performance metrics.

    For index-based models (daily returns): uses 252 trading days/year.
    For episode-based models (per-episode returns): scales by 252/N_EPISODE.
    """
    empty = dict(
        total_return=0.0, ann_return=0.0, volatility=0.0,
        sharpe=0.0, max_drawdown=0.0, excess_spy=0.0, excess_dji=0.0,
    )
    if returns is None or len(returns) == 0:
        return empty

    returns = np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n_per_year = 252.0 if returns_are_daily else (252.0 / N_EPISODE)

    total_return = float(np.prod(1 + returns) - 1)
    ann_return   = float(np.mean(returns) * n_per_year)
    vol          = float(np.std(returns) * np.sqrt(n_per_year))
    sharpe       = (ann_return - risk_free) / vol if vol > 1e-12 else 0.0

    cum         = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(np.clip(cum, 1e-10, None))
    max_dd      = float(np.min((cum - running_max) / running_max))

    def _tot(arr):
        if arr is None or len(arr) == 0:
            return None
        arr = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0)
        return float(np.prod(1 + arr) - 1)

    spy_tot = _tot(spy_returns)
    dji_tot = _tot(dji_returns)
    excess_spy = (total_return - spy_tot) if spy_tot is not None else 0.0
    excess_dji = (total_return - dji_tot) if dji_tot is not None else 0.0

    return dict(
        total_return=total_return,
        ann_return=ann_return,
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        excess_spy=excess_spy,
        excess_dji=excess_dji,
    )


def compute_metrics_aligned(
    index_df: pd.DataFrame,
    spy_series: pd.Series,
    dji_series: pd.Series,
) -> dict:
    """
    Align index_df (date + daily_return) with SPY/DJI by date, then compute metrics.
    Used for index-based models where proper date alignment is possible.
    """
    if len(index_df) == 0:
        return compute_metrics(np.array([]))

    spy_df = pd.DataFrame({"date": spy_series.index, "spy_ret": spy_series.values})
    dji_df = pd.DataFrame({"date": dji_series.index, "dji_ret": dji_series.values})

    merged = index_df[["date", "daily_return"]].copy()
    merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
    spy_df["date"] = pd.to_datetime(spy_df["date"]).dt.tz_localize(None)
    dji_df["date"] = pd.to_datetime(dji_df["date"]).dt.tz_localize(None)

    merged = merged.merge(spy_df, on="date", how="left")
    merged = merged.merge(dji_df, on="date", how="left")

    rets     = merged["daily_return"].fillna(0).values
    spy_rets = merged["spy_ret"].fillna(0).values
    dji_rets = merged["dji_ret"].fillna(0).values

    return compute_metrics(rets, spy_rets, dji_rets, returns_are_daily=True)


# ─────────────────────────────────────────────────────────────
# 3. Fetch IPO data (shares outstanding) for Models 1-3
# ─────────────────────────────────────────────────────────────

def fetch_split_ipo_data(split_df: pd.DataFrame, max_ipos: int) -> list:
    """
    Fetch price data + shares outstanding for up to max_ipos IPOs in a split.
    Uses fetch_ipo_data from build_ipo_index (calls yf.Ticker().info for shares).
    """
    from build_ipo_index import fetch_ipo_data

    subset = split_df.head(max_ipos)
    results = []
    n = len(subset)
    for i, (_, row) in enumerate(subset.iterrows()):
        ticker   = row["ticker"]
        ipo_date = pd.Timestamp(row["ipo_date"])
        print(f"  [{i+1}/{n}] {ticker} ({ipo_date.date()})...", end=" ", flush=True)
        res = fetch_ipo_data(ticker, ipo_date, days=180)
        if res["success"]:
            results.append(res)
            print(f"OK ({len(res['prices'])} days)")
        else:
            print(f"FAILED: {res['reason']}")
    return results


# ─────────────────────────────────────────────────────────────
# 4. Model runners
# ─────────────────────────────────────────────────────────────

def run_index_model(
    ipo_data: list,
    split: str,
    use_optimizer: bool = False,
    use_gru: bool = False,
) -> pd.DataFrame:
    """
    Run market-cap / optimizer / GRU index model.
    Returns index_df (columns: date, daily_return) — empty DataFrame if no data.
    """
    from build_ipo_index import build_ipo_index

    if not ipo_data:
        return pd.DataFrame(columns=["date", "daily_return"])

    min_date = min(d["ipo_date"] for d in ipo_data)
    max_date = max(d["prices"]["date"].max() for d in ipo_data)

    gru_kwargs = {
        "window_size": GRU_WINDOW,
        "n_epochs":    GRU_EPOCHS,
        "hidden_size": GRU_HIDDEN,
    } if use_gru else {}

    index_df, _ = build_ipo_index(
        ipo_data,
        min_date,
        max_date,
        days_window=180,
        use_optimizer=use_optimizer,
        optimizer_kwargs={"learning_rate": 0.01, "window_size": 30} if use_optimizer else {},
        use_gru=use_gru,
        gru_kwargs=gru_kwargs,
    )

    if len(index_df) == 0:
        return pd.DataFrame(columns=["date", "daily_return"])

    # Save index CSV
    tag = "gru" if use_gru else ("optimizer" if use_optimizer else "mktcap")
    out_path = RESULTS_DIR / f"ipo_index_{tag}_{split}.csv"
    index_df.to_csv(out_path, index=False)

    return index_df[["date", "daily_return"]].copy()


def run_reinforce_model(
    train_episodes: list,
    eval_episodes: list,
    n_epochs: int = REINFORCE_EPOCHS,
    batch_size: int = REINFORCE_BATCH,
) -> Tuple[np.ndarray, dict]:
    """
    Train REINFORCE (IPOPolicyNetwork) on train_episodes, apply to eval_episodes.
    Returns (net_returns_per_episode, training_summary).
    """
    if not train_episodes:
        print("  REINFORCE: no train episodes.")
        return np.array([]), {}

    try:
        import torch
    except ImportError:
        print("  REINFORCE: torch not available, skipping.")
        return np.array([]), {}

    from src.train_policy import train_reinforce
    from src.backtest import backtest_all_with_decisions
    from src.features import episodes_to_tensor

    # Pass a small subset of eval as validation during training
    val_for_train = eval_episodes[: min(15, len(eval_episodes))]

    result = train_reinforce(
        train_episodes,
        val_episodes=val_for_train,
        n_epochs=n_epochs,
        lr=1e-3,
        batch_size=min(batch_size, len(train_episodes)),
        seed=0,
        reward_type="net_ret",
    )

    policy  = result["policy"]
    device  = result["device"]
    summary = result["summary"]

    if not eval_episodes:
        return np.array([]), summary

    policy.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    with torch.no_grad():
        x         = episodes_to_tensor(eval_episodes, device)
        decisions, _ = policy.sample_actions(x, eval_episodes, gen)

    res_df, _ = backtest_all_with_decisions(eval_episodes, decisions)
    return res_df["net_ret"].values, summary


def run_policy_params_model(eval_episodes: list) -> np.ndarray:
    """Apply default PolicyParams (no training) to eval_episodes."""
    from src.policy import PolicyParams, decide_trade
    from src.backtest import backtest_episode

    params = PolicyParams()
    rets = []
    for ep in eval_episodes:
        dec = decide_trade(ep, params)
        r   = backtest_episode(ep, dec)
        rets.append(r["net_ret"])
    return np.array(rets)


# ─────────────────────────────────────────────────────────────
# 5. Print table
# ─────────────────────────────────────────────────────────────

def print_table(rows: list, title: str):
    W = 97
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    header = (
        f"  {'Model':<26} {'Tot Ret':>10} {'Ann Ret':>10} {'Vol':>8} "
        f"{'Sharpe':>8} {'MaxDD':>9} {'vs SPY':>8} {'vs DJI':>8}"
    )
    print(header)
    print(f"  {'-'*93}")
    for row in rows:
        m = row["metrics"]
        print(
            f"  {row['name']:<26} "
            f"{m['total_return']*100:>+9.2f}% "
            f"{m['ann_return']*100:>+9.2f}% "
            f"{m['volatility']*100:>7.2f}% "
            f"{m['sharpe']:>8.3f} "
            f"{m['max_drawdown']*100:>+8.2f}% "
            f"{m['excess_spy']*100:>+7.2f}% "
            f"{m['excess_dji']*100:>+7.2f}%"
        )
    print(f"{'='*W}")


# ─────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="5-Model IPO Strategy Comparison")
    parser.add_argument("--csv", default=WRDS_CSV,
                        help="Path to WRDS dailyhistorical CSV")
    parser.add_argument("--max_index_ipos", type=int, default=MAX_INDEX_IPOS,
                        help="Max IPOs per split for index models (Models 1-3; fetches shares outstanding)")
    parser.add_argument("--max_episode_ipos", type=int, default=MAX_EPISODE_IPOS,
                        help="Max IPOs per split for episode models (Models 4-5)")
    parser.add_argument("--no_gru", action="store_true",
                        help="Skip Model 3 (GRU) — useful if PyTorch not installed")
    parser.add_argument("--gru_epochs", type=int, default=GRU_EPOCHS)
    parser.add_argument("--reinforce_epochs", type=int, default=REINFORCE_EPOCHS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("  5-Model IPO Strategy Comparison — Real WRDS Data")
    print(f"{'='*65}\n")

    # ── Load & split data ───────────────────────────────────────
    print(f"Loading IPO data from {args.csv} ...")
    if not Path(args.csv).exists():
        sys.exit(f"ERROR: {args.csv} not found. Run from repo root.")

    ipo_all = load_wrds_ipo_dates(args.csv)
    splits  = split_by_year(ipo_all)
    print(f"  Total unique IPOs: {len(ipo_all)}")
    for s, df in splits.items():
        print(f"  {s:5s}: {len(df):4d} IPOs  ({df['ipo_year'].min()}-{df['ipo_year'].max()})")

    # ── Fetch benchmarks ────────────────────────────────────────
    print("\nFetching benchmark data (SPY and ^DJI, 2021-2025)...")
    spy = fetch_benchmark("SPY",  "2021-01-01", "2025-12-31")
    dji = fetch_benchmark("^DJI", "2021-01-01", "2025-12-31")
    print(f"  SPY: {len(spy)} trading days")
    print(f"  DJI: {len(dji)} trading days")

    # ── Fetch IPO data with shares outstanding (Models 1-3) ─────
    ipo_data_cache = {}
    for split_name, split_df in splits.items():
        n_req = min(args.max_index_ipos, len(split_df))
        print(f"\nFetching {n_req} IPOs (shares + prices) for '{split_name}' split ...")
        ipo_data_cache[split_name] = fetch_split_ipo_data(split_df, n_req)
        n_ok = len(ipo_data_cache[split_name])
        print(f"  => {n_ok}/{n_req} successful")

    # ── Fetch price data for episode-based models (4-5) ─────────
    from src.data import load_prices_from_yfinance, build_episodes

    episodes_cache = {}
    for split_name, split_df in splits.items():
        subset = split_df.head(args.max_episode_ipos)
        print(f"\nFetching episode prices for '{split_name}' ({len(subset)} tickers)...")
        meta = subset[["ticker", "ipo_date"]].copy()
        meta["ipo_date"] = pd.to_datetime(meta["ipo_date"])

        prices_map = load_prices_from_yfinance(meta, N=N_EPISODE)
        print(f"  Got prices for {len(prices_map)}/{len(meta)} tickers")

        meta_ok = meta[meta["ticker"].isin(prices_map)].copy()
        meta_ok["ipo_date"] = meta_ok["ipo_date"].dt.date
        eps = build_episodes(meta_ok, prices_map, N=N_EPISODE)
        episodes_cache[split_name] = eps
        print(f"  Built {len(eps)} episodes (N={N_EPISODE} days each)")

    # ── Model 1: Market-Cap Weighted ────────────────────────────
    print(f"\n{'='*65}")
    print("  Model 1: Market-Cap Weighted")
    print(f"{'='*65}")
    mc_dfs = {}
    for split_name in ["train", "val", "test"]:
        print(f"  [{split_name}] ", end="", flush=True)
        df = run_index_model(ipo_data_cache[split_name], split_name, use_optimizer=False)
        mc_dfs[split_name] = df
        print(f"{len(df)} days")

    # ── Model 2: Online Portfolio Optimizer ─────────────────────
    print(f"\n{'='*65}")
    print("  Model 2: Online Portfolio Optimizer")
    print(f"{'='*65}")
    opt_dfs = {}
    for split_name in ["train", "val", "test"]:
        print(f"  [{split_name}] ", end="", flush=True)
        df = run_index_model(ipo_data_cache[split_name], split_name, use_optimizer=True)
        opt_dfs[split_name] = df
        print(f"{len(df)} days")

    # ── Model 3: GRU Policy ─────────────────────────────────────
    gru_dfs = {s: pd.DataFrame(columns=["date", "daily_return"]) for s in ["train", "val", "test"]}
    if not args.no_gru:
        print(f"\n{'='*65}")
        print("  Model 3: GRU Policy (per-split, trains on own period history)")
        print(f"{'='*65}")
        try:
            import torch
            for split_name in ["train", "val", "test"]:
                print(f"  [{split_name}] training GRU ({args.gru_epochs} epochs)...", flush=True)
                df = run_index_model(
                    ipo_data_cache[split_name], split_name,
                    use_gru=True,
                )
                gru_dfs[split_name] = df
                print(f"    => {len(df)} days")
        except ImportError:
            print("  PyTorch not available — GRU skipped.")
    else:
        print("\n  Model 3: GRU skipped (--no_gru).")

    # ── Model 4: REINFORCE MLP ──────────────────────────────────
    print(f"\n{'='*65}")
    print("  Model 4: REINFORCE MLP (trained on train episodes)")
    print(f"{'='*65}")
    train_eps = episodes_cache["train"]
    print(f"  Training on {len(train_eps)} episodes ({args.reinforce_epochs} epochs)...")

    reinforce_val_rets, reinforce_summary = run_reinforce_model(
        train_eps, episodes_cache["val"],
        n_epochs=args.reinforce_epochs, batch_size=REINFORCE_BATCH,
    )
    reinforce_test_rets, _ = run_reinforce_model(
        train_eps, episodes_cache["test"],
        n_epochs=args.reinforce_epochs, batch_size=REINFORCE_BATCH,
    )

    if reinforce_summary:
        out_path = RESULTS_DIR / "model_reinforce_train.json"
        with open(out_path, "w") as f:
            json.dump(reinforce_summary, f, indent=2, default=str)
        print(f"  Training summary saved to {out_path}")

    # ── Model 5: PolicyParams ───────────────────────────────────
    print(f"\n{'='*65}")
    print("  Model 5: PolicyParams rule-based (no training)")
    print(f"{'='*65}")
    policy_val_rets  = run_policy_params_model(episodes_cache["val"])
    policy_test_rets = run_policy_params_model(episodes_cache["test"])
    print(f"  Val: {len(policy_val_rets)} episodes, Test: {len(policy_test_rets)} episodes")

    # ── Compute validation metrics ──────────────────────────────
    val_rows = [
        {
            "name":     "Market-Cap Weighted",
            "key":      "mktcap",
            "metrics":  compute_metrics_aligned(mc_dfs["val"], spy, dji),
            "is_daily": True,
        },
        {
            "name":     "Online Optimizer",
            "key":      "optimizer",
            "metrics":  compute_metrics_aligned(opt_dfs["val"], spy, dji),
            "is_daily": True,
        },
        {
            "name":     "GRU Policy",
            "key":      "gru",
            "metrics":  compute_metrics_aligned(gru_dfs["val"], spy, dji),
            "is_daily": True,
        },
        {
            "name":     "REINFORCE MLP",
            "key":      "reinforce",
            # Episode-based: no direct benchmark alignment; excess = 0
            "metrics":  compute_metrics(reinforce_val_rets, returns_are_daily=False),
            "is_daily": False,
        },
        {
            "name":     "PolicyParams (Rule)",
            "key":      "policy",
            "metrics":  compute_metrics(policy_val_rets, returns_are_daily=False),
            "is_daily": False,
        },
    ]

    print_table(val_rows, "VALIDATION RESULTS  (2022–2023 IPOs)")

    # Note about episode-based models
    print(
        "\n  Note: REINFORCE MLP and PolicyParams report per-episode metrics "
        "(annualized via ×252/21).\n"
        "  Index models (1-3) report daily metrics aligned to SPY/^DJI."
    )

    val_csv = RESULTS_DIR / "comparison_val.csv"
    pd.DataFrame([{"model": r["name"], **r["metrics"]} for r in val_rows]).to_csv(
        val_csv, index=False
    )
    print(f"\n  Saved: {val_csv}")

    # ── Identify best model by validation Sharpe ────────────────
    best_idx    = max(range(len(val_rows)), key=lambda i: val_rows[i]["metrics"]["sharpe"])
    best_row    = val_rows[best_idx]
    best_name   = best_row["name"]
    best_sharpe = best_row["metrics"]["sharpe"]
    best_key    = best_row["key"]
    best_daily  = best_row["is_daily"]

    print(f"\n{'='*65}")
    print(f"  BEST MODEL by Val Sharpe: {best_name}")
    print(f"  Val Sharpe = {best_sharpe:.4f}")
    print(f"{'='*65}")

    # ── Test the winner ─────────────────────────────────────────
    print(f"\n  Running test evaluation for: {best_name}")
    if best_key == "mktcap":
        test_result_df = mc_dfs["test"]
        test_metrics = compute_metrics_aligned(test_result_df, spy, dji)
    elif best_key == "optimizer":
        test_result_df = opt_dfs["test"]
        test_metrics = compute_metrics_aligned(test_result_df, spy, dji)
    elif best_key == "gru":
        test_result_df = gru_dfs["test"]
        test_metrics = compute_metrics_aligned(test_result_df, spy, dji)
    elif best_key == "reinforce":
        test_metrics = compute_metrics(reinforce_test_rets, returns_are_daily=False)
    else:  # policy
        test_metrics = compute_metrics(policy_test_rets, returns_are_daily=False)

    test_rows = [{"name": best_name, "metrics": test_metrics}]
    print_table(test_rows, f"TEST RESULTS  (2024–2025 IPOs) — Winner: {best_name}")

    test_csv = RESULTS_DIR / "comparison_test.csv"
    pd.DataFrame([{"model": best_name, **test_metrics}]).to_csv(test_csv, index=False)
    print(f"\n  Saved: {test_csv}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"  Results directory:  {RESULTS_DIR}/")
    print(f"  Validation table:   {val_csv}")
    print(f"  Test results:       {test_csv}")
    print(f"  Best model:         {best_name}  (Val Sharpe={best_sharpe:.4f})")
    print(f"  Test Sharpe:        {test_metrics['sharpe']:.4f}")
    print(f"  Test Total Return:  {test_metrics['total_return']*100:+.2f}%")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
