#!/usr/bin/env python3
"""
Run IPO Portfolio Optimizer on 2020-01-01 to 2025-12-31.
IPO data: SDC New Deals (all rows where ipodate is not null) + CRSP daily prices (split-adjusted).
Market: Market-cap weighted portfolio of S&P 500 (SPY) and Dow Jones (DIA) from CRSP.
Uses best config from results/ipo_optimizer_best_config.json if present (from tune_hyperparameters_wrds.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.wrds_data import (
    get_connection,
    load_ipo_data_from_sdc_wrds,
    load_market_returns_wrds,
    load_sdc_ipo_dates_wrds,
    load_sp500_dow_market_returns_wrds,
    load_stock_returns_wrds,
    load_ticker_sector_info_wrds,
    load_vix_wrds,
)
from src.data_layer import (
    align_returns,
    add_optional_features,
    build_rolling_windows,
    train_val_test_split,
)
from src.train import (
    mean_excess_vs_ew_selection_objective,
    path_metrics_numpy,
    run_training,
    rolling_tail_excess_objective,
)
from src.export import predict_weights, portfolio_stats, export_weights_csv, export_summary
from src.policy_layer import ipo_tilt_to_position_scale, policy_rule

DEFAULT_START_DATE = os.environ.get("IPO_START_DATE", "2020-01-01")
DEFAULT_END_DATE = os.environ.get("IPO_END_DATE", "2025-12-31")
DEFAULT_VAL_START = os.environ.get("IPO_VAL_START")
DEFAULT_TEST_START = os.environ.get("IPO_TEST_START")
DEFAULT_CACHE_DIR = os.environ.get("IPO_CACHE_DIR", str(ROOT / "results" / "cache_wrds"))

DEFAULTS = {
    "window_len": 126,
    "val_frac": 0.2,
    "test_frac": 0.1,
    "epochs": 200,
    "lr": 3e-4,
    "lr_decay": 0.1,
    "batch_size": 256,
    "patience": 200,
    "lambda_vol": 0.5,
    "lambda_cvar": 0.5,
    "lambda_turnover": 0.0001,
    "lambda_path": 0.0001,
    "lambda_vol_excess": 1.0,
    "lambda_vs_ew": 0.0,
    "lambda_log_return": 0.2,
    "train_segment_len": 63,
    "lambda_segment_log": 0.1,
    "target_vol_annual": 0.25,
    "hidden_size": 64,
    "lambda_diversify": 0.0,
    "min_weight": 0.1,
}


def load_best_config(model_type: str = "gru"):
    """Load best config from tuning; fall back to DEFAULTS if not found.

    Training mechanics (lr, lr_decay, batch_size, epochs, patience) are always
    taken from DEFAULTS so that manual edits there take effect immediately.
    Only model/loss hyperparameters (lambdas, window_len, hidden_size, etc.)
    are carried over from the saved tuning result.
    """
    TRAINING_KEYS = {"lr", "lr_decay", "batch_size", "epochs", "patience"}
    candidates = []
    if model_type and model_type != "gru":
        candidates.append(ROOT / "results" / f"ipo_optimizer_best_config_{model_type}.json")
    candidates.append(ROOT / "results" / "ipo_optimizer_best_config.json")

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return DEFAULTS.copy()
    with open(path) as f:
        out = json.load(f)
    best = out.get("best_config")
    if not best:
        return DEFAULTS.copy()
    cfg = DEFAULTS.copy()
    for k in cfg:
        if k in best and k not in TRAINING_KEYS:
            cfg[k] = best[k]
    return cfg


def build_ipo_index_mcap_legacy(
    prices_df,
    ipo_dates_df,
    shares_dict,
    sector_id_map=None,
    holding_days=180,
    min_names=1,
):
    ipo_lookup = dict(zip(ipo_dates_df["ticker"], ipo_dates_df["ipo_date"]))
    sector_id_map = sector_id_map or {}
    returns_df = prices_df.pct_change()
    trading_days = {
        t: prices_df[t].dropna().index.tolist()
        for t in prices_df.columns
        if t != "SPY" and t in ipo_lookup
    }
    all_dates = prices_df.index.tolist()
    index_data = []
    for date in all_dates:
        market_caps = {}
        for ticker, ipo_date in ipo_lookup.items():
            if ticker not in trading_days or ticker not in shares_dict:
                continue
            ticker_days = trading_days[ticker]
            first_trade_idx = next((i for i, d in enumerate(ticker_days) if d >= ipo_date), None)
            if first_trade_idx is None:
                continue
            if date in ticker_days:
                current_idx = ticker_days.index(date)
                if 0 <= current_idx - first_trade_idx < holding_days:
                    try:
                        cp = prices_df.loc[date, ticker]
                        if pd.notna(cp) and cp > 0:
                            market_caps[ticker] = cp * shares_dict[ticker]
                    except Exception:
                        pass
        total_mcap = sum(market_caps.values())
        if len(market_caps) >= min_names and total_mcap > 0:
            wr, vc = 0.0, 0
            sector_weighted_sum, sector_weighted_den = 0.0, 0.0
            for t, mcap in market_caps.items():
                try:
                    r = returns_df.loc[date, t]
                    if pd.notna(r):
                        wr += (mcap / total_mcap) * r
                        vc += 1
                except Exception:
                    pass
                sid = sector_id_map.get(t)
                if sid is not None and pd.notna(sid):
                    sector_weighted_sum += float(sid) * float(mcap)
                    sector_weighted_den += float(mcap)
            ipo_ret = wr if vc >= min_names else np.nan
            sector_id_wavg = (
                sector_weighted_sum / sector_weighted_den
                if sector_weighted_den > 0
                else np.nan
            )
            sector_count = len(
                {sector_id_map.get(t) for t in market_caps.keys() if pd.notna(sector_id_map.get(t))}
            )
        else:
            ipo_ret = np.nan
            sector_id_wavg = np.nan
            sector_count = 0
        index_data.append(
            {
                "date": date,
                "ipo_ret": ipo_ret,
                "ipo_sector_id_wavg": sector_id_wavg,
                "ipo_sector_count": sector_count,
            }
        )
    return pd.DataFrame(index_data).set_index("date")


def build_ipo_index_mcap_fast(
    prices_df: pd.DataFrame,
    ipo_dates_df: pd.DataFrame,
    shares_dict: dict,
    sector_id_map: dict | None = None,
    holding_days: int = 180,
    min_names: int = 1,
    progress_every: int = 1000,
) -> pd.DataFrame:
    """
    Vectorized-ish IPO index construction.

    Keeps the same economic definition as the legacy builder:
    - active window = first `holding_days` trading observations on/after IPO date
    - mcap weight per day = price * shares
    - return uses only names with non-null returns that day
    """
    sector_id_map = sector_id_map or {}
    dates = prices_df.index.values.astype("datetime64[ns]")
    T = len(dates)
    total_mcap = np.zeros(T, dtype=np.float64)
    ret_num = np.zeros(T, dtype=np.float64)
    ret_valid_count = np.zeros(T, dtype=np.int32)
    active_name_count = np.zeros(T, dtype=np.int32)
    sector_num = np.zeros(T, dtype=np.float64)
    sector_den = np.zeros(T, dtype=np.float64)

    ipo_lookup = dict(zip(ipo_dates_df["ticker"], ipo_dates_df["ipo_date"]))
    tickers = [t for t in prices_df.columns if t in ipo_lookup and t in shares_dict]

    sector_values = sorted(
        {
            float(v)
            for v in sector_id_map.values()
            if v is not None and pd.notna(v)
        }
    )
    sector_to_idx = {sid: i for i, sid in enumerate(sector_values)}
    sector_presence = np.zeros((T, len(sector_values)), dtype=bool) if sector_values else None

    for i, ticker in enumerate(tickers, start=1):
        try:
            shares = float(shares_dict[ticker])
            if not np.isfinite(shares) or shares <= 0:
                continue

            s = prices_df[ticker]
            p = s.to_numpy(dtype=np.float64)
            valid_non_nan = np.isfinite(p)
            valid_idx = np.flatnonzero(valid_non_nan)
            if valid_idx.size == 0:
                continue

            ipo_d = np.datetime64(pd.Timestamp(ipo_lookup[ticker]).to_datetime64())
            trade_dates = dates[valid_idx]
            start_ord = np.searchsorted(trade_dates, ipo_d, side="left")
            if start_ord >= valid_idx.size:
                continue
            end_ord = min(start_ord + holding_days, valid_idx.size)
            active_idx = valid_idx[start_ord:end_ord]
            if active_idx.size == 0:
                continue

            active_price = p[active_idx]
            pos_mask = active_price > 0
            if not np.any(pos_mask):
                continue
            active_idx = active_idx[pos_mask]
            mcap = active_price[pos_mask] * shares

            total_mcap[active_idx] += mcap
            active_name_count[active_idx] += 1

            r = s.pct_change().to_numpy(dtype=np.float64)
            valid_r = np.isfinite(r[active_idx])
            if np.any(valid_r):
                idx_r = active_idx[valid_r]
                ret_num[idx_r] += mcap[valid_r] * r[idx_r]
                ret_valid_count[idx_r] += 1

            sid = sector_id_map.get(ticker)
            if sid is not None and pd.notna(sid):
                sid = float(sid)
                sector_num[active_idx] += mcap * sid
                sector_den[active_idx] += mcap
                if sector_presence is not None and sid in sector_to_idx:
                    sector_presence[active_idx, sector_to_idx[sid]] = True
        except Exception:
            # Keep robust behavior of legacy path and continue ticker-by-ticker.
            continue

        if progress_every > 0 and (i % progress_every == 0):
            print(f"IPO index build progress: {i}/{len(tickers)} tickers processed", flush=True)

    active_ok = (active_name_count >= min_names) & (total_mcap > 0)
    ret_ok = active_ok & (ret_valid_count >= min_names)
    ipo_ret = np.full(T, np.nan, dtype=np.float64)
    ipo_ret[ret_ok] = ret_num[ret_ok] / total_mcap[ret_ok]

    sector_id_wavg = np.full(T, np.nan, dtype=np.float64)
    sector_valid = active_ok & (sector_den > 0)
    sector_id_wavg[sector_valid] = sector_num[sector_valid] / sector_den[sector_valid]
    if sector_presence is not None and sector_presence.shape[1] > 0:
        sector_count = sector_presence.sum(axis=1).astype(np.int32)
    else:
        sector_count = np.zeros(T, dtype=np.int32)
    sector_count[~active_ok] = 0

    return pd.DataFrame(
        {
            "ipo_ret": ipo_ret,
            "ipo_sector_id_wavg": sector_id_wavg,
            "ipo_sector_count": sector_count,
        },
        index=prices_df.index,
    )


def build_ipo_index_mcap(
    prices_df,
    ipo_dates_df,
    shares_dict,
    sector_id_map=None,
    holding_days=180,
    min_names=1,
    method: str = "fast",
):
    if method == "legacy":
        return build_ipo_index_mcap_legacy(
            prices_df,
            ipo_dates_df,
            shares_dict,
            sector_id_map=sector_id_map,
            holding_days=holding_days,
            min_names=min_names,
        )
    progress_every = int(os.environ.get("IPO_INDEX_PROGRESS_EVERY", "1000"))
    return build_ipo_index_mcap_fast(
        prices_df,
        ipo_dates_df,
        shares_dict,
        sector_id_map=sector_id_map,
        holding_days=holding_days,
        min_names=min_names,
        progress_every=progress_every,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run IPO optimizer with WRDS data.")
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Start date YYYY-MM-DD (default: env IPO_START_DATE or {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"End date YYYY-MM-DD (default: env IPO_END_DATE or {DEFAULT_END_DATE})",
    )
    parser.add_argument(
        "--max-history",
        action="store_true",
        help="Auto-start at the earliest available IPO date in WRDS SDC tables.",
    )
    parser.add_argument(
        "--model",
        default="gru",
        choices=["gru", "lstm", "transformer"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--val-start",
        default=DEFAULT_VAL_START,
        help="Validation start date YYYY-MM-DD. If omitted, fraction-based split is used.",
    )
    parser.add_argument(
        "--test-start",
        default=DEFAULT_TEST_START,
        help="Test start date YYYY-MM-DD. If set with --val-start, uses calendar train/val/test split.",
    )
    parser.add_argument(
        "--selection-metric",
        default="rolling_tail_excess",
        choices=[
            "val_loss",
            "rolling_tail_excess",
            "mean_excess_vs_ew",
            "val_compound_return",
            "val_sharpe",
            "val_sortino",
            "val_max_drawdown",
            "val_retail_composite",
        ],
        help=(
            "Checkpoint / early-stop criterion on chronological validation weights. "
            "val_* metrics use full val path: compound return, Sharpe, Sortino, max drawdown, "
            "or val_retail_composite (Sharpe + 0.5 Sortino - drawdown_penalty*|maxDD|). "
            "Use --selection-drawdown-penalty > 0 with val_retail_composite to penalize deep drawdowns."
        ),
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=21,
        help="Window length (days) for rolling excess metric vs 50/50.",
    )
    parser.add_argument(
        "--rolling-tail-quantile",
        type=float,
        default=0.10,
        help="Lower-tail quantile for rolling excess metric (e.g., 0.10 = 10th percentile).",
    )
    parser.add_argument(
        "--selection-drawdown-penalty",
        type=float,
        default=0.0,
        help=(
            "Rolling-tail objective: drawdown term. "
            "val_retail_composite: multiplies |max_drawdown| (use e.g. 0.5–2.0)."
        ),
    )
    parser.add_argument(
        "--ipo-index-method",
        default="fast",
        choices=["fast", "legacy"],
        help="Implementation used to build IPO mcap index.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Directory for prepared WRDS cache (df + feature columns).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use prebuilt cache from --cache-dir instead of pulling/building WRDS data.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build cache and exit (no model training).",
    )
    parser.add_argument(
        "--lambda-vs-ew",
        type=float,
        default=None,
        metavar="LAMBDA",
        help=(
            "If set, overrides config: weight on penalizing underperformance vs equal-weight "
            "daily return in each batch (see loss_excess_vs_equal_weight). "
            "Try 0.2–1.0 when the model is too conservative vs 50/50. Omit to use config/DEFAULTS."
        ),
    )
    parser.add_argument(
        "--risk-penalty-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplies λ_vol, λ_cvar, and λ_vol_excess after loading config. "
            "Values below 1.0 (e.g. 0.5) reduce risk/regularization pressure relative to mean return."
        ),
    )
    parser.add_argument(
        "--lambda-log-return",
        type=float,
        default=None,
        metavar="LAMBDA",
        help="If set, weight on -mean(log(1+r)) in batch loss (long-run growth proxy). Else config/DEFAULTS.",
    )
    parser.add_argument(
        "--train-segment-len",
        type=int,
        default=None,
        metavar="L",
        help=(
            "If >0 with --lambda-segment-log, each train batch adds random contiguous "
            "length-L log-growth loss (chronological subsample). Else config/DEFAULTS."
        ),
    )
    parser.add_argument(
        "--lambda-segment-log",
        type=float,
        default=None,
        metavar="LAMBDA",
        help="Weight on segment log-growth auxiliary loss (0 disables segment term).",
    )
    return parser.parse_args()


def save_prep_cache(cache_dir: Path, data_prep: dict, metadata: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_path = cache_dir / "prepared_df.pkl"
    meta_path = cache_dir / "prepared_meta.json"
    data_prep["df"].to_pickle(df_path)
    payload = {
        "feature_cols": data_prep["feature_cols"],
        **metadata,
    }
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved prepared cache: {df_path}")
    print(f"Saved cache metadata: {meta_path}")


def load_prep_cache(cache_dir: Path) -> dict:
    df_path = cache_dir / "prepared_df.pkl"
    meta_path = cache_dir / "prepared_meta.json"
    if not df_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Cache not found in {cache_dir}. Expected files: prepared_df.pkl and prepared_meta.json"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    df = pd.read_pickle(df_path)
    feature_cols = meta.get("feature_cols", list(df.columns))
    print(f"Loaded prepared cache: {df_path}")
    return {"df": df, "feature_cols": feature_cols, "meta": meta}


def find_earliest_ipo_date(conn, end_date: str | None = None, library: str = "sdc", date_column: str = "ipodate"):
    tables_to_try = ["wrds_ni_details", "globalnewiss", "new_issues", "sdc_new_issues", "newiss"]
    end_clause = f"AND {date_column} <= '{end_date}'" if end_date else ""
    best_date = None
    for tbl in tables_to_try:
        try:
            sql = f"""
                SELECT MIN({date_column}) AS min_date
                FROM {library}.{tbl}
                WHERE {date_column} IS NOT NULL
                  AND ipo = 'Yes'
                  {end_clause}
            """
            out = conn.raw_sql(sql)
            if out.empty or "min_date" not in out.columns:
                continue
            value = out["min_date"].iloc[0]
            if pd.isna(value):
                continue
            cand = pd.to_datetime(value).strftime("%Y-%m-%d")
            if best_date is None or cand < best_date:
                best_date = cand
        except Exception:
            continue
    return best_date


def prepare_data(
    conn,
    start_date: str,
    end_date: str,
    max_history: bool = False,
    ipo_index_method: str = "fast",
):
    """Load and prepare IPO + market data. Returns dict with df, feature_cols for rolling windows."""
    if max_history:
        earliest = find_earliest_ipo_date(conn, end_date=end_date, library="sdc", date_column="ipodate")
        if earliest:
            print(f"Max-history enabled: using earliest available IPO date {earliest}")
            start_date = earliest
        else:
            print(f"Max-history enabled but earliest date lookup failed; using start-date {start_date}")

    # IPO data: SDC New Deals (all rows where ipodate is not null) + CRSP daily prices
    ipo_csv = load_ipo_data_from_sdc_wrds(
        conn, start=start_date, end=end_date, library="sdc", price_source="crsp"
    )
    print(f"IPO data from SDC + CRSP: {len(ipo_csv)} rows, {ipo_csv['tic'].nunique()} tickers")

    ipo_csv["datadate"] = pd.to_datetime(ipo_csv["datadate"])
    ipo_csv = ipo_csv.drop_duplicates(subset=["tic", "datadate"], keep="first")

    prices_ipo = ipo_csv.pivot_table(index="datadate", columns="tic", values="prccd")
    prices_ipo.index = pd.to_datetime(prices_ipo.index).normalize()

    # IPO dates from SDC (not first trading date); filter to tickers with prices
    ipo_dates = load_sdc_ipo_dates_wrds(
        conn, start=start_date, end=end_date, library="sdc"
    )
    ipo_df = ipo_dates[ipo_dates["ticker"].isin(prices_ipo.columns)].copy()
    ipo_df = ipo_df.sort_values("ipo_date").reset_index(drop=True)

    start_d = prices_ipo.index.min().strftime("%Y-%m-%d")
    end_d = prices_ipo.index.max().strftime("%Y-%m-%d")
    print(f"IPO tickers: {len(ipo_df)}, Date range: {start_d} to {end_d}")

    # Market returns: market-cap weighted S&P 500 (82%) + Dow Jones (18%) from CRSP
    # Use full requested end date to avoid truncating after IPO data ends.
    market_end = max(end_d, end_date) if end_d else end_date
    market_ret = load_sp500_dow_market_returns_wrds(
        conn, start=start_d, end=market_end, w_sp500=0.82, w_dow=0.18
    )
    market_ret = market_ret.reindex(prices_ipo.index).dropna()
    if len(market_ret) < 50:
        market_ret = load_market_returns_wrds(conn, start=start_d, end=end_d)
        market_ret = market_ret.reindex(prices_ipo.index).dropna()
    if len(market_ret) < 50:
        raise RuntimeError(
            "Insufficient market return data from CRSP (SPY/DIA or dsi). "
            "Check date range and WRDS subscription."
        )

    # Shares: from CRSP shrout (when price_source=crsp) or comp.funda (gvkey)
    ipo_tickers = ipo_df["ticker"].tolist()
    shares_outstanding = {}
    if "shrout" in ipo_csv.columns:
        last_shrout = ipo_csv.dropna(subset=["shrout"]).sort_values("datadate").groupby("tic")["shrout"].last()
        for tic, s in last_shrout.items():
            if s and s > 0:
                shares_outstanding[tic] = float(s) * 1000  # CRSP shrout in thousands
    elif "gvkey" in ipo_csv.columns:
        gvkeys = ipo_csv[["tic", "gvkey"]].drop_duplicates()
        gvkey_list = "','".join(gvkeys["gvkey"].astype(str).str.zfill(6).unique().tolist())
        shares_df = conn.raw_sql(
            f"""
            select gvkey, datadate, csho
            from comp.funda
            where gvkey in ('{gvkey_list}')
                and datadate >= '{start_d}'
                and csho > 0
                and indfmt = 'INDL' and datafmt = 'STD'
            """,
            date_cols=["datadate"],
        )
        if len(shares_df) > 0:
            last_csho = shares_df.sort_values("datadate").groupby("gvkey")["csho"].last()
            gvkey_to_tic = dict(zip(gvkeys["gvkey"].astype(str).str.zfill(6), gvkeys["tic"]))
            for gvkey, csho in last_csho.items():
                t = gvkey_to_tic.get(str(gvkey).zfill(6))
                if t:
                    shares_outstanding[t] = float(csho) * 1000

    for t in ipo_tickers:
        if t in prices_ipo.columns and t not in shares_outstanding:
            p = prices_ipo[t].dropna()
            if len(p) > 0 and p.iloc[-1] > 0:
                shares_outstanding[t] = 1e6 / p.iloc[-1]

    prices = prices_ipo.copy().ffill().bfill()
    print(f"Market return days: {len(market_ret)}, Tickers with shares: {len(shares_outstanding)}")

    # Pull sector metadata from WRDS (no yfinance) and build a ticker->sector_id map.
    sector_info = load_ticker_sector_info_wrds(conn, ipo_tickers)
    sector_id_map = (
        dict(zip(sector_info["ticker"], sector_info["sector_id"]))
        if len(sector_info) > 0
        else {}
    )
    print(
        f"WRDS sector mapping: {len(sector_id_map)} / {len(ipo_tickers)} tickers have sector_id"
    )

    # Persist mapping for auditability and reuse.
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    if len(sector_info) > 0:
        sector_info.to_csv(out_dir / "ticker_sector_cache.csv", index=False)

    # Build IPO index
    ipo_index = build_ipo_index_mcap(
        prices,
        ipo_df,
        shares_outstanding,
        sector_id_map=sector_id_map,
        holding_days=180,
        method=ipo_index_method,
    )
    print(f"IPO index: {ipo_index['ipo_ret'].notna().sum()} days with valid returns")

    # Train
    ipo_ret = ipo_index["ipo_ret"].rename("ipo_return")
    df = align_returns(market_ret, ipo_ret)
    # Add WRDS-based sector features so GRU/LSTM/Transformer share the same sector signal.
    df["ipo_sector_id_wavg"] = ipo_index["ipo_sector_id_wavg"].reindex(df.index).ffill().bfill()
    df["ipo_sector_count"] = (
        ipo_index["ipo_sector_count"].reindex(df.index).fillna(0).astype(float)
    )
    vix_series = load_vix_wrds(conn, start=start_d, end=market_end)
    print(f"VIX data from CBOE: {len(vix_series)} days")
    df = add_optional_features(df, vix_series=vix_series)
    feature_cols = list(df.columns)
    return {"df": df, "feature_cols": feature_cols}


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    if args.use_cache:
        cached = load_prep_cache(cache_dir)
        data_prep = {"df": cached["df"], "feature_cols": cached["feature_cols"]}
    else:
        print("Connecting to WRDS...")
        conn = get_connection()
        print("Connected.")
        data_prep = prepare_data(
            conn,
            start_date=args.start_date,
            end_date=args.end_date,
            max_history=args.max_history,
            ipo_index_method=args.ipo_index_method,
        )
        cache_meta = {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "max_history": bool(args.max_history),
            "prepared_at_utc": pd.Timestamp.utcnow().isoformat(),
        }
        save_prep_cache(cache_dir, data_prep, cache_meta)
        try:
            conn.close()
        except Exception:
            pass

    if args.prepare_only:
        print("Prepare-only mode complete. Exiting without training.")
        return 0

    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]

    cfg = load_best_config(model_type=args.model)
    if args.lambda_vs_ew is not None:
        cfg["lambda_vs_ew"] = float(args.lambda_vs_ew)
    if args.risk_penalty_scale != 1.0:
        s = float(args.risk_penalty_scale)
        cfg["lambda_vol"] = float(cfg["lambda_vol"]) * s
        cfg["lambda_cvar"] = float(cfg["lambda_cvar"]) * s
        cfg["lambda_vol_excess"] = float(cfg.get("lambda_vol_excess", 1.0)) * s
    if args.lambda_log_return is not None:
        cfg["lambda_log_return"] = float(args.lambda_log_return)
    if args.train_segment_len is not None:
        cfg["train_segment_len"] = int(args.train_segment_len)
    if args.lambda_segment_log is not None:
        cfg["lambda_segment_log"] = float(args.lambda_segment_log)
    print(f"Hyperparameters ({args.model}): {cfg}")

    X, R, dates = build_rolling_windows(
        df, window_len=cfg["window_len"], feature_cols=feature_cols
    )
    (
        X_train,
        R_train,
        d_train,
        X_val,
        R_val,
        d_val,
        X_test,
        R_test,
        d_test,
    ) = train_val_test_split(
        X, R, dates,
        val_start=args.val_start,
        test_start=args.test_start,
        val_frac=cfg["val_frac"],
        test_frac=cfg.get("test_frac", 0.10),
    )

    data = {
        "X_train": X_train,
        "R_train": R_train,
        "dates_train": d_train,
        "X_val": X_val,
        "R_val": R_val,
        "dates_val": d_val,
        "X_test": X_test,
        "R_test": R_test,
        "dates_test": d_test,
        "feature_cols": feature_cols,
        "df": df,
        "n_assets": 2,
        "window_len": cfg["window_len"],
    }
    print(f"Train windows: {X_train.shape[0]}, Val windows: {X_val.shape[0]}, Test windows: {X_test.shape[0]}")
    if len(d_train) > 0:
        print(f"Train period: {pd.Timestamp(d_train[0]).date()} -> {pd.Timestamp(d_train[-1]).date()}")
    if len(d_val) > 0:
        print(f"Val period:   {pd.Timestamp(d_val[0]).date()} -> {pd.Timestamp(d_val[-1]).date()}")
    if len(d_test) > 0:
        print(f"Test period:  {pd.Timestamp(d_test[0]).date()} -> {pd.Timestamp(d_test[-1]).date()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = run_training(
        data,
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        lr_decay=cfg.get("lr_decay", 0.1),
        batch_size=cfg["batch_size"],
        patience=cfg["patience"],
        lambda_vol=cfg["lambda_vol"],
        lambda_cvar=cfg["lambda_cvar"],
        lambda_turnover=cfg.get("lambda_turnover", 0.01),
        lambda_path=cfg.get("lambda_path", 0.01),
        lambda_diversify=cfg.get("lambda_diversify", 1.0),
        min_weight=cfg.get("min_weight", 0.1),
        lambda_vol_excess=cfg.get("lambda_vol_excess", 1.0),
        lambda_vs_ew=cfg.get("lambda_vs_ew", 0.0),
        lambda_log_return=cfg.get("lambda_log_return", 0.0),
        train_segment_len=int(cfg.get("train_segment_len", 0)),
        lambda_segment_log=float(cfg.get("lambda_segment_log", 0.0)),
        target_vol_annual=cfg.get("target_vol_annual", 0.25),
        hidden_size=cfg["hidden_size"],
        model_type=args.model,
        selection_metric=args.selection_metric,
        rolling_window=args.rolling_window,
        rolling_tail_quantile=args.rolling_tail_quantile,
        selection_drawdown_penalty=args.selection_drawdown_penalty,
    )
    print(f"Trained for {len(history)} epochs")

    # Save loss history for auditing
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    result_suffix = "" if args.model == "gru" else f"_{args.model}"
    pd.DataFrame(history).to_csv(ROOT / "results" / f"training_history{result_suffix}.csv", index=False)

    # Plot train/val loss — semilog with smoothed train curve and LR-drop marker
    epochs_x  = [h["epoch"] for h in history]
    train_loss = np.array([h["train_loss"] for h in history])
    val_loss   = np.array([h["val_loss"]   for h in history])

    # Smooth noisy train loss with a 10-epoch rolling mean for readability
    smooth_window = min(10, len(train_loss))
    train_smooth = np.convolve(train_loss, np.ones(smooth_window) / smooth_window, mode="same")

    # Clip to positive for log scale (loss can be negative when return > risk penalties)
    t_raw    = np.clip(np.abs(train_loss),  1e-8, None)
    t_smooth = np.clip(np.abs(train_smooth), 1e-8, None)
    v_plot   = np.clip(np.abs(val_loss),    1e-8, None)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(epochs_x, t_raw,    alpha=0.25, color="#1f77b4", linewidth=0.8)
    ax.semilogy(epochs_x, t_smooth, color="#1f77b4", linewidth=2,
                label="Train loss (10-ep smoothed)")
    ax.semilogy(epochs_x, v_plot,   color="#ff7f0e", linewidth=2,
                marker="s", markersize=2, label="Validation loss")
    # Mark LR drop after epoch 1
    ax.axvline(x=1, color="red", linewidth=1.2, linestyle="--", alpha=0.7,
               label=f"LR drop ×{cfg.get('lr_decay', 0.1):.1f} (epoch 1)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|Loss|  (log scale)")
    model_label = args.model.upper()
    ax.set_title(
        f"{model_label} Training — {len(history)} epochs  |  "
        f"lr={cfg['lr']:.0e}→{cfg['lr']*cfg.get('lr_decay',0.1):.0e}  "
        f"batch={cfg['batch_size']}  "
        f"λ_path={cfg.get('lambda_path',0):.0e}  λ_turn={cfg.get('lambda_turnover',0):.0e}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"ipo_optimizer_loss{result_suffix}.png", dpi=150)
    fig.savefig(fig_dir / f"ipo_optimizer_loss_semilog{result_suffix}.png", dpi=150)
    plt.close()
    print(f"Saved loss plot to {fig_dir / f'ipo_optimizer_loss_semilog{result_suffix}.png'}")

    # Diagnostics plot to debug flat-loss behavior.
    hist_df = pd.DataFrame(history)
    if {"train_diag_grad_l2", "val_diag_ipo_weight_std", "val_diag_shift_from_5050"}.issubset(hist_df.columns):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(hist_df["epoch"], hist_df["train_diag_grad_l2"], color="#1f77b4")
        axes[0].set_ylabel("Grad L2")
        axes[0].set_title("Training Diagnostics")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(hist_df["epoch"], hist_df["val_diag_ipo_weight_std"], color="#ff7f0e")
        axes[1].set_ylabel("Val IPO w std")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(hist_df["epoch"], hist_df["val_diag_shift_from_5050"], color="#2ca02c")
        axes[2].set_ylabel("Val |IPO-0.5|")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        diag_path = fig_dir / f"ipo_optimizer_diagnostics{result_suffix}.png"
        fig.savefig(diag_path, dpi=150)
        plt.close(fig)
        print(f"Saved diagnostics plot to {diag_path}")

    weights_val = predict_weights(model, data["X_val"], device)
    stats_val = portfolio_stats(weights_val, data["R_val"])
    weights_test = predict_weights(model, data["X_test"], device)
    stats_test = portfolio_stats(weights_test, data["R_test"])

    val_sel_obj, val_sel_diag = rolling_tail_excess_objective(
        weights_val,
        data["R_val"],
        window=args.rolling_window,
        tail_quantile=args.rolling_tail_quantile,
        drawdown_penalty=args.selection_drawdown_penalty,
    )
    test_sel_obj, test_sel_diag = rolling_tail_excess_objective(
        weights_test,
        data["R_test"],
        window=args.rolling_window,
        tail_quantile=args.rolling_tail_quantile,
        drawdown_penalty=args.selection_drawdown_penalty,
    )
    _, val_mean_ex_diag = mean_excess_vs_ew_selection_objective(weights_val, data["R_val"])
    _, test_mean_ex_diag = mean_excess_vs_ew_selection_objective(weights_test, data["R_test"])
    port_val = (weights_val * data["R_val"]).sum(axis=1)
    port_test = (weights_test * data["R_test"]).sum(axis=1)
    path_val = path_metrics_numpy(port_val)
    path_test = path_metrics_numpy(port_test)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    weights_val_path = out_dir / f"ipo_optimizer_weights_val{result_suffix}.csv"
    weights_test_path = out_dir / f"ipo_optimizer_weights_test{result_suffix}.csv"
    summary_val_path = out_dir / f"ipo_optimizer_summary_val{result_suffix}.txt"
    summary_test_path = out_dir / f"ipo_optimizer_summary_test{result_suffix}.txt"
    comparison_path = out_dir / f"ipo_optimizer_selection_metrics{result_suffix}.json"
    returns_val_path = out_dir / f"ipo_optimizer_returns_val{result_suffix}.csv"
    returns_test_path = out_dir / f"ipo_optimizer_returns_test{result_suffix}.csv"

    export_weights_csv(data["dates_val"], weights_val, weights_val_path)
    export_weights_csv(data["dates_test"], weights_test, weights_test_path)
    export_summary(stats_val, weights_val, summary_val_path, R=data["R_val"])
    export_summary(stats_test, weights_test, summary_test_path, R=data["R_test"])
    pd.DataFrame(
        {
            "date": pd.to_datetime(data["dates_val"]),
            "market_return": data["R_val"][:, 0],
            "ipo_return": data["R_val"][:, 1],
            "equal_weight_return": data["R_val"].mean(axis=1),
        }
    ).to_csv(returns_val_path, index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(data["dates_test"]),
            "market_return": data["R_test"][:, 0],
            "ipo_return": data["R_test"][:, 1],
            "equal_weight_return": data["R_test"].mean(axis=1),
        }
    ).to_csv(returns_test_path, index=False)

    selection_payload = {
        "model": args.model,
        "selection_metric": args.selection_metric,
        "lambda_vs_ew": cfg.get("lambda_vs_ew", 0.0),
        "lambda_log_return": cfg.get("lambda_log_return", 0.0),
        "train_segment_len": int(cfg.get("train_segment_len", 0)),
        "lambda_segment_log": float(cfg.get("lambda_segment_log", 0.0)),
        "risk_penalty_scale_applied": float(args.risk_penalty_scale),
        "rolling_window": args.rolling_window,
        "rolling_tail_quantile": args.rolling_tail_quantile,
        "selection_drawdown_penalty": args.selection_drawdown_penalty,
        "validation": {"objective": val_sel_obj, **val_sel_diag},
        "test": {"objective": test_sel_obj, **test_sel_diag},
        "final_mean_excess_vs_ew_validation": val_mean_ex_diag["mean_excess_vs_ew"],
        "final_mean_excess_vs_ew_test": test_mean_ex_diag["mean_excess_vs_ew"],
        "final_path_metrics_validation": path_val,
        "final_path_metrics_test": path_test,
    }
    with open(comparison_path, "w") as f:
        json.dump(selection_payload, f, indent=2)

    print(f"Exported val weights to {weights_val_path}")
    print(f"Exported test weights to {weights_test_path}")
    print(f"Exported val summary to {summary_val_path}")
    print(f"Exported test summary to {summary_test_path}")
    print(f"Exported val returns to {returns_val_path}")
    print(f"Exported test returns to {returns_test_path}")
    print(f"Exported selection metrics to {comparison_path}")

    avg_ipo = float(weights_test[:, 1].mean()) if weights_test.shape[1] >= 2 else 0.0
    scale = ipo_tilt_to_position_scale(avg_ipo)
    print(policy_rule(avg_ipo))
    print(f"Suggested position scale: {scale:.2f}")
    print(
        "Validation metrics: "
        f"Sharpe={stats_val['sharpe_annualized']:.2f}, "
        f"MaxDD={stats_val['max_drawdown']:.2%}, "
        f"TailQExcess={val_sel_diag['tail_q_excess']:.4f}"
    )
    print(
        "Test metrics: "
        f"Sharpe={stats_test['sharpe_annualized']:.2f}, "
        f"MaxDD={stats_test['max_drawdown']:.2%}, "
        f"TailQExcess={test_sel_diag['tail_q_excess']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
