#!/usr/bin/env python3
"""
Run IPO Portfolio Optimizer on 2020-01-01 to 2025-12-31.
IPO data: SDC New Deals (all rows where ipodate is not null) + CRSP daily prices (split-adjusted).
Market: Market-cap weighted portfolio of S&P 500 (SPY) and Dow Jones (DIA) from CRSP.
Uses best config from results/ipo_optimizer_best_config.json if present (from tune_hyperparameters_wrds.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

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
)
from src.data_layer import align_returns, add_optional_features, build_rolling_windows, train_val_split
from src.train import run_training
from src.export import predict_weights, portfolio_stats, export_weights_csv, export_summary
from src.policy_layer import ipo_tilt_to_position_scale, policy_rule

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

DEFAULTS = {
    "window_len": 126,
    "val_frac": 0.2,
    "epochs": 50,
    "lr": 1e-3,
    "batch_size": 32,
    "patience": 10,
    "lambda_vol": 0.5,
    "lambda_cvar": 0.5,
    "lambda_vol_excess": 1.0,
    "target_vol_annual": 0.25,
    "hidden_size": 64,
    "lambda_diversify": 0.0,
    "min_weight": 0.1,
}


def load_best_config():
    """Load best config from tuning; fall back to DEFAULTS if not found."""
    path = ROOT / "results" / "ipo_optimizer_best_config.json"
    if not path.exists():
        return DEFAULTS.copy()
    with open(path) as f:
        out = json.load(f)
    best = out.get("best_config")
    if not best:
        return DEFAULTS.copy()
    cfg = DEFAULTS.copy()
    for k in cfg:
        if k in best:
            cfg[k] = best[k]
    return cfg


def build_ipo_index_mcap(prices_df, ipo_dates_df, shares_dict, holding_days=180, min_names=1):
    ipo_lookup = dict(zip(ipo_dates_df["ticker"], ipo_dates_df["ipo_date"]))
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
            for t, mcap in market_caps.items():
                try:
                    r = returns_df.loc[date, t]
                    if pd.notna(r):
                        wr += (mcap / total_mcap) * r
                        vc += 1
                except Exception:
                    pass
            ipo_ret = wr if vc >= min_names else np.nan
        else:
            ipo_ret = np.nan
        index_data.append({"date": date, "ipo_ret": ipo_ret})
    return pd.DataFrame(index_data).set_index("date")


def prepare_data(conn):
    """Load and prepare IPO + market data. Returns dict with df, feature_cols for rolling windows."""
    # IPO data: SDC New Deals (all rows where ipodate is not null) + CRSP daily prices
    ipo_csv = load_ipo_data_from_sdc_wrds(
        conn, start=START_DATE, end=END_DATE, library="sdc", price_source="crsp"
    )
    print(f"IPO data from SDC + CRSP: {len(ipo_csv)} rows, {ipo_csv['tic'].nunique()} tickers")

    ipo_csv["datadate"] = pd.to_datetime(ipo_csv["datadate"])
    ipo_csv = ipo_csv.drop_duplicates(subset=["tic", "datadate"], keep="first")

    prices_ipo = ipo_csv.pivot_table(index="datadate", columns="tic", values="prccd")
    prices_ipo.index = pd.to_datetime(prices_ipo.index).normalize()

    # IPO dates from SDC (not first trading date); filter to tickers with prices
    ipo_dates = load_sdc_ipo_dates_wrds(
        conn, start=START_DATE, end=END_DATE, library="sdc"
    )
    ipo_df = ipo_dates[ipo_dates["ticker"].isin(prices_ipo.columns)].copy()
    ipo_df = ipo_df.sort_values("ipo_date").reset_index(drop=True)

    start_d = prices_ipo.index.min().strftime("%Y-%m-%d")
    end_d = prices_ipo.index.max().strftime("%Y-%m-%d")
    print(f"IPO tickers: {len(ipo_df)}, Date range: {start_d} to {end_d}")

    # Market returns: market-cap weighted S&P 500 (82%) + Dow Jones (18%) from CRSP
    # Use full date range through END_DATE to extend to end of 2025
    market_end = max(end_d, END_DATE) if end_d else END_DATE
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
                and datadate >= '2020-01-01'
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

    # Build IPO index
    ipo_index = build_ipo_index_mcap(prices, ipo_df, shares_outstanding, holding_days=180)
    print(f"IPO index: {ipo_index['ipo_ret'].notna().sum()} days with valid returns")

    # Train
    ipo_ret = ipo_index["ipo_ret"].rename("ipo_return")
    df = align_returns(market_ret, ipo_ret)
    df = add_optional_features(df, include_vix=False)
    feature_cols = list(df.columns)
    return {"df": df, "feature_cols": feature_cols}


def main():
    print("Connecting to WRDS...")
    conn = get_connection()
    print("Connected.")

    data_prep = prepare_data(conn)
    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]

    cfg = load_best_config()
    print(f"Hyperparameters: {cfg}")

    X, R, dates = build_rolling_windows(
        df, window_len=cfg["window_len"], feature_cols=feature_cols
    )
    X_train, R_train, d_train, X_val, R_val, d_val = train_val_split(
        X, R, dates, val_frac=cfg["val_frac"]
    )

    data = {
        "X_train": X_train,
        "R_train": R_train,
        "dates_train": d_train,
        "X_val": X_val,
        "R_val": R_val,
        "dates_val": d_val,
        "feature_cols": feature_cols,
        "df": df,
        "n_assets": 2,
        "window_len": cfg["window_len"],
    }
    print(f"Train windows: {X_train.shape[0]}, Val windows: {X_val.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = run_training(
        data,
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        patience=cfg["patience"],
        lambda_vol=cfg["lambda_vol"],
        lambda_cvar=cfg["lambda_cvar"],
        lambda_diversify=cfg.get("lambda_diversify", 1.0),
        min_weight=cfg.get("min_weight", 0.1),
        lambda_vol_excess=cfg.get("lambda_vol_excess", 1.0),
        target_vol_annual=cfg.get("target_vol_annual", 0.25),
        hidden_size=cfg["hidden_size"],
        model_type="gru",
    )
    print(f"Trained for {len(history)} epochs")

    # Plot train/val loss (log y-scale; use abs for semilogy since loss can be negative)
    epochs_x = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    t_plot = [max(abs(x), 1e-8) for x in train_loss]
    v_plot = [max(abs(x), 1e-8) for x in val_loss]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs_x, t_plot, label="Train loss", marker="o", markersize=3)
    ax.semilogy(epochs_x, v_plot, label="Validation loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("IPO Optimizer: Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "ipo_optimizer_loss.png", dpi=150)
    plt.close()
    print(f"Saved loss plot to {fig_dir / 'ipo_optimizer_loss.png'}")

    weights = predict_weights(model, data["X_val"], device)
    stats = portfolio_stats(weights, data["R_val"])

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    weights_path = out_dir / "ipo_optimizer_weights.csv"
    summary_path = out_dir / "ipo_optimizer_summary.txt"

    export_weights_csv(data["dates_val"], weights, weights_path)
    export_summary(stats, weights, summary_path, R=data["R_val"])
    print(f"Exported weights to {weights_path}")
    print(f"Exported summary to {summary_path}")

    avg_ipo = float(weights[:, 1].mean()) if weights.shape[1] >= 2 else 0.0
    scale = ipo_tilt_to_position_scale(avg_ipo)
    print(policy_rule(avg_ipo))
    print(f"Suggested position scale: {scale:.2f}")
    print(f"Metrics: Sharpe={stats['sharpe_annualized']:.2f}, MaxDD={stats['max_drawdown']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
