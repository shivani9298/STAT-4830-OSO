#!/usr/bin/env python3
"""
Run IPO Portfolio Optimizer on 2020-01-01 to 2025-12-31.
IPO data: SDC New Deals (all rows where ipodate is not null) + CRSP daily prices (split-adjusted).
Market: Market-cap weighted portfolio of S&P 500 (SPY) and Dow Jones (DIA) from CRSP.
Uses best config from results/ipo_optimizer_best_config.json if present (from tune_hyperparameters_wrds.py).
"""
from __future__ import annotations

import json
import os
import sys
import copy
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
    load_vix_wrds,
)
from src.data_layer import (
    align_returns,
    add_optional_features,
    build_online_schedule,
    build_rolling_windows,
    slice_windows_by_index,
    train_val_split,
)
from src.train import run_training, run_training_online_step
from src.export import (
    evaluate_online_path,
    export_online_path_csv,
    export_online_summary,
    export_summary,
    export_weights_csv,
    portfolio_stats,
    predict_weights,
)
from src.policy_layer import ipo_tilt_to_position_scale, policy_rule

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

DEFAULTS = {
    "mode": "online",
    "window_len": 126,
    "val_frac": 0.2,
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
    "target_vol_annual": 0.25,
    "hidden_size": 64,
    "lambda_diversify": 0.0,
    "min_weight": 0.1,
    "lambda_vs_ew": 0.0,
    "model_type": "gru",
    "warmup_windows": 252,
    "update_freq": "W",
    "epochs_step": 2,
    "cost_bps": 5.0,
    "decision_lag": 0,
    "online_train_lookback": 0,
    "online_lambda_turnover": 5e-05,
    "online_lambda_path": 5e-05,
    "online_stop_on_deterioration": True,
    "online_deterioration_tol": 0.0,
    "update_benefit_horizon": 5,
    "update_gate_mode": "cadence",
    "gate_min_val_improvement": 0.0,
    "gate_min_relative_improvement": 0.0,
    "gate_min_history_windows": 252,
}


def load_best_config():
    """Load best config from tuning; fall back to DEFAULTS if not found.

    Training mechanics (lr, lr_decay, batch_size, epochs, patience) are always
    taken from DEFAULTS so that manual edits there take effect immediately.
    Only model/loss hyperparameters (lambdas, window_len, hidden_size, etc.)
    are carried over from the saved tuning result.
    """
    TRAINING_KEYS = {"lr", "lr_decay", "batch_size", "epochs", "patience"}
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
        if k in best and k not in TRAINING_KEYS:
            cfg[k] = best[k]
    return cfg


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_env_overrides(cfg: dict) -> dict:
    env_map = {
        "IPO_MODE": ("mode", str),
        "IPO_ONLINE_UPDATE_FREQ": ("update_freq", str),
        "IPO_ONLINE_LOOKBACK": ("online_train_lookback", int),
        "IPO_ONLINE_EPOCHS_STEP": ("epochs_step", int),
        "IPO_ONLINE_WARMUP_WINDOWS": ("warmup_windows", int),
        "IPO_ONLINE_COST_BPS": ("cost_bps", float),
        "IPO_ONLINE_LAMBDA_TURNOVER": ("online_lambda_turnover", float),
        "IPO_ONLINE_LAMBDA_PATH": ("online_lambda_path", float),
        "IPO_ONLINE_STOP_ON_DETERIORATION": ("online_stop_on_deterioration", _parse_bool),
        "IPO_ONLINE_DETERIORATION_TOL": ("online_deterioration_tol", float),
        "IPO_UPDATE_BENEFIT_HORIZON": ("update_benefit_horizon", int),
        "IPO_UPDATE_GATE_MODE": ("update_gate_mode", str),
        "IPO_GATE_MIN_VAL_IMPROVEMENT": ("gate_min_val_improvement", float),
        "IPO_GATE_MIN_RELATIVE_IMPROVEMENT": ("gate_min_relative_improvement", float),
        "IPO_GATE_MIN_HISTORY_WINDOWS": ("gate_min_history_windows", int),
    }
    out = dict(cfg)
    for env_k, (cfg_k, caster) in env_map.items():
        raw = os.getenv(env_k)
        if raw is None or raw == "":
            continue
        out[cfg_k] = caster(raw)
    return out


def compute_update_benefit(
    net_returns: np.ndarray,
    updated_flags: np.ndarray,
    horizon: int,
) -> dict:
    r = np.asarray(net_returns, dtype=float).reshape(-1)
    u = np.asarray(updated_flags).astype(bool).reshape(-1)
    n = len(r)
    h = max(1, int(horizon))
    post_update = []
    post_no_update = []
    for i in range(n):
        seg = r[i + 1 : i + 1 + h]
        if len(seg) == 0:
            continue
        val = float(np.prod(1.0 + seg) - 1.0)
        if u[i]:
            post_update.append(val)
        else:
            post_no_update.append(val)
    m_up = float(np.mean(post_update)) if post_update else float("nan")
    m_no = float(np.mean(post_no_update)) if post_no_update else float("nan")
    return {
        "horizon_days": h,
        "n_update_points": len(post_update),
        "n_no_update_points": len(post_no_update),
        "mean_post_update_return": m_up,
        "mean_post_no_update_return": m_no,
        "difference_update_minus_no_update": m_up - m_no if np.isfinite(m_up) and np.isfinite(m_no) else float("nan"),
    }


def _normalize_gate_mode(v: str) -> str:
    mode = str(v).strip().lower()
    if mode in {"cadence", "none", "off"}:
        return "cadence"
    if mode in {"confidence", "gated", "gate"}:
        return "confidence"
    raise ValueError(f"Unknown update gate mode: {v}")


def _compute_gate_metrics(step_history: list[dict]) -> dict:
    if not step_history:
        return {
            "first_val_loss": float("nan"),
            "last_val_loss": float("nan"),
            "val_improvement": float("nan"),
            "relative_improvement": float("nan"),
            "n_epochs_step": 0,
        }
    first = float(step_history[0].get("val_loss", float("nan")))
    last = float(step_history[-1].get("val_loss", float("nan")))
    improvement = first - last if np.isfinite(first) and np.isfinite(last) else float("nan")
    rel = (
        improvement / (abs(first) + 1e-12)
        if np.isfinite(improvement) and np.isfinite(first)
        else float("nan")
    )
    return {
        "first_val_loss": first,
        "last_val_loss": last,
        "val_improvement": improvement,
        "relative_improvement": rel,
        "n_epochs_step": int(len(step_history)),
    }


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
    vix_series = load_vix_wrds(conn, start=start_d, end=market_end)
    print(f"VIX data from CBOE: {len(vix_series)} days")
    df = add_optional_features(df, vix_series=vix_series)
    feature_cols = list(df.columns)
    return {"df": df, "feature_cols": feature_cols}


def main():
    print("Connecting to WRDS...")
    conn = get_connection()
    print("Connected.")

    data_prep = prepare_data(conn)
    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]

    cfg = apply_env_overrides(load_best_config())
    print(f"Hyperparameters: {cfg}")

    X, R, dates = build_rolling_windows(df, window_len=cfg["window_len"], feature_cols=feature_cols)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = str(cfg.get("mode", "online")).strip().lower()

    if mode == "batch":
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
            target_vol_annual=cfg.get("target_vol_annual", 0.25),
            hidden_size=cfg["hidden_size"],
            model_type=cfg.get("model_type", "gru"),
        )
        print(f"Trained for {len(history)} epochs")
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        weights = predict_weights(model, data["X_val"], device)
        stats = portfolio_stats(weights, data["R_val"])
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

    warmup_windows = int(cfg.get("warmup_windows", 252))
    update_freq = str(cfg.get("update_freq", "W"))
    decision_lag = int(cfg.get("decision_lag", 0))
    schedule = build_online_schedule(
        dates,
        warmup_windows=warmup_windows,
        update_freq=update_freq,
        step=1,
        decision_lag=decision_lag,
    )
    if not schedule:
        raise RuntimeError("Online schedule is empty. Reduce warmup_windows or check data length.")
    print(
        f"Online mode: decisions={len(schedule)} warmup={warmup_windows} "
        f"update_freq={update_freq} lag={decision_lag}"
    )

    model = None
    optimizer = None
    online_rows: list[dict] = []
    action_dates: list[pd.Timestamp] = []
    weights_path_rows: list[np.ndarray] = []
    realized_rows: list[np.ndarray] = []
    updated_flags: list[int] = []
    online_hist_rows: list[dict] = []

    lookback = int(cfg.get("online_train_lookback", 0))
    epochs_step = int(cfg.get("epochs_step", 2))
    gate_mode = _normalize_gate_mode(cfg.get("update_gate_mode", "cadence"))
    gate_min_val_impr = float(cfg.get("gate_min_val_improvement", 0.0))
    gate_min_rel_impr = float(cfg.get("gate_min_relative_improvement", 0.0))
    gate_min_hist = int(cfg.get("gate_min_history_windows", 1))
    if gate_min_hist < 1:
        gate_min_hist = 1
    for step_idx, step in enumerate(schedule):
        decision_idx = int(step["decision_idx"])
        eval_idx = int(step["eval_idx"])
        train_end_idx = int(step["train_end_idx"])
        is_update = bool(step["is_update_step"])

        update_attempted = False
        update_applied = False
        gate_passed = True
        gate_reason = "not_update_step"
        gate_metrics = {
            "first_val_loss": float("nan"),
            "last_val_loss": float("nan"),
            "val_improvement": float("nan"),
            "relative_improvement": float("nan"),
            "n_epochs_step": 0,
        }

        if is_update:
            update_attempted = True
            X_hist, R_hist, _ = slice_windows_by_index(
                X, R, dates, end_idx=train_end_idx, lookback_windows=lookback if lookback > 0 else None
            )
            n_hist = X_hist.shape[0]
            if n_hist < 2:
                continue
            prev_model_state = copy.deepcopy(model.state_dict()) if model is not None else None
            prev_optimizer_state = (
                copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
            )
            n_val = max(1, int(n_hist * float(cfg.get("val_frac", 0.2))))
            split = max(1, n_hist - n_val)
            if split >= n_hist:
                split = n_hist - 1
            X_tr, R_tr = X_hist[:split], R_hist[:split]
            X_va, R_va = X_hist[split:], R_hist[split:]
            step_data = {
                "X_train": X_tr,
                "R_train": R_tr,
                "X_val": X_va,
                "R_val": R_va,
                "n_assets": 2,
            }
            model, optimizer, step_history = run_training_online_step(
                step_data,
                model=model,
                optimizer=optimizer,
                device=device,
                epochs_step=epochs_step,
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                patience=max(1, min(int(cfg["patience"]), epochs_step)),
                stop_on_val_deterioration=bool(cfg.get("online_stop_on_deterioration", False)),
                deterioration_tol=float(cfg.get("online_deterioration_tol", 0.0)),
                warm_start=True,
                lambda_vol=cfg["lambda_vol"],
                lambda_cvar=cfg["lambda_cvar"],
                lambda_turnover=float(cfg.get("online_lambda_turnover", cfg.get("lambda_turnover", 0.01))),
                lambda_path=float(cfg.get("online_lambda_path", cfg.get("lambda_path", 0.01))),
                lambda_diversify=cfg.get("lambda_diversify", 0.0),
                min_weight=cfg.get("min_weight", 0.1),
                lambda_vol_excess=cfg.get("lambda_vol_excess", 0.0),
                target_vol_annual=cfg.get("target_vol_annual", 0.25),
                hidden_size=cfg["hidden_size"],
                model_type=cfg.get("model_type", "gru"),
                weight_decay=1e-5,
                lr_schedule=cfg.get("lr_schedule", "constant"),
                lr_decay=cfg.get("lr_decay", 0.1),
                plateau_patience=2,
                verbose=False,
            )
            gate_metrics = _compute_gate_metrics(step_history)
            if prev_model_state is None:
                # Bootstrap update has no prior model to compare against.
                gate_passed = True
                gate_reason = "bootstrap_accept"
            elif gate_mode == "cadence":
                gate_passed = True
                gate_reason = "cadence_accept"
            else:
                val_impr = float(gate_metrics["val_improvement"])
                rel_impr = float(gate_metrics["relative_improvement"])
                enough_hist = n_hist >= gate_min_hist
                enough_val = np.isfinite(val_impr) and val_impr >= gate_min_val_impr
                enough_rel = np.isfinite(rel_impr) and rel_impr >= gate_min_rel_impr
                gate_passed = bool(enough_hist and enough_val and enough_rel)
                if not enough_hist:
                    gate_reason = "insufficient_history"
                elif not enough_val:
                    gate_reason = "weak_val_improvement"
                elif not enough_rel:
                    gate_reason = "weak_relative_improvement"
                else:
                    gate_reason = "confidence_accept"
            if not gate_passed:
                model.load_state_dict(prev_model_state)
                if optimizer is not None and prev_optimizer_state is not None:
                    optimizer.load_state_dict(prev_optimizer_state)
            update_applied = bool(gate_passed)
            for row in step_history:
                row_copy = dict(row)
                row_copy["online_step"] = step_idx
                row_copy["decision_idx"] = decision_idx
                row_copy["gate_mode"] = gate_mode
                row_copy["gate_passed"] = int(gate_passed)
                row_copy["gate_reason"] = gate_reason
                row_copy["gate_val_improvement"] = gate_metrics["val_improvement"]
                row_copy["gate_relative_improvement"] = gate_metrics["relative_improvement"]
                row_copy["n_hist"] = int(n_hist)
                online_hist_rows.append(row_copy)

        if model is None:
            continue

        x_now = X[decision_idx : decision_idx + 1]
        w_now = predict_weights(model, x_now, device)[0]
        r_realized = R[eval_idx]
        action_dates.append(pd.Timestamp(dates[eval_idx]))
        weights_path_rows.append(w_now)
        realized_rows.append(r_realized)
        updated_flags.append(int(update_applied))
        online_rows.append(
            {
                "decision_idx": decision_idx,
                "eval_idx": eval_idx,
                "train_end_idx": train_end_idx,
                "decision_date": pd.Timestamp(dates[decision_idx]),
                "eval_date": pd.Timestamp(dates[eval_idx]),
                "is_update_step": int(is_update),
                "update_attempted": int(update_attempted),
                "update_applied": int(update_applied),
                "gate_mode": gate_mode,
                "gate_passed": int(gate_passed),
                "gate_reason": gate_reason,
                "gate_val_improvement": gate_metrics["val_improvement"],
                "gate_relative_improvement": gate_metrics["relative_improvement"],
            }
        )

    if not weights_path_rows:
        raise RuntimeError("No online predictions were produced.")

    weights_online = np.asarray(weights_path_rows, dtype=np.float64)
    R_online = np.asarray(realized_rows, dtype=np.float64)
    dates_online = np.asarray(action_dates)
    eval_indices = np.asarray([int(r["eval_idx"]) for r in online_rows], dtype=int)
    metrics = evaluate_online_path(
        weights_online,
        R_online,
        cost_bps=float(cfg.get("cost_bps", 0.0)),
    )
    net_stats = metrics["net_stats"]
    gross_stats = metrics["gross_stats"]

    # Baselines on identical online dates
    w_market = np.tile(np.array([1.0, 0.0]), (R_online.shape[0], 1))
    w_ipo = np.tile(np.array([0.0, 1.0]), (R_online.shape[0], 1))
    w_eq = np.tile(np.array([0.5, 0.5]), (R_online.shape[0], 1))
    first_train_end = int(schedule[0]["train_end_idx"])
    _, R_init, _ = slice_windows_by_index(X, R, dates, end_idx=first_train_end)
    grid = np.linspace(0.0, 1.0, 101)
    best_w = 0.5
    best_sharpe = -1e9
    for w_ipo_cand in grid:
        w_cand = np.tile(np.array([1.0 - w_ipo_cand, w_ipo_cand]), (R_init.shape[0], 1))
        s_cand = portfolio_stats(w_cand, R_init)
        if s_cand["sharpe_annualized"] > best_sharpe:
            best_sharpe = s_cand["sharpe_annualized"]
            best_w = float(w_ipo_cand)
    w_best_static = np.tile(np.array([1.0 - best_w, best_w]), (R_online.shape[0], 1))
    baseline_stats = {
        "Market only:": evaluate_online_path(w_market, R_online, cost_bps=float(cfg.get("cost_bps", 0.0)))["net_stats"],
        "IPO only:": evaluate_online_path(w_ipo, R_online, cost_bps=float(cfg.get("cost_bps", 0.0)))["net_stats"],
        "Equal 50/50:": evaluate_online_path(w_eq, R_online, cost_bps=float(cfg.get("cost_bps", 0.0)))["net_stats"],
        "Best static train:": evaluate_online_path(
            w_best_static, R_online, cost_bps=float(cfg.get("cost_bps", 0.0))
        )["net_stats"],
    }
    # Offline-static model evaluated on the exact same online dates.
    X_hist0, R_hist0, _ = slice_windows_by_index(X, R, dates, end_idx=first_train_end)
    n0 = X_hist0.shape[0]
    if n0 >= 2:
        n0_val = max(1, int(n0 * float(cfg.get("val_frac", 0.2))))
        s0 = max(1, n0 - n0_val)
        if s0 >= n0:
            s0 = n0 - 1
        offline_data = {
            "X_train": X_hist0[:s0],
            "R_train": R_hist0[:s0],
            "X_val": X_hist0[s0:],
            "R_val": R_hist0[s0:],
            "n_assets": 2,
            "window_len": cfg["window_len"],
        }
        offline_model, _ = run_training(
            offline_data,
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
            lambda_diversify=cfg.get("lambda_diversify", 0.0),
            min_weight=cfg.get("min_weight", 0.1),
            lambda_vol_excess=cfg.get("lambda_vol_excess", 1.0),
            target_vol_annual=cfg.get("target_vol_annual", 0.25),
            hidden_size=cfg["hidden_size"],
            model_type=cfg.get("model_type", "gru"),
            verbose=False,
            log_every=0,
        )
        w_offline_same_dates = predict_weights(offline_model, X[eval_indices], device)
        baseline_stats["Offline static (same dates):"] = evaluate_online_path(
            w_offline_same_dates, R_online, cost_bps=float(cfg.get("cost_bps", 0.0))
        )["net_stats"]

    weights_path = out_dir / "ipo_optimizer_weights.csv"
    summary_path = out_dir / "ipo_optimizer_summary.txt"
    online_path = out_dir / "ipo_optimizer_online_path.csv"
    history_path = out_dir / "training_history_online.csv"
    schedule_path = out_dir / "online_schedule.csv"
    update_benefit_path = out_dir / "update_benefit_summary.json"
    export_weights_csv(dates_online, weights_online, weights_path)
    export_online_path_csv(
        dates_online,
        weights_online,
        R_online,
        metrics,
        online_path,
        was_model_updated=np.asarray(updated_flags),
    )
    export_online_summary(net_stats, gross_stats, summary_path, baseline_stats=baseline_stats)
    pd.DataFrame(online_hist_rows).to_csv(history_path, index=False)
    pd.DataFrame(online_rows).to_csv(schedule_path, index=False)
    update_benefit = compute_update_benefit(
        metrics["net_returns"],
        np.asarray(updated_flags),
        horizon=int(cfg.get("update_benefit_horizon", 5)),
    )
    update_benefit["gate_mode"] = gate_mode
    update_benefit["n_update_attempts"] = int(np.sum([r["update_attempted"] for r in online_rows]))
    update_benefit["n_updates_applied"] = int(np.sum(updated_flags))
    update_benefit["update_accept_rate"] = (
        float(update_benefit["n_updates_applied"]) / float(update_benefit["n_update_attempts"])
        if update_benefit["n_update_attempts"] > 0
        else float("nan")
    )
    update_benefit_path.write_text(json.dumps(update_benefit, indent=2))
    print(f"Exported weights to {weights_path}")
    print(f"Exported online path to {online_path}")
    print(f"Exported summary to {summary_path}")
    print(f"Exported update benefit to {update_benefit_path}")
    print(
        f"Online metrics (net): Sharpe={net_stats['sharpe_annualized']:.2f}, "
        f"MaxDD={net_stats['max_drawdown']:.2%}, Total={net_stats['total_return']:.2%}"
    )
    if np.isfinite(update_benefit.get("difference_update_minus_no_update", float("nan"))):
        print(
            f"Update benefit ({update_benefit['horizon_days']}d): "
            f"{update_benefit['difference_update_minus_no_update']:.4%} "
            f"(update minus no-update)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
