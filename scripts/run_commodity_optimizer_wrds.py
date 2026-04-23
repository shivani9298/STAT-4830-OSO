#!/usr/bin/env python3
"""
Run a commodity-vs-market portfolio optimizer on WRDS data.

This script is isolated from the IPO pipeline and writes distinct outputs:
  - results/commodity_optimizer_*.{csv,txt,json}
  - figures/commodity_optimizer/<model>/*.png

Assets:
  1) Market sleeve: SPY/DIA blend (default 82/18)
  2) Commodity sleeve: value-weighted commodity ETF basket from CRSP

Model architecture and training loop are reused from src.model / src.train.

Walk-forward mode:
  python3 scripts/run_commodity_optimizer_wrds.py --walk-forward
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
import torch

from src.data_layer import add_optional_features, build_rolling_windows, train_val_split
from src.export import portfolio_stats, predict_weights
from src.plot_loss import (
    plot_cumulative_returns_vs_equal_weight,
    plot_train_val_rolling_and_test,
    plot_training_loss,
    slim_history_for_json,
)
from src.train import run_training
from src.wrds_data import (
    close_wrds_connection,
    get_connection,
    load_portfolio_returns_value_weighted_wrds,
    load_sp500_dow_market_returns_wrds,
    load_vix_wrds,
)


START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

DEFAULT_COMMODITY_TICKERS = [
    "DBC",
    "GSG",
    "GLD",
    "IAU",
    "SLV",
    "USO",
    "BNO",
    "UNG",
    "DBA",
    "CORN",
    "WEAT",
    "SOYB",
    "CPER",
    "PPLT",
]

DEFAULTS = {
    "window_len": 126,
    "val_frac": 0.2,
    "epochs": 80,
    "lr": 6e-4,
    "batch_size": 64,
    "patience": 20,
    "lambda_vol": 0.35,
    "lambda_cvar": 0.8,
    "lambda_turnover": 0.0,
    "lambda_path": 0.0,
    "lambda_vol_excess": 0.5,
    "target_vol_annual": 0.25,
    "hidden_size": 64,
    "lambda_diversify": 0.0,
    "min_weight": 0.0,
    "mean_return_weight": 1.0,
    "log_growth_weight": 0.0,
    "max_grad_norm": 2.0,
    "grad_norm_mode": "clip",
    "weight_decay": 1e-6,
    "dropout": 0.1,
    "cosine_lr": False,
    "lr_schedule": "cosine",
    "lr_decay": 0.5,
    "plateau_patience": 4,
    "min_lr": 1e-6,
    "exponential_gamma": 0.99,
    "model_type": "gru",
    "seed": 42,
}


def _commodity_tickers_from_env() -> list[str]:
    raw = os.environ.get("COMMODITY_TICKERS", "").strip()
    if not raw:
        return list(DEFAULT_COMMODITY_TICKERS)
    toks = [t.strip().upper() for t in raw.split(",")]
    toks = [t for t in toks if t]
    return sorted(set(toks))


def load_config() -> dict:
    cfg = {**DEFAULTS}
    path = ROOT / "results" / "commodity_optimizer_config.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            local = json.load(f)
        for k, v in local.items():
            if k in cfg:
                cfg[k] = v
    mt = os.environ.get("COMMODITY_MODEL_TYPE", "").strip().lower()
    if mt:
        cfg["model_type"] = mt
    return cfg


def _align_market_commodity_returns(
    market: pd.Series,
    commodity: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "market_return": market.astype(float),
            "commodity_return": commodity.astype(float),
        }
    )
    df["market_return"] = df["market_return"].clip(-0.10, 0.10)
    # Commodity ETFs can move more than broad equities.
    df["commodity_return"] = df["commodity_return"].clip(-0.30, 0.30)
    df = df.sort_index().dropna()
    return df


def _build_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spread = out["commodity_return"] - out["market_return"]
    mkt = out["market_return"]
    cmd = out["commodity_return"]

    out["spread_return"] = spread
    out["spread_mom_5"] = spread.rolling(5, min_periods=2).mean()
    out["spread_mom_21"] = spread.rolling(21, min_periods=5).mean()
    out["spread_mom_63"] = spread.rolling(63, min_periods=10).mean()
    out["spread_vol_21"] = spread.rolling(21, min_periods=5).std()
    out["market_mom_21"] = mkt.rolling(21, min_periods=5).mean()
    out["commodity_mom_21"] = cmd.rolling(21, min_periods=5).mean()
    out["market_vol_21"] = mkt.rolling(21, min_periods=5).std()
    out["commodity_vol_21"] = cmd.rolling(21, min_periods=5).std()
    out["vol_ratio_21"] = out["commodity_vol_21"] / (out["market_vol_21"] + 1e-6)
    out["corr_21"] = mkt.rolling(21, min_periods=10).corr(cmd)

    if "vix" in out.columns:
        out["vix_chg_5"] = out["vix"].pct_change(5)
        out["vix_z_21"] = (
            (out["vix"] - out["vix"].rolling(21, min_periods=5).mean())
            / (out["vix"].rolling(21, min_periods=5).std() + 1e-6)
        )

    clip_cols = [
        "spread_return",
        "spread_mom_5",
        "spread_mom_21",
        "spread_mom_63",
        "spread_vol_21",
        "market_mom_21",
        "commodity_mom_21",
        "market_vol_21",
        "commodity_vol_21",
        "vol_ratio_21",
        "corr_21",
        "vix_chg_5",
        "vix_z_21",
    ]
    for c in clip_cols:
        if c in out.columns:
            out[c] = out[c].clip(-5.0, 5.0)
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return out


def _standardize_windows(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-split standardization for all input features (no target leakage)."""
    if X_train.size == 0:
        return X_train, X_val
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sd = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    X_train_n = (X_train - mu[None, None, :]) / sd[None, None, :]
    X_val_n = (X_val - mu[None, None, :]) / sd[None, None, :]
    return X_train_n.astype(np.float32), X_val_n.astype(np.float32)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_walk_forward_slices(
    n_windows: int,
    *,
    min_train_windows: int,
    val_windows: int,
    step_windows: int,
    max_folds: Optional[int],
    embargo_windows: int,
) -> list[dict]:
    """
    Build expanding-train, fixed-validation walk-forward splits over rolling-window rows.
    """
    min_tr = int(min_train_windows)
    vws = int(val_windows)
    step = int(step_windows)
    emb = max(0, int(embargo_windows))
    if min_tr <= 0 or vws <= 0 or step <= 0:
        raise ValueError("walk-forward windows must be positive integers")
    out: list[dict] = []
    val_start = min_tr
    fold = 1
    while True:
        val_end = val_start + vws
        if val_end > n_windows:
            break
        train_end = max(0, val_start - emb)
        if train_end < 50:
            break
        out.append(
            {
                "fold": fold,
                "train_slice": slice(0, train_end),
                "val_slice": slice(val_start, val_end),
                "train_end_idx": train_end - 1,
                "val_start_idx": val_start,
                "val_end_idx": val_end - 1,
            }
        )
        fold += 1
        if max_folds is not None and len(out) >= int(max_folds):
            break
        val_start += step
    return out


def _evaluate_against_static(weights: np.ndarray, R: np.ndarray) -> dict:
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(R, dtype=np.float64)
    model = portfolio_stats(w, r)
    w_m = np.tile([1.0, 0.0], (r.shape[0], 1))
    w_c = np.tile([0.0, 1.0], (r.shape[0], 1))
    w_eq = np.tile([0.5, 0.5], (r.shape[0], 1))
    m_only = portfolio_stats(w_m, r)
    c_only = portfolio_stats(w_c, r)
    eq = portfolio_stats(w_eq, r)
    wc = w[:, 1]
    return {
        "model": model,
        "market_only": m_only,
        "commodity_only": c_only,
        "equal_50_50": eq,
        "delta_vs_50_50_total_return": float(model["total_return"] - eq["total_return"]),
        "delta_vs_50_50_sharpe": float(model["sharpe_annualized"] - eq["sharpe_annualized"]),
        "weight_commodities_std": float(np.std(wc)),
        "weight_commodities_avg_abs_change": (
            float(np.mean(np.abs(np.diff(wc)))) if len(wc) > 1 else 0.0
        ),
        "weight_commodities_unique": int(np.unique(np.round(wc, 8)).size),
    }


def _run_walk_forward_cv(
    *,
    cfg: dict,
    X: np.ndarray,
    R: np.ndarray,
    dates: np.ndarray,
    feature_cols: list[str],
    df: pd.DataFrame,
    commodity_tickers: list[str],
    device: torch.device,
    walk_forward_min_train_windows: int,
    walk_forward_val_windows: int,
    walk_forward_step_windows: int,
    walk_forward_max_folds: Optional[int],
    walk_forward_embargo_windows: int,
    walk_forward_epochs: int,
) -> None:
    folds = _build_walk_forward_slices(
        n_windows=int(X.shape[0]),
        min_train_windows=walk_forward_min_train_windows,
        val_windows=walk_forward_val_windows,
        step_windows=walk_forward_step_windows,
        max_folds=walk_forward_max_folds,
        embargo_windows=walk_forward_embargo_windows,
    )
    if not folds:
        raise RuntimeError("No valid walk-forward folds; relax min_train/val/step settings.")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    model_tag = str(cfg.get("model_type", "gru")).strip().lower() or "gru"
    wf_dir = out_dir / "commodity_optimizer_walk_forward"
    wf_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fold_payloads: list[dict] = []
    print(
        f"[COMMODITY][WF] Running {len(folds)} folds "
        f"(min_train={walk_forward_min_train_windows}, val={walk_forward_val_windows}, "
        f"step={walk_forward_step_windows}, embargo={walk_forward_embargo_windows})",
        flush=True,
    )

    for spec in folds:
        fold_id = int(spec["fold"])
        tr = spec["train_slice"]
        va = spec["val_slice"]
        X_train, R_train, d_train = X[tr], R[tr], dates[tr]
        X_val, R_val, d_val = X[va], R[va], dates[va]
        X_train, X_val = _standardize_windows(X_train, X_val)
        _set_seed(int(cfg.get("seed", 42)) + fold_id)

        data_fold = {
            "X_train": X_train,
            "R_train": R_train,
            "dates_train": d_train,
            "X_val": X_val,
            "R_val": R_val,
            "dates_val": d_val,
            "feature_cols": feature_cols,
            "df": df,
            "n_assets": 2,
            "window_len": int(cfg["window_len"]),
        }
        print(
            f"[COMMODITY][WF] Fold {fold_id}: train={len(X_train)} val={len(X_val)} "
            f"({pd.Timestamp(d_val[0]).date()} to {pd.Timestamp(d_val[-1]).date()})",
            flush=True,
        )
        model, history = run_training(
            data_fold,
            device=device,
            epochs=int(walk_forward_epochs),
            lr=float(cfg["lr"]),
            batch_size=int(cfg["batch_size"]),
            patience=int(cfg["patience"]),
            lambda_vol=float(cfg["lambda_vol"]),
            lambda_cvar=float(cfg["lambda_cvar"]),
            lambda_turnover=float(cfg.get("lambda_turnover", 0.0)),
            lambda_path=float(cfg.get("lambda_path", 0.0)),
            lambda_diversify=float(cfg.get("lambda_diversify", 0.0)),
            min_weight=float(cfg.get("min_weight", 0.0)),
            lambda_vol_excess=float(cfg.get("lambda_vol_excess", 0.5)),
            target_vol_annual=float(cfg.get("target_vol_annual", 0.25)),
            hidden_size=int(cfg["hidden_size"]),
            model_type=str(cfg.get("model_type", "gru")),
            verbose=False,
            log_every=0,
            weight_decay=float(cfg.get("weight_decay", 1e-6)),
            dropout=float(cfg.get("dropout", 0.1)),
            cosine_lr=bool(cfg.get("cosine_lr", False)),
            lr_schedule=cfg.get("lr_schedule"),
            lr_decay=float(cfg.get("lr_decay", 0.5)),
            plateau_patience=int(cfg.get("plateau_patience", 4)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
            exponential_gamma=float(cfg.get("exponential_gamma", 0.99)),
            mean_return_weight=float(cfg.get("mean_return_weight", 1.0)),
            log_growth_weight=float(cfg.get("log_growth_weight", 0.0)),
            max_grad_norm=float(cfg.get("max_grad_norm", 2.0)),
            grad_norm_mode=str(cfg.get("grad_norm_mode", "clip")),
        )

        weights = predict_weights(model, X_val, device)
        eval_pack = _evaluate_against_static(weights, R_val)
        model_stats = eval_pack["model"]
        eq_stats = eval_pack["equal_50_50"]
        rows.append(
            {
                "fold": fold_id,
                "model_type": model_tag,
                "train_windows": int(len(X_train)),
                "val_windows": int(len(X_val)),
                "val_start": str(pd.Timestamp(d_val[0]).date()),
                "val_end": str(pd.Timestamp(d_val[-1]).date()),
                "model_total_return": float(model_stats["total_return"]),
                "model_ann_return": float(model_stats["return_annualized"]),
                "model_ann_vol": float(model_stats["volatility_annualized"]),
                "model_sharpe": float(model_stats["sharpe_annualized"]),
                "model_max_drawdown": float(model_stats["max_drawdown"]),
                "eq50_total_return": float(eq_stats["total_return"]),
                "eq50_sharpe": float(eq_stats["sharpe_annualized"]),
                "delta_vs_50_50_total_return": float(eval_pack["delta_vs_50_50_total_return"]),
                "delta_vs_50_50_sharpe": float(eval_pack["delta_vs_50_50_sharpe"]),
                "w_commodity_std": float(eval_pack["weight_commodities_std"]),
                "w_commodity_avg_abs_change": float(eval_pack["weight_commodities_avg_abs_change"]),
                "w_commodity_unique": int(eval_pack["weight_commodities_unique"]),
                "epochs_completed": int(len(history)),
                "best_val_loss": float(min(h["val_loss"] for h in history)) if history else float("nan"),
            }
        )
        fold_payload = {
            "fold": fold_id,
            "history": slim_history_for_json(history),
            "metrics": eval_pack,
            "val_start": str(pd.Timestamp(d_val[0]).date()),
            "val_end": str(pd.Timestamp(d_val[-1]).date()),
        }
        fold_payloads.append(fold_payload)

        fold_tag = f"fold{fold_id:02d}_{pd.Timestamp(d_val[0]).strftime('%Y%m%d')}_{pd.Timestamp(d_val[-1]).strftime('%Y%m%d')}"
        _write_weights_csv(d_val, weights, wf_dir / f"commodity_optimizer_weights_{fold_tag}.csv")
        _write_returns_csv(d_val, R_val, wf_dir / f"commodity_optimizer_returns_{fold_tag}.csv")
        with open(wf_dir / f"commodity_optimizer_history_{fold_tag}.json", "w", encoding="utf-8") as f:
            json.dump(slim_history_for_json(history), f, indent=2)

        print(
            f"[COMMODITY][WF] Fold {fold_id} done: "
            f"delta_vs_50_50_total={eval_pack['delta_vs_50_50_total_return']:+.2%}, "
            f"delta_vs_50_50_sharpe={eval_pack['delta_vs_50_50_sharpe']:+.3f}, "
            f"w_unique={eval_pack['weight_commodities_unique']}",
            flush=True,
        )

    summary_df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    summary_csv = out_dir / "commodity_optimizer_walk_forward_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    agg = {
        "model_type": model_tag,
        "commodity_universe": commodity_tickers,
        "n_folds": int(len(summary_df)),
        "mean_delta_vs_50_50_total_return": float(summary_df["delta_vs_50_50_total_return"].mean()),
        "median_delta_vs_50_50_total_return": float(summary_df["delta_vs_50_50_total_return"].median()),
        "mean_delta_vs_50_50_sharpe": float(summary_df["delta_vs_50_50_sharpe"].mean()),
        "folds_positive_total_return_vs_50_50": int((summary_df["delta_vs_50_50_total_return"] > 0).sum()),
        "folds_positive_sharpe_vs_50_50": int((summary_df["delta_vs_50_50_sharpe"] > 0).sum()),
        "mean_weight_std": float(summary_df["w_commodity_std"].mean()),
        "mean_weight_abs_change": float(summary_df["w_commodity_avg_abs_change"].mean()),
    }
    agg_json = out_dir / "commodity_optimizer_walk_forward_aggregate.json"
    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "folds": fold_payloads}, f, indent=2)

    print(f"[COMMODITY][WF] Summary CSV: {summary_csv}", flush=True)
    print(f"[COMMODITY][WF] Aggregate JSON: {agg_json}", flush=True)


def _write_weights_csv(dates: np.ndarray, weights: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        weights,
        index=pd.DatetimeIndex(dates),
        columns=["weight_market", "weight_commodities"],
    )
    df.index.name = "date"
    df.to_csv(out_path)


def _write_returns_csv(dates: np.ndarray, R: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(R, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"R must be (N,2); got {arr.shape}")
    eq = 0.5 * arr[:, 0] + 0.5 * arr[:, 1]
    df = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "market_return": arr[:, 0],
            "commodity_return": arr[:, 1],
            "equal_weight_return": eq,
        }
    )
    df.to_csv(out_path, index=False)


def _write_dataset_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index)
    out.index.name = "date"
    out.to_csv(out_path)


def _write_summary(
    stats: dict,
    weights: np.ndarray,
    R: np.ndarray,
    out_path: Path,
    tickers: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(R, dtype=np.float64)
    avg_c = float(np.mean(w[:, 1])) if w.shape[1] >= 2 else 0.0
    pct_c_gt20 = float(np.mean(w[:, 1] > 0.2)) * 100.0 if w.shape[1] >= 2 else 0.0
    w_std = float(np.std(w[:, 1])) if w.shape[1] >= 2 else 0.0
    avg_turn = float(np.mean(np.abs(np.diff(w[:, 1])))) if w.shape[0] > 1 and w.shape[1] >= 2 else 0.0
    n_unique = int(np.unique(np.round(w[:, 1], 8)).size) if w.shape[1] >= 2 else 0

    w_m = np.tile([1.0, 0.0], (r.shape[0], 1))
    w_c = np.tile([0.0, 1.0], (r.shape[0], 1))
    w_eq = np.tile([0.5, 0.5], (r.shape[0], 1))
    s_m = portfolio_stats(w_m, r)
    s_c = portfolio_stats(w_c, r)
    s_eq = portfolio_stats(w_eq, r)

    def _line(name: str, s: dict) -> str:
        return (
            f"  {name:13} Total={s['total_return']:.2%}  "
            f"AnnRet={s['return_annualized']:.2%}  "
            f"AnnVol={s['volatility_annualized']:.2%}  "
            f"Sharpe={s['sharpe_annualized']:.2f}  "
            f"MaxDD={s['max_drawdown']:.2%}"
        )

    lines = [
        "Commodity Optimizer — Summary",
        "==============================",
        "",
        f"Commodity universe ({len(tickers)} tickers): {', '.join(tickers)}",
        "",
        "Model Portfolio:",
        f"  Total return:           {stats.get('total_return', 0):.2%}",
        f"  Return (annualized):    {stats.get('return_annualized', 0):.2%}",
        f"  Volatility (annual):    {stats.get('volatility_annualized', 0):.2%}",
        f"  Sharpe (annualized):    {stats.get('sharpe_annualized', 0):.2f}",
        f"  Max drawdown:           {stats.get('max_drawdown', 0):.2%}",
        f"  Avg turnover:           {stats.get('avg_turnover', 0):.4f}",
        f"  Average commodity wt:   {avg_c:.2%}",
        f"  % days commodity wt>20%: {pct_c_gt20:.1f}%",
        f"  Commodity weight std:   {w_std:.4f}",
        f"  Mean abs wt change/day: {avg_turn:.4f}",
        f"  Unique commodity wts:   {n_unique}",
        "",
        "Baseline Comparison (same period):",
        _line("Market only:", s_m),
        _line("Commodity only:", s_c),
        _line("Equal 50/50:", s_eq),
        "",
        "Model minus 50/50:",
        f"  Delta total return:     {stats.get('total_return', 0.0) - s_eq['total_return']:+.2%}",
        f"  Delta ann return:       {stats.get('return_annualized', 0.0) - s_eq['return_annualized']:+.2%}",
        f"  Delta Sharpe:           {stats.get('sharpe_annualized', 0.0) - s_eq['sharpe_annualized']:+.3f}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_commodity_data(
    conn,
    *,
    start: str,
    end: str,
    commodity_tickers: list[str],
) -> dict:
    print(f"[COMMODITY] Loading SPY/DIA market returns ({start}–{end})...", flush=True)
    market_ret = load_sp500_dow_market_returns_wrds(
        conn,
        start=start,
        end=end,
        w_sp500=0.82,
        w_dow=0.18,
    )
    if len(market_ret) < 100:
        raise RuntimeError("Insufficient market data from WRDS SPY/DIA blend.")

    print(
        f"[COMMODITY] Loading value-weighted commodity basket from CRSP using "
        f"{len(commodity_tickers)} tickers...",
        flush=True,
    )
    commodity_ret = load_portfolio_returns_value_weighted_wrds(
        conn,
        start=start,
        end=end,
        tickers=commodity_tickers,
    )
    if len(commodity_ret) < 100:
        raise RuntimeError(
            "Insufficient commodity basket data from WRDS. "
            "Try adjusting COMMODITY_TICKERS or date range."
        )
    commodity_ret = commodity_ret.rename("commodity_return")

    df = _align_market_commodity_returns(market_ret, commodity_ret)
    if len(df) < 300:
        raise RuntimeError(f"Aligned market/commodity sample too short: {len(df)} rows")

    vix = pd.Series(dtype=float)
    try:
        print("[COMMODITY] Loading WRDS VIX (cboe.cboe)...", flush=True)
        vix = load_vix_wrds(conn, start=start, end=end)
    except Exception as e:
        print(f"[COMMODITY] WRDS VIX unavailable; using fallback constant VIX. Detail: {e}", flush=True)
    df = add_optional_features(df, include_vix=False, vix_series=vix if len(vix) > 0 else None)
    df = _build_timing_features(df)

    feature_cols = ["market_return", "commodity_return"] + [
        c for c in df.columns if c not in ("market_return", "commodity_return")
    ]
    return {"df": df, "feature_cols": feature_cols}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train commodity optimizer on WRDS")
    p.add_argument(
        "--model",
        default=None,
        choices=["gru", "lstm", "transformer", "mlp", "hybrid"],
        help="Override model architecture for this run",
    )
    p.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run expanding-window walk-forward CV instead of a single train/val split.",
    )
    p.add_argument("--wf-min-train-windows", type=int, default=1260)
    p.add_argument("--wf-val-windows", type=int, default=252)
    p.add_argument("--wf-step-windows", type=int, default=126)
    p.add_argument("--wf-max-folds", type=int, default=6)
    p.add_argument("--wf-embargo-windows", type=int, default=21)
    p.add_argument("--wf-epochs", type=int, default=40)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config()
    if args.model:
        cfg["model_type"] = args.model
    _set_seed(int(cfg.get("seed", 42)))

    start = os.environ.get("COMMODITY_DATA_START", "").strip() or START_DATE
    end = os.environ.get("COMMODITY_DATA_END", "").strip() or END_DATE
    commodity_tickers = _commodity_tickers_from_env()

    print("Connecting to WRDS...", flush=True)
    conn = get_connection()
    print("Connected.", flush=True)
    try:
        prep = prepare_commodity_data(
            conn,
            start=start,
            end=end,
            commodity_tickers=commodity_tickers,
        )
    finally:
        close_wrds_connection(conn)

    df = prep["df"]
    feature_cols = prep["feature_cols"]

    print(f"[COMMODITY] Hyperparameters: {cfg}", flush=True)
    X, R, dates = build_rolling_windows(df, window_len=int(cfg["window_len"]), feature_cols=feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.walk_forward:
        _run_walk_forward_cv(
            cfg=cfg,
            X=X,
            R=R,
            dates=dates,
            feature_cols=feature_cols,
            df=df,
            commodity_tickers=commodity_tickers,
            device=device,
            walk_forward_min_train_windows=int(args.wf_min_train_windows),
            walk_forward_val_windows=int(args.wf_val_windows),
            walk_forward_step_windows=int(args.wf_step_windows),
            walk_forward_max_folds=int(args.wf_max_folds) if args.wf_max_folds > 0 else None,
            walk_forward_embargo_windows=int(args.wf_embargo_windows),
            walk_forward_epochs=int(args.wf_epochs),
        )
        return 0

    X_train, R_train, d_train, X_val, R_val, d_val = train_val_split(
        X, R, dates, val_frac=float(cfg["val_frac"])
    )
    X_train, X_val = _standardize_windows(X_train, X_val)
    print(f"[COMMODITY] Train windows: {len(X_train)}, Val windows: {len(X_val)}", flush=True)

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
        "window_len": int(cfg["window_len"]),
    }

    print("[COMMODITY] Starting model training...", flush=True)
    model, history = run_training(
        data,
        device=device,
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        batch_size=int(cfg["batch_size"]),
        patience=int(cfg["patience"]),
        lambda_vol=float(cfg["lambda_vol"]),
        lambda_cvar=float(cfg["lambda_cvar"]),
        lambda_turnover=float(cfg.get("lambda_turnover", 0.0)),
        lambda_path=float(cfg.get("lambda_path", 0.0)),
        lambda_diversify=float(cfg.get("lambda_diversify", 0.0)),
        min_weight=float(cfg.get("min_weight", 0.1)),
        lambda_vol_excess=float(cfg.get("lambda_vol_excess", 1.0)),
        target_vol_annual=float(cfg.get("target_vol_annual", 0.25)),
        hidden_size=int(cfg["hidden_size"]),
        model_type=str(cfg.get("model_type", "gru")),
        verbose=True,
        log_every=1,
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        dropout=float(cfg.get("dropout", 0.1)),
        cosine_lr=bool(cfg.get("cosine_lr", False)),
        lr_schedule=cfg.get("lr_schedule"),
        lr_decay=float(cfg.get("lr_decay", 0.5)),
        plateau_patience=int(cfg.get("plateau_patience", 4)),
        min_lr=float(cfg.get("min_lr", 1e-6)),
        exponential_gamma=float(cfg.get("exponential_gamma", 0.99)),
        mean_return_weight=float(cfg.get("mean_return_weight", 1.0)),
        log_growth_weight=float(cfg.get("log_growth_weight", 0.0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        grad_norm_mode=str(cfg.get("grad_norm_mode", "clip")),
    )
    print(f"[COMMODITY] Trained for {len(history)} epochs", flush=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    model_tag = str(cfg.get("model_type", "gru")).strip().lower() or "gru"
    hist_path = out_dir / "commodity_optimizer_training_history.json"
    hist_tagged = out_dir / f"commodity_optimizer_training_history_{model_tag}.json"
    slim = slim_history_for_json(history)
    for p in (hist_path, hist_tagged):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2)
    print(f"[COMMODITY] Saved history to {hist_path} and {hist_tagged}", flush=True)

    fig_dir = ROOT / "figures" / "commodity_optimizer" / model_tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("COMMODITY_SAVE_LOSS_PLOTS", "1").strip().lower() in ("1", "true", "yes"):
        rolling_path = plot_train_val_rolling_and_test(
            history,
            fig_dir / "loss_train_val_rolling.png",
            test_loss=None,
            rolling_epochs=max(1, int(os.environ.get("COMMODITY_LOSS_ROLLING_EPOCHS", "3"))),
            title="Commodity optimizer: training/validation loss",
        )
        semilog_path = plot_training_loss(
            history,
            fig_dir / "loss_semilogy.png",
            title="Commodity optimizer: training/validation loss",
            semilogy=True,
        )
        print(f"[COMMODITY] Saved loss plots: {rolling_path}, {semilog_path}", flush=True)

    weights = predict_weights(model, data["X_val"], device)
    stats = portfolio_stats(weights, data["R_val"])
    weights_path = out_dir / "commodity_optimizer_weights.csv"
    summary_path = out_dir / "commodity_optimizer_summary.txt"
    returns_path = out_dir / "commodity_optimizer_returns_val.csv"
    dataset_path = out_dir / "commodity_optimizer_dataset.csv"

    _write_weights_csv(data["dates_val"], weights, weights_path)
    _write_returns_csv(data["dates_val"], data["R_val"], returns_path)
    _write_dataset_csv(df, dataset_path)
    _write_summary(stats, weights, data["R_val"], summary_path, commodity_tickers)
    print(f"[COMMODITY] Exported weights to {weights_path}", flush=True)
    print(f"[COMMODITY] Exported summary to {summary_path}", flush=True)
    print(f"[COMMODITY] Exported returns to {returns_path}", flush=True)
    print(f"[COMMODITY] Exported aligned dataset to {dataset_path}", flush=True)

    cmp_path = plot_cumulative_returns_vs_equal_weight(
        weights,
        data["R_val"],
        data["dates_val"],
        fig_dir / "validation_returns_vs_equal_weight.png",
        title="Validation: cumulative growth vs 50/50 market/commodities",
    )
    print(f"[COMMODITY] Saved returns-vs-50/50 plot to {cmp_path}", flush=True)
    print(
        f"[COMMODITY] Metrics: Sharpe={stats['sharpe_annualized']:.2f}, "
        f"MaxDD={stats['max_drawdown']:.2%}, AnnRet={stats['return_annualized']:.2%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
