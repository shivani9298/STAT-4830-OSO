#!/usr/bin/env python3
"""
Run IPO Portfolio Optimizer on 2005-01-01 to 2024-12-31 (see START_DATE / END_DATE).
IPO data: SDC New Deals (all rows where ipodate is not null) + CRSP daily prices (split-adjusted).
Market: Market-cap weighted portfolio of S&P 500 (SPY) and Dow Jones (DIA) from CRSP.
Uses best config from results/ipo_optimizer_best_config.json if present (from tune_hyperparameters_wrds.py).
Set ``model_type`` to ``transformer`` in ``best_config``, in gitignored ``local/ipo_optimizer_config.json``,
or via env ``IPO_MODEL_TYPE=transformer``. When ``model_type`` is ``transformer``, ``TRANSFORMER_CONFIG``
(lr, batch_size, lambda_cvar, hidden_size, weight_decay, dropout, cosine_lr) is merged after ``best_config``;
``local/ipo_optimizer_config.json`` is applied last so you can override any key.
Set ``IPO_LOCAL_CONFIG`` to a path (repo-relative or absolute) to use a different JSON file
for the same merge (e.g. run several transformer variants in sequence).
Set ``IPO_EXPORT_ATTENTION=1`` to save self-attention maps (``results/ipo_optimizer_attention.npz`` and
``figures/ipo_optimizer_attention_layer0.png``) when ``model_type`` is ``transformer``.
Set ``IPO_SECTOR_PORTFOLIOS=0`` for a single market-vs-IPO index (overrides ``SECTOR_PORTFOLIOS``).
Set ``IPO_SECTOR_SOURCE`` to ``yfinance`` (Yahoo ``info['sector']``), ``compustat`` (default:
Compustat GICS via ``comp.funda``/``comp.company`` join on ticker), or ``ccm`` / ``wrds_chain``
(date-valid chain: ``match_date`` = max(ipo_date, first CRSP price date) → ``stocknames`` /
``dsenames`` → CCM → Compustat GICS; see ``docs/SECTOR_CCM_WORKFLOW.md`` and
``src/wrds_ipo_gics_enrichment.py``). Cache files: ``results/ticker_sector_cache_*.csv``.
Pre-build CCM labels only: ``python scripts/generate_sector_cache_ccm.py`` (defaults
2010–2024 IPO offer dates).

When SECTOR_PORTFOLIOS is True (default): sector labels (GICS or Yahoo) group IPOs into baskets. A mcap-weighted
IPO basket is built per sector; one shared encoder (GRU/LSTM or Transformer) feeds separate two-way softmax heads (market vs
that sector basket). Exports ``results/ipo_optimizer_weights_sector_*.csv`` and
``results/ipo_optimizer_summary_by_sector.txt``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Windows: PyTorch + NumPy/MKL each link Intel OpenMP; loading both aborts without this.
# Must be set before importing numpy or torch.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.wrds_data import (
    close_wrds_connection,
    get_connection,
    load_ipo_data_from_sdc_wrds,
    load_market_returns_wrds,
    load_sdc_ipo_dates_wrds,
    load_sp500_dow_market_returns_wrds,
    load_stock_returns_wrds,
)
from src.data_layer import (
    add_optional_features,
    align_returns,
    build_rolling_windows,
    build_rolling_windows_sector_heads,
    train_val_split,
)
from src.train import run_training, run_training_sector_heads
from src.export import (
    export_sector_group_outputs,
    export_summary,
    export_weights_csv,
    predict_sector_head_weights,
    predict_weights,
    portfolio_stats,
)
from src.plot_loss import slim_history_for_json
from src.policy_layer import ipo_tilt_to_position_scale, policy_rule
from src.sector_ipo import (
    fetch_ticker_sectors,
    fetch_ticker_sectors_ccm_chain,
    fetch_ticker_sectors_compustat,
    group_tickers_by_sector,
    sector_column_name,
)

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
# Rolling-window splits (prediction date); must match tune_hyperparameters_wrds embargo logic.
VAL_START = "2019-01-01"
TEST_START = "2022-01-01"

# When True: Yahoo Finance sectors → per-sector IPO baskets, shared GRU + one 2-way softmax per sector.
SECTOR_PORTFOLIOS = True
MIN_TICKERS_PER_SECTOR_GROUP = 12

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
    "weight_decay": 1e-5,
    "dropout": 0.1,
    "cosine_lr": False,
}

# Applied when ``model_type`` is ``transformer`` (after JSON / local overrides), so GRU/LSTM defaults stay unchanged.
TRANSFORMER_CONFIG = {
    "lr": 3e-4,
    "batch_size": 64,
    "lambda_cvar": 1.0,
    "hidden_size": 128,
    "weight_decay": 1e-2,
    "dropout": 0.1,
    "cosine_lr": False,
}

_CONFIG_OPTIONAL = frozenset(
    {"model_type", "num_layers", "weight_decay", "dropout", "cosine_lr"}
)


def load_best_config():
    """Load best config from tuning; fall back to DEFAULTS if not found."""
    cfg = {**DEFAULTS}
    path = ROOT / "results" / "ipo_optimizer_best_config.json"
    if path.exists():
        with open(path) as f:
            out = json.load(f)
        best = out.get("best_config") or {}
        for k, v in best.items():
            if k in cfg or k in _CONFIG_OPTIONAL:
                cfg[k] = v
    mt = os.environ.get("IPO_MODEL_TYPE", "").strip()
    if mt:
        cfg["model_type"] = mt
    if cfg.get("model_type") == "transformer":
        cfg.update(TRANSFORMER_CONFIG)
    local_override = os.environ.get("IPO_LOCAL_CONFIG", "").strip()
    if local_override:
        local_path = Path(local_override)
        if not local_path.is_absolute():
            local_path = ROOT / local_path
    else:
        local_path = ROOT / "local" / "ipo_optimizer_config.json"
    if local_path.exists():
        with open(local_path) as f:
            local = json.load(f)
        for k, v in local.items():
            if k in cfg or k in _CONFIG_OPTIONAL:
                cfg[k] = v
    return cfg


def sector_portfolios_effective() -> bool:
    """If ``IPO_SECTOR_PORTFOLIOS`` is set, use it; else ``SECTOR_PORTFOLIOS``."""
    v = os.environ.get("IPO_SECTOR_PORTFOLIOS", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return SECTOR_PORTFOLIOS


def build_ipo_index_mcap(
    prices_df,
    ipo_dates_df,
    shares_dict,
    holding_days=180,
    min_names=1,
    ticker_allowlist: set[str] | None = None,
):
    ipo_lookup = dict(zip(ipo_dates_df["ticker"], ipo_dates_df["ipo_date"]))
    if ticker_allowlist is not None:
        ipo_lookup = {k: v for k, v in ipo_lookup.items() if k in ticker_allowlist}
    returns_df = prices_df.pct_change()
    trading_days = {
        t: prices_df[t].dropna().index.tolist()
        for t in prices_df.columns
        if t != "SPY" and t in ipo_lookup
    }
    all_dates = prices_df.index.tolist()
    n_all = len(all_dates)
    index_data = []
    progress_every = max(1, n_all // 20)
    for i, date in enumerate(all_dates):
        if i == 0 or (i + 1) % progress_every == 0 or i == n_all - 1:
            print(
                f"  [IPO] IPO index progress: day {i + 1}/{n_all} ({100 * (i + 1) / n_all:.0f}%)",
                flush=True,
            )
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


def prepare_data(
    conn,
    start: str | None = None,
    end: str | None = None,
    *,
    sector_portfolios: bool = False,
):
    """Load and prepare IPO + market data. Returns dict with df, feature_cols for rolling windows."""
    start = start or START_DATE
    end = end or END_DATE
    print(
        f"[IPO] Loading data {start}–{end}: SDC IPO list + CRSP prices (large WRDS query; "
        "often several minutes, not frozen)...",
        flush=True,
    )
    # IPO data: SDC New Deals (all rows where ipodate is not null) + CRSP daily prices
    ipo_csv = load_ipo_data_from_sdc_wrds(
        conn, start=start, end=end, library="sdc", price_source="crsp"
    )
    print(f"IPO data from SDC + CRSP: {len(ipo_csv)} rows, {ipo_csv['tic'].nunique()} tickers")

    ipo_csv["datadate"] = pd.to_datetime(ipo_csv["datadate"])
    ipo_csv = ipo_csv.drop_duplicates(subset=["tic", "datadate"], keep="first")

    prices_ipo = ipo_csv.pivot_table(index="datadate", columns="tic", values="prccd")
    prices_ipo.index = pd.to_datetime(prices_ipo.index).normalize()

    # IPO dates from SDC (not first trading date); filter to tickers with prices
    print("[IPO] Loading SDC IPO dates for tickers...", flush=True)
    ipo_dates = load_sdc_ipo_dates_wrds(
        conn, start=start, end=end, library="sdc"
    )
    ipo_df = ipo_dates[ipo_dates["ticker"].isin(prices_ipo.columns)].copy()
    ipo_df = ipo_df.sort_values("ipo_date").reset_index(drop=True)

    start_d = prices_ipo.index.min().strftime("%Y-%m-%d")
    end_d = prices_ipo.index.max().strftime("%Y-%m-%d")
    print(f"IPO tickers: {len(ipo_df)}, Date range: {start_d} to {end_d}")

    # Market returns: market-cap weighted S&P 500 (82%) + Dow Jones (18%) from CRSP
    # Use full date range through `end` to align with requested sample
    market_end = max(end_d, end) if end_d else end
    print("[IPO] Loading market returns (CRSP SPY/DIA blend)...", flush=True)
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
                and datadate >= '{start}'
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

    n_days = len(prices.index)

    if sector_portfolios:
        _src = os.environ.get("IPO_SECTOR_SOURCE", "compustat").strip().lower()
        if _src in ("yfinance", "yahoo", "yf"):
            cache = ROOT / "results" / "ticker_sector_cache_yfinance.csv"
            sec_series = fetch_ticker_sectors(ipo_tickers, cache_path=cache, verbose=True)
        elif _src in ("ccm", "wrds_chain", "chain", "ccm_gics"):
            cache = ROOT / "results" / "ticker_sector_cache_ccm.csv"
            ipo_ccm = ipo_df[["ticker", "ipo_date"]].copy()
            _fd: dict[str, pd.Timestamp] = {}
            for _tic in ipo_ccm["ticker"]:
                if _tic in prices_ipo.columns:
                    _s = prices_ipo[_tic].dropna()
                    if len(_s) > 0:
                        _fd[_tic] = pd.Timestamp(_s.index.min()).normalize()
            ipo_ccm["first_crsp_date"] = ipo_ccm["ticker"].map(_fd)
            sec_series = fetch_ticker_sectors_ccm_chain(
                conn,
                ipo_ccm,
                ipo_tickers,
                cache_path=cache,
                verbose=True,
            )
        else:
            cache = ROOT / "results" / "ticker_sector_cache_compustat.csv"
            sec_series = fetch_ticker_sectors_compustat(
                conn, ipo_tickers, cache_path=cache, verbose=True
            )
        groups = group_tickers_by_sector(
            ipo_tickers, sec_series, min_names=MIN_TICKERS_PER_SECTOR_GROUP
        )
        if not groups:
            raise RuntimeError(
                "No sector groups met MIN_TICKERS_PER_SECTOR_GROUP="
                f"{MIN_TICKERS_PER_SECTOR_GROUP}. Lower that constant or check sector cache."
            )
        sector_labels_sorted = sorted(groups.keys(), key=lambda x: (-len(groups[x]), x))
        print(
            f"[IPO] Sector baskets: {len(sector_labels_sorted)} groups — "
            f"{[(k, len(groups[k])) for k in sector_labels_sorted[:8]]}"
            + (" ..." if len(sector_labels_sorted) > 8 else ""),
            flush=True,
        )
        combined = pd.DataFrame({"market_return": market_ret})
        combined["market_return"] = combined["market_return"].clip(-0.10, 0.10)
        sector_cols: list[str] = []
        for label in sector_labels_sorted:
            allow = set(groups[label])
            col = sector_column_name(label)
            print(
                f"[IPO] Sector IPO index ({label}, n={len(allow)}): "
                f"{n_days} days (CPU-heavy)...",
                flush=True,
            )
            ipo_index_s = build_ipo_index_mcap(
                prices,
                ipo_df,
                shares_outstanding,
                holding_days=180,
                min_names=1,
                ticker_allowlist=allow,
            )
            s = ipo_index_s["ipo_ret"].clip(-0.50, 0.50).rename(col)
            combined[col] = s
            sector_cols.append(col)
        # Days with no sector IPO basket activity are NaN in the index; R uses nan_to_num(sec)
        # but model inputs X are built from full feature_cols — NaNs there become NaN weights and
        # garbage per-sector Sharpe/MaxDD. Treat missing sector return as 0 (no IPO sleeve).
        combined[sector_cols] = combined[sector_cols].fillna(0.0)
        combined = combined.sort_index().dropna(subset=["market_return"])
        df = add_optional_features(combined, include_vix=False)
        feature_cols = ["market_return"] + sector_cols + [c for c in ("rolling_vol", "vix") if c in df.columns]
        return {
            "df": df,
            "feature_cols": feature_cols,
            "sector_portfolios": True,
            "sector_labels": sector_labels_sorted,
            "sector_ret_cols": sector_cols,
        }

    print(
        f"[IPO] Building market-cap IPO index over {n_days} trading days (CPU-heavy; "
        f"can take 1–15+ min with many tickers)...",
        flush=True,
    )
    ipo_index = build_ipo_index_mcap(prices, ipo_df, shares_outstanding, holding_days=180)
    print(f"IPO index: {ipo_index['ipo_ret'].notna().sum()} days with valid returns")

    ipo_ret = ipo_index["ipo_ret"].rename("ipo_return")
    df = align_returns(market_ret, ipo_ret)
    df = add_optional_features(df, include_vix=False)
    feature_cols = list(df.columns)
    return {"df": df, "feature_cols": feature_cols, "sector_portfolios": False}


def main():
    print("Connecting to WRDS...")
    conn = get_connection()
    print("Connected.")
    print(
        "[IPO] Next steps load SDC/CRSP data then build the IPO index. "
        "Long gaps in output are normal—queries are running on WRDS.",
        flush=True,
    )

    try:
        sp = sector_portfolios_effective()
        print(
            f"[IPO] sector_portfolios={sp}  (env IPO_SECTOR_PORTFOLIOS or SECTOR_PORTFOLIOS={SECTOR_PORTFOLIOS})",
            flush=True,
        )
        data_prep = prepare_data(conn, sector_portfolios=sp)
    finally:
        close_wrds_connection(conn)
    df = data_prep["df"]
    feature_cols = data_prep["feature_cols"]
    use_sectors = bool(data_prep.get("sector_portfolios"))
    sector_labels = data_prep.get("sector_labels") or []
    sector_ret_cols = data_prep.get("sector_ret_cols") or []

    cfg = load_best_config()
    print(f"Hyperparameters: {cfg}")

    if use_sectors:
        X, R, dates = build_rolling_windows_sector_heads(
            df,
            window_len=cfg["window_len"],
            feature_cols=feature_cols,
            sector_ret_cols=sector_ret_cols,
        )
    else:
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
    if use_sectors:
        data["n_sectors"] = len(sector_labels)
    print(f"Train windows: {X_train.shape[0]}, Val windows: {X_val.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[IPO] Starting model training (watch for [IPO] epoch lines below)...", flush=True)
    if use_sectors:
        model, history = run_training_sector_heads(
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
            model_type=cfg.get("model_type", "gru"),
            verbose=True,
            log_every=1,
            weight_decay=cfg.get("weight_decay", 1e-5),
            dropout=cfg.get("dropout", 0.1),
            cosine_lr=cfg.get("cosine_lr", False),
        )
    else:
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
            model_type=cfg.get("model_type", "gru"),
            verbose=True,
            log_every=1,
            weight_decay=cfg.get("weight_decay", 1e-5),
            dropout=cfg.get("dropout", 0.1),
            cosine_lr=cfg.get("cosine_lr", False),
        )
    print(f"Trained for {len(history)} epochs")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    hist_path = out_dir / "ipo_optimizer_training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(slim_history_for_json(history), f, indent=2)
    print(f"Saved training history to {hist_path}")

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
    title = "IPO Optimizer: Training and Validation Loss"
    if use_sectors:
        title += f" ({len(sector_labels)} sector heads)"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "ipo_optimizer_loss.png", dpi=150)
    plt.close()
    print(f"Saved loss plot to {fig_dir / 'ipo_optimizer_loss.png'}")

    if os.environ.get("IPO_EXPORT_ATTENTION", "").strip().lower() in ("1", "true", "yes"):
        if cfg.get("model_type") == "transformer":
            from src.attention_export import save_attention_heatmap_png, save_attention_npz
            from src.model import SectorMultiHeadTransformerAllocator, TransformerAllocator

            if isinstance(model, (TransformerAllocator, SectorMultiHeadTransformerAllocator)):
                n = min(32, int(data["X_val"].shape[0]))
                x_s = torch.as_tensor(data["X_val"][:n], device=device, dtype=torch.float32)
                ap = out_dir / "ipo_optimizer_attention.npz"
                _, maps = save_attention_npz(
                    model,
                    x_s,
                    ap,
                    meta={"window_len": str(cfg["window_len"]), "batch": str(n)},
                )
                if maps:
                    save_attention_heatmap_png(
                        maps[0],
                        ROOT / "figures" / "ipo_optimizer_attention_layer0.png",
                        title="Layer 0 self-attention (mean over batch)",
                    )
                print(
                    f"[IPO] Saved attention to {ap} and "
                    f"{ROOT / 'figures' / 'ipo_optimizer_attention_layer0.png'}",
                    flush=True,
                )

    if use_sectors:
        weights = predict_sector_head_weights(model, data["X_val"], device)
        export_sector_group_outputs(
            data["dates_val"], weights, data["R_val"], sector_labels, out_dir
        )
        print(f"Exported per-sector weights + {out_dir / 'ipo_optimizer_summary_by_sector.txt'}")
        avg_ipo = float(np.mean(weights[:, :, 1])) if weights.ndim == 3 else 0.0
        scale = ipo_tilt_to_position_scale(avg_ipo)
        print(policy_rule(avg_ipo))
        print(f"Suggested position scale (avg across sector IPO sleeves): {scale:.2f}")
        g = weights.shape[1]
        for idx in range(g):
            st = portfolio_stats(weights[:, idx, :], data["R_val"][:, idx, :])
            print(
                f"  [{sector_labels[idx]}] Sharpe={st['sharpe_annualized']:.2f}  "
                f"MaxDD={st['max_drawdown']:.2%}"
            )
    else:
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


if __name__ == "__main__":
    raise SystemExit(main())
