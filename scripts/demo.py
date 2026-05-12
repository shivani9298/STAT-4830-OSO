from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

# Short narrative: what each exported policy is meant to represent (shown above the charts).
POLICY_INTERPRETATION_FULL_OFFLINE = (
    "**What you’re looking at (offline, full training goal).** "
    "The model was trained to balance **several things at once**—not just long-run growth, but also "
    "keeping risk, trading, and diversification in check. Each day it decides how to split money between "
    "broad market and the IPO slice. When the model line **separates** from the 50/50 line or from "
    "putting 100% in one asset, that gap is the trade-off it learned: **more cautious or more tilted** "
    "than a simple blend, depending on recent history—not “growth at any cost.”"
)

POLICY_INTERPRETATION_FULL_ONLINE = (
    "**What you’re looking at (online, full training goal).** "
    "This is the **live-style** run: the model gets **re-trained from time to time** as new data "
    "come in, instead of using one frozen brain forever. The return chart is what you actually would "
    "have earned along that path (after costs). The weights chart shows how the split between market "
    "and IPO **drifts** over the window you chose."
)

POLICY_INTERPRETATION_COMPOUND_OVERLAY = (
    "**What the two curves mean (compound-growth training only).** "
    "One line is **train once, then roll forward**: same trained model every day; only the **inputs** "
    "(recent history) change, so weights move as the world moves. The other line is **train again "
    "regularly**: same *kind* of goal (compound growth), but the model is **updated on a schedule**. "
    "When those two lines sit at different heights or move differently, you’re seeing **one brain vs "
    "a brain that keeps studying**—not “static vs dynamic,” because both lines move; only the second "
    "one refreshes how decisions are made."
)

POLICY_INTERPRETATION_COMPOUND_OFFLINE = (
    "**What you’re looking at (offline, compound-growth training only).** "
    "Here the model was trained with **growth-style rewards turned up** and the extra “play it safe” "
    "knobs turned down. So it’s closer to “help me compound” without the full recipe that pulled the "
    "other panel toward balance. It still picks weights **every day** from what it sees in the data, "
    "but you should expect **bolder** splits vs an even 50/50 when the model thinks that’s how to grow "
    "wealth fastest over the stretch you’re viewing."
)

POLICY_INTERPRETATION_COMPOUND_ONLINE = (
    "**What you’re looking at (online, compound-growth training only).** "
    "Same growth-focused training idea as above, but run **in production style**: the model is **updated "
    "on a schedule**, so two things move the needle—today’s data **and** occasional fresh training. "
    "If the online IPO weight **keeps a different level or pattern** than the offline line above, that "
    "usually means **re-learning from newer history**, not just “another day with the same old model.”"
)

POLICY_INTERPRETATION_BONDS = (
    "**What you’re looking at (bond mix).** "
    "The model chooses how much to hold in the **broad market** piece versus the **bond** piece each "
    "day. When the bond side’s share goes **up**, it’s leaning toward bond-like exposure for that "
    "period; when it goes **down**, it’s leaning more equity-like. Compare the model to **equal split** "
    "and to **each side alone** to see whether those shifts helped once you account for how concentrated "
    "the portfolio was."
)

POLICY_INTERPRETATION_COMMODITIES = (
    "**What you’re looking at (commodity mix).** "
    "The model mixes **broad market** with a **commodity** bucket. The weight line tells you how "
    "strongly it favors commodities versus stocks **right now**, given what it has seen in the window "
    "you chose. Stack that against the 50/50 blend and the single-asset lines: you’re checking whether "
    "leaning into commodities **paid off** for the risk of **not** staying evenly spread."
)

# Prefer a dedicated full-objective WRDS export; fall back to the cadence online run
# (full multi-term objective, AB update gate experiment, 2021-05-18 → 2024-12-31).
ONLINE_FULL_OBJECTIVE_PATH_CANDIDATES: list[str] = [
    "results/ipo_optimizer_online_path_full_objective.csv",   # reserved: place a new full-obj run here to override
    "results/ipo_optimizer_online_wrds_full_loss.csv",        # reserved: alternate override path
    "online_training_work/results/online_path_cadence_lb504.csv",   # cadence-gated online model (primary fallback)
    "online_training_work/results/ipo_optimizer_online_path.csv",   # confidence online model (secondary fallback)
    "results/older/c0d127c/ipo_optimizer_online_path.csv",          # archived fallback
]

# Compound-growth-only loss: prefer a dedicated export so a later full-objective WRDS run
# does not silently replace the online path this tab is meant to show.
ONLINE_COMPOUND_ONLY_PATH_CANDIDATES: list[str] = [
    "results/ipo_optimizer_online_path_compound_only.csv",
    "results/ipo_optimizer_online_path.csv",
]


@dataclass
class AssetPanel:
    name: str
    frame: pd.DataFrame
    source: str
    asset_label: str
    model_label: str = "Model"
    model_b_label: Optional[str] = None
    notes: str = ""


def resolve_artifact(rel_path: str) -> Optional[Path]:
    rel = Path(rel_path)
    direct = ROOT / rel
    if direct.exists():
        return direct
    name = rel.name
    search_roots = [ROOT, ROOT / "online_training_work"]
    candidates: list[Path] = []
    for sr in search_roots:
        if sr.exists():
            candidates.extend([p for p in sr.rglob(name) if p.is_file()])
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_csv_resolved(rel_path: str, **kwargs) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    p = resolve_artifact(rel_path)
    if p is None:
        return None, None
    return pd.read_csv(p, **kwargs), p


def resolve_first_existing(rel_paths: list[str]) -> Optional[Path]:
    for rel in rel_paths:
        p = resolve_artifact(rel)
        if p is not None:
            return p
    return None


def annualized_stats(returns: np.ndarray) -> dict[str, float]:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return {"total": 0.0, "ann": 0.0, "vol": 0.0, "sharpe": 0.0, "mdd": 0.0}
    wealth = np.cumprod(1.0 + arr)
    total = wealth[-1] - 1.0
    ann = (1.0 + total) ** (252.0 / max(arr.size, 1)) - 1.0
    vol = float(np.std(arr) * np.sqrt(252.0))
    sharpe = float(ann / vol) if vol > 1e-12 else 0.0
    dd = wealth / np.maximum.accumulate(wealth) - 1.0
    return {"total": 100 * float(total), "ann": 100 * float(ann), "vol": 100 * vol, "sharpe": sharpe, "mdd": 100 * float(dd.min())}


def cum_pct(arr: np.ndarray) -> np.ndarray:
    return (np.cumprod(1.0 + np.asarray(arr, dtype=float)) - 1.0) * 100.0


def _coerce_plotly_x_to_timestamp(x: Any) -> pd.Timestamp:
    ts = pd.to_datetime(x, utc=False)
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0]
    return pd.Timestamp(ts).normalize()


def _asof_from_plotly_event(event: Any) -> Optional[pd.Timestamp]:
    if event is None:
        return None
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")
    if not sel:
        return None
    pts = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    if not pts:
        return None
    return _coerce_plotly_x_to_timestamp(pts[0].get("x"))


def _slice_through_asof(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    d = pd.to_datetime(df["date"])
    asof_ts = pd.Timestamp(asof).normalize()
    mask = d <= asof_ts
    if not bool(mask.any()):
        return df.iloc[:0].copy()
    return df.loc[mask].reset_index(drop=True)


def _prep_ipo_from_weights_and_returns(
    weight_rel_path: str,
    return_candidates: list[str],
    *,
    model_label: str,
    notes: str = "",
) -> Optional[AssetPanel]:
    w_df, w_src = read_csv_resolved(weight_rel_path, parse_dates=["date"])
    if w_df is None or w_src is None:
        return None
    r_df = None
    r_src: Optional[Path] = None
    for c in return_candidates:
        r_df, r_src = read_csv_resolved(c, parse_dates=["date"])
        if r_df is not None:
            break
    if r_df is None or r_src is None:
        return None
    mw = next((c for c in ["weight_market", "market_weight"] if c in w_df.columns), None)
    iw = next((c for c in ["weight_IPO", "weight_ipo", "weight_asset"] if c in w_df.columns), None)
    mr = next((c for c in ["market_return", "realized_market", "SPY_Only"] if c in r_df.columns), None)
    ir = next((c for c in ["ipo_return", "realized_ipo", "IPO_Only"] if c in r_df.columns), None)
    if not all([mw, iw, mr, ir]):
        return None
    m = (
        w_df[["date", mw, iw]]
        .rename(columns={mw: "weight_market", iw: "weight_asset"})
        .merge(
            r_df[["date", mr, ir]].rename(columns={mr: "market_only", ir: "asset_only"}),
            on="date",
            how="inner",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    if m.empty:
        return None
    model_ret = m["weight_market"].astype(float) * m["market_only"].astype(float) + m["weight_asset"].astype(float) * m["asset_only"].astype(float)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(m["date"]),
            "model_ret": model_ret,
            "model_ret_b": np.nan,
            "benchmark_5050": 0.5 * m["market_only"].astype(float) + 0.5 * m["asset_only"].astype(float),
            "market_only": m["market_only"].astype(float),
            "asset_only": m["asset_only"].astype(float),
            "weight_market": m["weight_market"].astype(float),
            "weight_asset": m["weight_asset"].astype(float),
            "weight_market_b": np.nan,
            "weight_asset_b": np.nan,
            "gate": np.nan,
        }
    )
    return AssetPanel(
        "IPO",
        out,
        f"{w_src} + {r_src}",
        "IPO",
        model_label=model_label,
        notes=notes,
    )


def prep_online_panel_from_ipo_path_csv(path: Path, model_label: str, notes: str) -> Optional[AssetPanel]:
    df = pd.read_csv(path, parse_dates=["date"])
    req = {"weight_market", "weight_ipo", "realized_market", "realized_ipo", "net_ret"}
    if not req.issubset(df.columns):
        return None
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "model_ret": df["net_ret"].astype(float),
            "model_ret_b": np.nan,
            "benchmark_5050": 0.5 * df["realized_market"].astype(float) + 0.5 * df["realized_ipo"].astype(float),
            "market_only": df["realized_market"].astype(float),
            "asset_only": df["realized_ipo"].astype(float),
            "weight_market": df["weight_market"].astype(float),
            "weight_asset": df["weight_ipo"].astype(float),
            "weight_market_b": np.nan,
            "weight_asset_b": np.nan,
            "gate": np.nan,
        }
    ).sort_values("date")
    return AssetPanel("IPO", out, str(path), "IPO", model_label=model_label, notes=notes)


def prep_offline_full_objective() -> Optional[AssetPanel]:
    ret_candidates = ["results/recent/ipo_optimizer_returns_val.csv", "results/older/ipo_180day_mcap_returns.csv"]
    return _prep_ipo_from_weights_and_returns(
        "results/recent/ipo_optimizer_weights_val_sector_mean.csv",
        ret_candidates,
        model_label="Offline GRU (mean across 11 sector sleeves)",
        notes=(
            "Weights are the mean across all 11 sector sleeve heads "
            "(`results/recent/ipo_optimizer_weights_val_sector_mean.csv`). "
            "Returns use the **aggregate IPO index** as a proxy for each sector basket — "
            "the report/slides use the actual per-sector IPO baskets computed during training "
            "(not stored separately as a single CSV), so the model total here (~48%) is a close "
            "approximation of the slide value (49.25%).\n\n"
            "**Why the annualized return and Sharpe look different from the report:**\n"
            "The report quotes 45.12% ann and Sharpe 2.30. Those figures were computed during "
            "training on the per-sector baskets over a specific validation window. The CSV artifact "
            "(`ipo_optimizer_returns_val.csv`) covers the full 2022–2024 period (~2.93 years), and "
            "the demo annualizes the blended model return (≈88% market / 12% IPO) over that longer "
            "window — producing a lower annualized figure. The total-return gap (~49% vs ~48%) comes "
            "from the per-sector vs aggregate IPO proxy; the ann/Sharpe gap comes from the different "
            "date window used to compute those figures during training. Neither is an error — they are "
            "different views of the same model evaluated at different aggregation levels."
        ),
    )


def prep_online_full_objective() -> Optional[AssetPanel]:
    p = resolve_first_existing(ONLINE_FULL_OBJECTIVE_PATH_CANDIDATES)
    if p is None:
        return None
    notes = (
        f"**Source file in use:** `{p}`\n\n"
        "Online path uses `net_ret` from the WRDS-style online CSV. "
        "To override with a new full-objective run, save it to "
        "`results/ipo_optimizer_online_path_full_objective.csv`."
    )
    return prep_online_panel_from_ipo_path_csv(p, model_label="Online (full objective)", notes=notes)


def prep_compound_offline_panel() -> Optional[AssetPanel]:
    cmp_df, cmp_src = read_csv_resolved("results/online_compound_vs_offline_compound_same_dates.csv", parse_dates=["date"])
    if cmp_df is None or cmp_src is None:
        return None
    c = cmp_df.sort_values("date").reset_index(drop=True)
    wm = 1.0 - c["weight_ipo_offline_compound_static"].astype(float)
    wa = c["weight_ipo_offline_compound_static"].astype(float)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(c["date"]),
            "model_ret": c["ret_offline_compound_static"].astype(float),
            "model_ret_b": np.nan,
            "benchmark_5050": c["ret_5050"].astype(float),
            "market_only": c["ret_market_only"].astype(float),
            "asset_only": c["ret_ipo_only"].astype(float),
            "weight_market": wm,
            "weight_asset": wa,
            "weight_market_b": np.nan,
            "weight_asset_b": np.nan,
            "gate": np.nan,
        }
    )
    return AssetPanel(
        "IPO",
        out,
        str(cmp_src),
        "IPO",
        model_label="Offline (compound-growth loss)",
        notes=(
            "Offline comparator: one compound-loss model trained on initial history only; "
            "weights still vary by day with rolling-window features (no periodic retraining). "
            "Series from `plot_online_compound_vs_offline_compound.py`."
        ),
    )


def prep_compound_online_panel() -> Optional[AssetPanel]:
    cmp_df, cmp_src = read_csv_resolved("results/online_compound_vs_offline_compound_same_dates.csv", parse_dates=["date"])
    if cmp_df is None or cmp_src is None:
        return None
    c = cmp_df.sort_values("date").reset_index(drop=True)

    on_path = resolve_first_existing(ONLINE_COMPOUND_ONLY_PATH_CANDIDATES)
    on_df: Optional[pd.DataFrame] = None
    if on_path is not None:
        on_df = pd.read_csv(on_path, parse_dates=["date"])

    wa_cmp = c["weight_ipo_online_compound"].astype(float)
    wa = wa_cmp
    if on_df is not None and "weight_ipo" in on_df.columns:
        path_m = on_df[["date", "weight_ipo"]].copy()
        path_m["date"] = pd.to_datetime(path_m["date"])
        path_m = path_m.rename(columns={"weight_ipo": "weight_ipo_wrds_path"})
        c = c.merge(path_m, on="date", how="left")
        wa_path = c["weight_ipo_wrds_path"].astype(float)
        wa = wa_path.where(wa_path.notna(), wa_cmp)

    wm = 1.0 - wa
    src = str(cmp_src) if on_path is None else f"{cmp_src} + {on_path}"
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(c["date"]),
            "model_ret": c["ret_online_compound"].astype(float),
            "model_ret_b": np.nan,
            "benchmark_5050": c["ret_5050"].astype(float),
            "market_only": c["ret_market_only"].astype(float),
            "asset_only": c["ret_ipo_only"].astype(float),
            "weight_market": wm,
            "weight_asset": wa,
            "weight_market_b": np.nan,
            "weight_asset_b": np.nan,
            "gate": np.nan,
        }
    )
    return AssetPanel(
        "IPO",
        out,
        src,
        "IPO",
        model_label="Online (compound-growth loss)",
        notes=(
            "Online WRDS compound-loss run: `weight_ipo` comes from the resolved compound-only path export "
            "(see `ONLINE_COMPOUND_ONLY_PATH_CANDIDATES` in `demo.py`). "
            "Returns/benchmarks stay aligned to `online_compound_vs_offline_compound_same_dates.csv`."
        ),
    )


def prep_bond_data() -> Optional[AssetPanel]:
    df, src = read_csv_resolved("results/bonds/bonbond_optimizer_returns.csv", parse_dates=["date"])
    if df is None or src is None:
        return None
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "model_ret": df["Model_Portfolio"].astype(float),
            "model_ret_b": np.nan,
            "benchmark_5050": df["Equal_Weight"].astype(float),
            "market_only": df["Market_Only"].astype(float),
            "asset_only": df["Bond_Only"].astype(float),
            "weight_market": df["Market_Weight"].astype(float),
            "weight_asset": df["Bond_Weight"].astype(float),
            "weight_market_b": np.nan,
            "weight_asset_b": np.nan,
            "gate": np.nan,
        }
    ).sort_values("date")
    return AssetPanel("Bonds", out, str(src), "Bond")


def prep_commodity_data() -> Optional[AssetPanel]:
    w, w_src = read_csv_resolved("results/commodity_optimizer_weights.csv", parse_dates=["date"])
    r, r_src = read_csv_resolved("results/commodity_optimizer_returns_val.csv", parse_dates=["date"])
    if w is None or r is None or w_src is None or r_src is None:
        return None
    w = w.rename(columns={"weight_commodities": "weight_asset"})
    m = (
        w[["date", "weight_market", "weight_asset"]]
        .merge(r[["date", "market_return", "commodity_return", "equal_weight_return"]], on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    model_ret = m["weight_market"].astype(float) * m["market_return"].astype(float) + m["weight_asset"].astype(float) * m["commodity_return"].astype(float)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(m["date"]),
            "model_ret": model_ret,
            "model_ret_b": np.nan,
            "benchmark_5050": m["equal_weight_return"].astype(float),
            "market_only": m["market_return"].astype(float),
            "asset_only": m["commodity_return"].astype(float),
            "weight_market": m["weight_market"].astype(float),
            "weight_asset": m["weight_asset"].astype(float),
            "weight_market_b": np.nan,
            "weight_asset_b": np.nan,
            "gate": np.nan,
        }
    )
    return AssetPanel("Commodities", out, f"{w_src} + {r_src}", "Commodity")


def apply_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    x = df.copy()
    if scenario == "Asset Rally":
        x["asset_only"] *= 1.25
        x["market_only"] *= 0.90
    elif scenario == "Risk-Off":
        x["asset_only"] -= 0.004
        x["market_only"] -= 0.002
    x["benchmark_5050"] = 0.5 * x["market_only"] + 0.5 * x["asset_only"]
    x["model_ret"] = x["weight_market"] * x["market_only"] + x["weight_asset"] * x["asset_only"]
    if x["model_ret_b"].notna().any():
        x["model_ret_b"] = x["weight_market_b"] * x["market_only"] + x["weight_asset_b"] * x["asset_only"]
    return x


def render_performance_allocation_block(
    panel: AssetPanel,
    scenario: str,
    window: int,
    chart_key: str,
    *,
    interpretation: str = "",
) -> None:
    df0 = apply_scenario(panel.frame, scenario)
    dfw = df0.iloc[-min(window, len(df0)) :].reset_index(drop=True)
    if interpretation.strip():
        st.markdown(interpretation)
    if dfw.empty:
        st.warning("No rows in the selected window.")
        return

    first_dt = pd.Timestamp(dfw["date"].iloc[0]).normalize()
    last_dt = pd.Timestamp(dfw["date"].iloc[-1]).normalize()
    asof_key = f"{chart_key}_asof"
    if asof_key not in st.session_state:
        st.session_state[asof_key] = last_dt
    prev_asof = pd.Timestamp(st.session_state[asof_key]).normalize()
    if prev_asof < first_dt or prev_asof > last_dt:
        st.session_state[asof_key] = last_dt

    perf_fig = go.Figure()
    perf_fig.add_trace(go.Scatter(x=dfw["date"], y=cum_pct(dfw["model_ret"].to_numpy(float)), mode="lines", name=panel.model_label))
    model_b = dfw["model_ret_b"].to_numpy(float) if dfw["model_ret_b"].notna().any() else None
    if model_b is not None:
        perf_fig.add_trace(go.Scatter(x=dfw["date"], y=cum_pct(model_b), mode="lines", name=panel.model_b_label or "Comparator"))
    perf_fig.add_trace(go.Scatter(x=dfw["date"], y=cum_pct(dfw["benchmark_5050"].to_numpy(float)), mode="lines", name="50/50", line={"dash": "dash"}))
    perf_fig.add_trace(go.Scatter(x=dfw["date"], y=cum_pct(dfw["market_only"].to_numpy(float)), mode="lines", name="Market-only", line={"dash": "dot"}))
    perf_fig.add_trace(
        go.Scatter(
            x=dfw["date"],
            y=cum_pct(dfw["asset_only"].to_numpy(float)),
            mode="lines",
            name=f"{panel.asset_label}-only",
            line={"dash": "dashdot"},
        )
    )
    perf_fig.update_layout(
        title=f"{panel.name}: cumulative return (click a point to set metrics as-of date)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        height=420,
    )

    event = st.plotly_chart(perf_fig, use_container_width=True, key=chart_key, on_select="rerun", selection_mode=("points",))

    clicked = _asof_from_plotly_event(event)
    if clicked is not None:
        st.session_state[asof_key] = max(first_dt, min(pd.Timestamp(clicked).normalize(), last_dt))

    st.caption(
        f"Metrics below use returns **through** the as-of date (last day in window: **{last_dt.date()}**). "
        "Click the cumulative return chart or use the date picker."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        cur = pd.Timestamp(st.session_state[asof_key]).normalize()
        manual = st.date_input(
            "As-of date (metrics through this day)",
            value=cur.date(),
            min_value=first_dt.date(),
            max_value=last_dt.date(),
            key=f"{chart_key}_manual",
        )
        manual_ts = pd.Timestamp(manual).normalize()
        if manual_ts != cur:
            st.session_state[asof_key] = manual_ts
    with c2:
        if st.button("Reset metrics to latest in window", key=f"{chart_key}_latest"):
            st.session_state[asof_key] = last_dt
            st.rerun()

    asof = pd.Timestamp(st.session_state[asof_key]).normalize()
    st.caption(f"Current as-of: **{asof.date()}**.")

    df_metrics = _slice_through_asof(dfw, asof)
    if df_metrics.empty:
        st.warning("As-of date is before the first date in the window; widen the window or pick a later date.")
        return

    model = df_metrics["model_ret"].to_numpy(float)
    model_b_arr = df_metrics["model_ret_b"].to_numpy(float) if df_metrics["model_ret_b"].notna().any() else None
    eq = df_metrics["benchmark_5050"].to_numpy(float)
    market = df_metrics["market_only"].to_numpy(float)
    asset = df_metrics["asset_only"].to_numpy(float)

    rows = [[panel.model_label, *annualized_stats(model).values()]]
    if model_b_arr is not None:
        rows.append([panel.model_b_label or "Comparator", *annualized_stats(model_b_arr).values()])
    rows.append(["50/50", *annualized_stats(eq).values()])
    rows.append(["Market-only", *annualized_stats(market).values()])
    rows.append([f"{panel.asset_label}-only", *annualized_stats(asset).values()])
    st.dataframe(
        pd.DataFrame(rows, columns=["Strategy", "Total %", "Ann %", "Vol %", "Sharpe", "Max DD %"]).round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Allocation dynamics")
    w_fig = go.Figure()
    w_fig.add_trace(go.Scatter(x=dfw["date"], y=dfw["weight_market"], mode="lines", name="weight_market"))
    w_fig.add_trace(go.Scatter(x=dfw["date"], y=dfw["weight_asset"], mode="lines", name=f"weight_{panel.asset_label.lower()}"))
    if dfw["weight_market_b"].notna().any():
        w_fig.add_trace(go.Scatter(x=dfw["date"], y=dfw["weight_market_b"], mode="lines", name="weight_market (comparator)"))
        w_fig.add_trace(go.Scatter(x=dfw["date"], y=dfw["weight_asset_b"], mode="lines", name=f"weight_{panel.asset_label.lower()} (comparator)"))
    w_fig.update_layout(title=f"{panel.name}: dynamic weights", yaxis_range=[0, 1], height=360)
    st.plotly_chart(w_fig, use_container_width=True)
    st.caption(f"{panel.asset_label} weight range in window: {dfw['weight_asset'].min():.6f} to {dfw['weight_asset'].max():.6f}")

    with st.expander("Notes & data sources"):
        st.write(panel.notes or "No additional notes.")
        st.code(panel.source)


def windowed_panel_df(panel: AssetPanel, scenario: str, window: int) -> pd.DataFrame:
    df0 = apply_scenario(panel.frame, scenario)
    return df0.iloc[-min(window, len(df0)) :].reset_index(drop=True)


def render_compound_allocation_overlay(compound_off: AssetPanel, compound_on: AssetPanel, scenario: str, window: int) -> None:
    of = windowed_panel_df(compound_off, scenario, window)
    on = windowed_panel_df(compound_on, scenario, window)
    if of.empty or on.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=of["date"], y=of["weight_asset"], mode="lines", name="Offline comparator — IPO weight", line={"width": 2.2})
    )
    fig.add_trace(go.Scatter(x=on["date"], y=on["weight_asset"], mode="lines", name="Online WRDS — IPO weight", line={"width": 2.2}))
    fig.update_layout(
        title="IPO weight — offline comparator vs online WRDS (shared 0–1 axis)",
        xaxis_title="Date",
        yaxis_title="IPO allocation",
        height=420,
        yaxis_range=[0, 1],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Per-section charts below used to auto-scale the y-axis to each series’ local range, "
        "which makes different allocation levels look equally “active.” This overlay locks [0, 1] so you can see the true gap."
    )


def main() -> None:
    st.set_page_config(page_title="STAT 4830 Control Room", layout="wide")
    st.title("STAT 4830 Multi-Asset Control Room")

    offline_full = prep_offline_full_objective()
    online_full = prep_online_full_objective()
    compound_off = prep_compound_offline_panel()
    compound_on = prep_compound_online_panel()
    bond_panel = prep_bond_data()
    commodity_panel = prep_commodity_data()

    tab_full, tab_compound, tab_bonds, tab_cmdty = st.tabs(
        ["IPO — full objective", "IPO — compound growth only", "Bonds", "Commodities"]
    )

    with st.sidebar:
        st.header("Controls")
        scenario = st.selectbox("Scenario", ["Historical", "Asset Rally", "Risk-Off"], index=0)
        window = st.slider("Window (trading days)", min_value=30, max_value=504, value=180, step=5)

    with tab_full:
        st.markdown("Compare **offline** and **online** IPO models trained with the **full** objective. Scroll for performance, interactive metrics, then allocation.")
        if offline_full is None:
            st.error("Offline full-objective artifacts missing (`results/recent/ipo_optimizer_weights_val_sector_mean.csv` + returns).")
        else:
            st.subheader("Offline — full objective")
            render_performance_allocation_block(
                offline_full, scenario, window, chart_key="ipo_full_offline", interpretation=POLICY_INTERPRETATION_FULL_OFFLINE
            )
        st.divider()
        if online_full is None:
            st.error(
                "Online full-objective path not found. Add `results/ipo_optimizer_online_path_full_objective.csv` "
                "or keep a fallback under `results/older/...`."
            )
        else:
            used = resolve_first_existing(ONLINE_FULL_OBJECTIVE_PATH_CANDIDATES)
            st.caption(f"Online full-objective CSV in use: `{used}`")
            st.subheader("Online — full objective")
            render_performance_allocation_block(
                online_full, scenario, window, chart_key="ipo_full_online", interpretation=POLICY_INTERPRETATION_FULL_ONLINE
            )

    with tab_compound:
        st.markdown(
            "**Compound-growth loss only**: offline comparator (single trained model, **no periodic refits**) vs "
            "**online WRDS** path with periodic updates. Both allocation paths vary day-to-day with features; "
            "they are not the same series."
        )
        if compound_off is None or compound_on is None:
            st.error("Missing `results/online_compound_vs_offline_compound_same_dates.csv`.")
        else:
            used_compound = resolve_first_existing(ONLINE_COMPOUND_ONLY_PATH_CANDIDATES)
            if used_compound is not None:
                st.caption(
                    f"**Compound-loss WRDS online path in use:** `{used_compound}`. "
                    "Save a compound-only run as `results/ipo_optimizer_online_path_compound_only.csv` "
                    "so it is not overwritten when you run full-objective online training."
                )
            else:
                st.warning("No online path CSV found; compound online weights use the merged comparison file only.")
            st.subheader("IPO weight — quick comparison (read this first)")
            st.markdown(POLICY_INTERPRETATION_COMPOUND_OVERLAY)
            render_compound_allocation_overlay(compound_off, compound_on, scenario, window)
            st.divider()
            st.subheader("Offline — compound-growth loss")
            render_performance_allocation_block(
                compound_off, scenario, window, chart_key="ipo_cmp_offline", interpretation=POLICY_INTERPRETATION_COMPOUND_OFFLINE
            )
            st.divider()
            st.subheader("Online — compound-growth loss")
            render_performance_allocation_block(
                compound_on, scenario, window, chart_key="ipo_cmp_online", interpretation=POLICY_INTERPRETATION_COMPOUND_ONLINE
            )

    with tab_bonds:
        if bond_panel is None:
            st.error("Bond panel unavailable (`results/bonds/bonbond_optimizer_returns.csv`).")
        else:
            render_performance_allocation_block(
                bond_panel, scenario, window, chart_key="bonds_panel", interpretation=POLICY_INTERPRETATION_BONDS
            )

    with tab_cmdty:
        if commodity_panel is None:
            st.error("Commodity panel unavailable (weights + returns CSVs).")
        else:
            render_performance_allocation_block(
                commodity_panel, scenario, window, chart_key="cmdty_panel", interpretation=POLICY_INTERPRETATION_COMMODITIES
            )


if __name__ == "__main__":
    main()
