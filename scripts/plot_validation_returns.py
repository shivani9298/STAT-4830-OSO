#!/usr/bin/env python3
"""
Plot cumulative returns over the validation period for the IPO optimizer run
associated with ipo_optimizer_loss_semilog.png.

Validation period: 2024-01-31 to 2024-12-31 (from ipo_optimizer_weights.csv).
Asset returns sourced from ipo_180day_mcap_returns.csv (SPY_Only, IPO_Only columns).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
results = ROOT / "results"
figures = ROOT / "figures" / "old diagrams"
figures.mkdir(parents=True, exist_ok=True)

# --- Load data ---
weights = pd.read_csv(results / "ipo_optimizer_weights.csv", index_col="date", parse_dates=True)
returns = pd.read_csv(results / "ipo_180day_mcap_returns.csv", index_col=0, parse_dates=True)

# Align to validation period dates (use weights dates as the reference)
ret_val = returns[["SPY_Only", "IPO_Only"]].reindex(weights.index).dropna()
wts_val = weights.reindex(ret_val.index)

# --- Compute daily returns ---
model_daily  = (wts_val["weight_market"] * ret_val["SPY_Only"]
              + wts_val["weight_IPO"]    * ret_val["IPO_Only"])
market_daily = ret_val["SPY_Only"]
ipo_daily    = ret_val["IPO_Only"]
equal_daily  = 0.5 * ret_val["SPY_Only"] + 0.5 * ret_val["IPO_Only"]

# --- Cumulative returns (starting at 0%) ---
def cum_ret(daily):
    return (1 + daily).cumprod() - 1

model_cum  = cum_ret(model_daily)
market_cum = cum_ret(market_daily)
ipo_cum    = cum_ret(ipo_daily)
equal_cum  = cum_ret(equal_daily)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(model_cum.index,  model_cum  * 100, label="Model (GRU optimizer)", linewidth=2, color="#1f77b4")
ax.plot(equal_cum.index,  equal_cum  * 100, label="Equal 50/50",           linewidth=1.5, color="#2ca02c", linestyle="--")
ax.plot(market_cum.index, market_cum * 100, label="Market only (SPY/DIA)", linewidth=1.5, color="#ff7f0e", linestyle="--")
ax.plot(ipo_cum.index,    ipo_cum    * 100, label="IPO index only",        linewidth=1.5, color="#9467bd", linestyle="--")

ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
fig.autofmt_xdate()

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (%)")
ax.set_title("IPO Optimizer — Cumulative Returns over Validation Period\n(2024-01-31 to 2024-12-31)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# Annotate final values
for series, label, color in [
    (model_cum,  "Model",   "#1f77b4"),
    (equal_cum,  "50/50",   "#2ca02c"),
    (market_cum, "Market",  "#ff7f0e"),
    (ipo_cum,    "IPO idx", "#9467bd"),
]:
    ax.annotate(
        f"{series.iloc[-1]*100:.1f}%",
        xy=(series.index[-1], series.iloc[-1] * 100),
        xytext=(5, 0), textcoords="offset points",
        va="center", fontsize=9, color=color,
    )

fig.tight_layout()
out = figures / "validation_cumulative_returns.png"
fig.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# Print summary
print(f"\nValidation period: {ret_val.index[0].date()} to {ret_val.index[-1].date()}  ({len(ret_val)} days)")
print(f"  Model:   {model_cum.iloc[-1]*100:.2f}%")
print(f"  Equal:   {equal_cum.iloc[-1]*100:.2f}%")
print(f"  Market:  {market_cum.iloc[-1]*100:.2f}%")
print(f"  IPO idx: {ipo_cum.iloc[-1]*100:.2f}%")
