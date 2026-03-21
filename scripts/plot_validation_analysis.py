#!/usr/bin/env python3
"""
Three plots for the validation period of the run_ipo_optimizer_wrds run:
  1. Cumulative returns (model vs baselines)
  2. Rolling optimization objective value
  3. Rolling portfolio variance
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent

# ── load data ──────────────────────────────────────────────────────────────────
weights = pd.read_csv(
    ROOT / "results" / "ipo_optimizer_weights.csv",
    index_col="date", parse_dates=True,
)
mcap = pd.read_csv(
    ROOT / "results" / "ipo_180day_mcap_returns.csv",
    index_col=0, parse_dates=True,
)

# align to the validation period dates
ret = mcap[["SPY_Only", "IPO_Only"]].reindex(weights.index).dropna()
wts = weights.reindex(ret.index)

w_m = wts["weight_market"].values   # (N,)
w_i = wts["weight_IPO"].values
r_m = ret["SPY_Only"].values
r_i = ret["IPO_Only"].values
dates = ret.index

# ── daily portfolio returns ────────────────────────────────────────────────────
model_ret  = w_m * r_m + w_i * r_i
market_ret = r_m
ipo_ret    = r_i
equal_ret  = 0.5 * r_m + 0.5 * r_i

# ── PLOT 1: cumulative returns ─────────────────────────────────────────────────
cum = lambda r: (1 + r).cumprod() - 1

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(dates, cum(model_ret)  * 100, lw=2,   color="#1f77b4", label="Model (GRU optimizer)")
ax1.plot(dates, cum(equal_ret)  * 100, lw=1.5, color="#2ca02c", ls="--", label="Equal 50/50")
ax1.plot(dates, cum(market_ret) * 100, lw=1.5, color="#ff7f0e", ls="--", label="Market only")
ax1.plot(dates, cum(ipo_ret)    * 100, lw=1.5, color="#9467bd", ls="--", label="IPO index only")
ax1.axhline(0, color="k", lw=0.7, ls=":")

for r, label, color in [
    (model_ret, "Model", "#1f77b4"),
    (equal_ret, "50/50", "#2ca02c"),
    (market_ret, "Market", "#ff7f0e"),
    (ipo_ret, "IPO", "#9467bd"),
]:
    final = cum(r)[-1] * 100
    ax1.annotate(f"{final:.1f}%", xy=(dates[-1], final),
                 xytext=(4, 0), textcoords="offset points",
                 va="center", fontsize=9, color=color)

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
fig1.autofmt_xdate()
ax1.set_ylabel("Cumulative Return (%)")
ax1.set_title("Cumulative Returns — Validation Period (2024-01-31 to 2024-12-31)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(ROOT / "figures" / "validation_cumulative_returns.png", dpi=150)
plt.close(fig1)
print("Saved: figures/validation_cumulative_returns.png")

# ── PLOT 2: rolling optimization objective ────────────────────────────────────
# Best config lambdas
lam_vol       = 1.0
lam_cvar      = 1.0
lam_turnover  = 0.0025
lam_path      = 0.01
lam_vol_exc   = 0.5
target_vol    = 0.25
alpha         = 0.05
WINDOW        = 21   # ~1 month rolling

def rolling_objective(model_r, wts_m, wts_i, win=21):
    """Rolling combined objective over `win` days (numpy version of combined_loss)."""
    N = len(model_r)
    obj  = np.full(N, np.nan)
    L_mean_comp  = np.full(N, np.nan)
    L_vol_comp   = np.full(N, np.nan)
    L_cvar_comp  = np.full(N, np.nan)
    L_vexc_comp  = np.full(N, np.nan)
    for t in range(win - 1, N):
        r  = model_r[t - win + 1 : t + 1]
        wm = wts_m[t - win + 1 : t + 1]
        wi = wts_i[t - win + 1 : t + 1]
        mean_r = r.mean()
        var_r  = r.var()
        std_r  = r.std()
        ann_vol = std_r * np.sqrt(252)
        # CVaR: expected value of worst alpha fraction
        k = max(1, int(alpha * win))
        sorted_r = np.sort(r)
        cvar_val = sorted_r[:k].mean()
        # turnover
        dw = np.abs(np.diff(np.stack([wm, wi], axis=1), axis=0)).sum(axis=1).mean()
        # path (weight instability)
        path = ((np.diff(np.stack([wm, wi], axis=1), axis=0))**2).sum(axis=1).mean()
        # vol excess
        vol_exc = max(0.0, ann_vol - target_vol)

        L_mean_comp[t]  = mean_r
        L_vol_comp[t]   = var_r
        L_cvar_comp[t]  = cvar_val
        L_vexc_comp[t]  = vol_exc
        L = (-mean_r
             + lam_vol * var_r
             + lam_cvar * (-cvar_val)
             + lam_turnover * dw
             + lam_path * path
             + lam_vol_exc * vol_exc)
        obj[t] = L
    return obj, L_mean_comp, L_vol_comp, L_cvar_comp, L_vexc_comp

obj, comp_mean, comp_vol, comp_cvar, comp_vexc = rolling_objective(
    model_ret, w_m, w_i, win=WINDOW
)

fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax = axes2[0]
ax.plot(dates, obj, color="#d62728", lw=1.5, label="Combined objective L")
ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_ylabel("Objective value (lower = better)")
ax.set_title(f"Rolling Optimization Objective ({WINDOW}-day window) — Validation Period")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.plot(dates, comp_mean, color="#1f77b4", lw=1.2, label="Mean return (reward)")
ax.plot(dates, -comp_cvar, color="#e377c2", lw=1.2, label="-CVaR penalty")
ax.plot(dates, comp_vol,   color="#ff7f0e", lw=1.2, label="Variance penalty")
ax.plot(dates, comp_vexc,  color="#bcbd22", lw=1.2, label="Vol-excess penalty")
ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_ylabel("Component value")
ax.set_title("Objective Components")
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
axes2[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
fig2.autofmt_xdate()
fig2.tight_layout()
fig2.savefig(ROOT / "figures" / "validation_objective.png", dpi=150)
plt.close(fig2)
print("Saved: figures/validation_objective.png")

# ── PLOT 3: rolling portfolio variance ────────────────────────────────────────
roll_var_model  = pd.Series(model_ret,  index=dates).rolling(WINDOW).var() * 252
roll_var_market = pd.Series(market_ret, index=dates).rolling(WINDOW).var() * 252
roll_var_ipo    = pd.Series(ipo_ret,    index=dates).rolling(WINDOW).var() * 252
roll_var_equal  = pd.Series(equal_ret,  index=dates).rolling(WINDOW).var() * 252

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(dates, roll_var_model  * 100, lw=2,   color="#1f77b4", label="Model")
ax3.plot(dates, roll_var_equal  * 100, lw=1.5, color="#2ca02c", ls="--", label="Equal 50/50")
ax3.plot(dates, roll_var_market * 100, lw=1.5, color="#ff7f0e", ls="--", label="Market only")
ax3.plot(dates, roll_var_ipo    * 100, lw=1.5, color="#9467bd", ls="--", label="IPO index only")
ax3.axhline(
    target_vol**2 * 100, color="red", lw=1, ls=":",
    label=f"Target variance ({target_vol**2*100:.2f}%)"
)

ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
fig3.autofmt_xdate()
ax3.set_ylabel("Annualized Variance (%)")
ax3.set_title(f"Rolling Portfolio Variance ({WINDOW}-day, annualized) — Validation Period")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(ROOT / "figures" / "validation_variance.png", dpi=150)
plt.close(fig3)
print("Saved: figures/validation_variance.png")

# ── summary stats ─────────────────────────────────────────────────────────────
print(f"\nValidation: {dates[0].date()} to {dates[-1].date()}  ({len(dates)} days)")
print(f"  Model cum return : {cum(model_ret)[-1]*100:.2f}%")
print(f"  Market cum return: {cum(market_ret)[-1]*100:.2f}%")
print(f"  IPO cum return   : {cum(ipo_ret)[-1]*100:.2f}%")
print(f"  Equal cum return : {cum(equal_ret)[-1]*100:.2f}%")
print(f"  Model ann vol    : {model_ret.std()*np.sqrt(252)*100:.2f}%")
print(f"  Model avg turnover: {np.abs(np.diff(np.stack([w_m, w_i], axis=1), axis=0)).sum(axis=1).mean():.6f}")
