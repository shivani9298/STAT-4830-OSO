# Why the Model Allocates ~100% to IPOs

## Root Cause

The training loss has **no penalty for concentration**. It optimizes:

```
L = -mean_return + λ_cvar·CVaR + λ_vol·variance + λ_turnover·turnover + λ_path·weight_instability
```

1. **IPO vastly outperformed market in-sample** (2020–2024): ~192% vs ~19% total return.
2. To minimize `-mean_return`, the model is rewarded for putting more weight on the higher-return asset (IPO).
3. Volatility and CVaR do not favor diversification: a 100% IPO portfolio can have similar or acceptable risk for the period.
4. **No prior term** penalizes max weight or rewards entropy — so the optimal allocation is to go all-in on the better-performing asset.

## Fix: Diversification Penalty

A `loss_diversification_min_weight` term was added: it penalizes when the minimum asset weight falls below `min_weight` (default 10%). This forces the model to hold at least ~10% in both market and IPO.

- **λ_diversify** (default 1.0): strength of the diversification penalty
- **min_weight** (default 0.1): minimum target weight per asset

Set `lambda_diversify=0` to restore the original (no diversification) behavior.
