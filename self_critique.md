# Self-Critique: Week 4 Deliverable

## OBSERVE: Critical Reading of the Report

After reviewing the report and running the implementation:
- The GRU-based allocator runs end-to-end and produces valid weights
- Mathematical formulation is present; loss terms are implemented and tunable
- Real WRDS data (SDC + CRSP) replaces prior yfinance setup
- Validation metrics (Sharpe 2.53, MaxDD -7.66%) are strong but limited to in-sample/validation period
- Weights are nearly constant (~84% market, ~16% IPO); little tactical tilting
- Turnover displays as 0.0000 due to formatting; actual value ~1e-5

---

## ORIENT: Analysis

### Strengths 

1. **Rigorous Data Pipeline**: Uses institutional WRDS data (SDC for IPO dates, CRSP for prices and market returns) with explicit handling of split-adjusted prices and shares. Data volume (770K rows, 1,136 tickers) and date alignment are well managed.

2. **Differentiable, Modular Loss**: The objective function is clearly decomposed (mean return, CVaR, variance, vol excess, turnover, path stability) with configurable hyperparameters. This supports interpretability and systematic tuning.

3. **Evidence of Validation**: Hyperparameter grid search (32 configs) optimizes validation Sharpe; early stopping prevents overfitting; baseline comparisons (market, IPO-only, 50/50) provide context.

### Areas for Improvement 

1. **No True Out-of-Sample Test**: All reported metrics are on the validation set (last 20% of 2020–2024). There is no held-out test period (e.g., 2025 or a fixed future window) to assess generalization. Validation Sharpe can overstate performance.

2. **Near-Constant Weights Undercut “Optimization”**: The model outputs almost identical weights every day. The GRU may be underfitting to time-varying signals, or the loss heavily penalizes turnover/path instability. The value of a neural allocator over a simple static 84/16 rule is unclear.

3. **Mathematical Formulation Incomplete in Report**: The CVaR smoothing and softmax mapping are mentioned but not fully specified. A reader cannot reproduce the objective without reading the code. Equations for \(L_{\text{cvar}}\) and the GRU → MLP → softmax architecture should be written explicitly.

### Critical Risks/Assumptions

**Overfitting to validation**: The best config is chosen to maximize validation Sharpe. Without a true test set, we do not know if 2.53 Sharpe generalizes. The 2020–2024 period (including COVID and IPO boom) may not represent future regimes. We assume WRDS data is accurate and that our IPO index construction (180 days, market-cap weighted) is a reasonable proxy for investable IPO exposure—both are material assumptions.

---

## DECIDE: Concrete Next Actions

1. **Introduce a True Test Split**: Reserve the last 3–6 months of data (or 2025 when available) as a never-touched test set. Train and tune only on train/validation; report Sharpe, return, drawdown, and turnover exclusively on the test period. Document the split dates clearly in the report.

2. **Diagnose Why Weights Are Static**: Add logging of weight variance over time and correlation with features (e.g., rolling vol, momentum). If the model is collapsing to constants, experiment with (a) reducing turnover/path penalties, (b) adding features that vary more, or (c) increasing model capacity. Document findings in the report.

3. **Complete Mathematical Specification in Report**: Add explicit formulas for the CVaR approximation (soft sorting, temperature), the GRU/MLP architecture, and the softmax output. Include a small diagram or pseudocode so the approach is reproducible from the report alone.

---

## ACT: Resource Needs

To implement these actions, we need: (1) A clear date boundary for the test set (e.g., "2025-01-01 onward" or "last 60 trading days") and a script change to enforce it; (2) Basic feature analysis (e.g., pandas rolling stats) to inspect input variance; (3) Optional: learning PyTorch hooks or activation logging if we need to debug the GRU hidden states. Blocker: CRSP data for 2025 may not be available yet; we may need to use the last portion of 2024 as a proxy test period and note this limitation.
