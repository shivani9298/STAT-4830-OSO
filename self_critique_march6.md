# Self-Critique: March 6 Deliverable

## OBSERVE: Critical Reading of the Report

After re-reading `report_march6.md` and re-running the full pipeline (`run_ipo_optimizer_wrds.py`):

- The model trains end-to-end and produces valid simplex-constrained weights (69% market / 31% IPO average)
- Sharpe of 3.27 on the validation period is strong, but this is the same data used for hyperparameter selection — no configs were evaluated on truly unseen data
- Real VIX replaced the constant 20.0 placeholder; the model now receives four informative features instead of three
- Turnover and path penalties are now tuned (previously hardcoded), which improved Sharpe from 3.07 to 3.27
- Training converges in ~4 effective epochs (11 total with early stopping), suggesting the model saturates quickly on ~930 training samples
- The IPO-only baseline has a *higher* Sharpe (3.92) than our model (3.27) — the model's value is in drawdown reduction (-7.47% vs -10.08%), not raw risk-adjusted return

---

## ORIENT: Analysis

### Strengths

1. **Systematic penalty tuning**: All six loss penalties are now searchable hyperparameters. The 288-config grid search is reproducible, saves intermediate results, and found that reducing `lambda_turnover` from 0.01 to 0.0025 meaningfully improved performance — a finding that would not emerge from intuition alone.

2. **Real feature inputs**: Replacing the constant VIX placeholder with actual CBOE VIX data from WRDS gives the GRU a genuine volatility regime signal. The data pipeline now pulls from four WRDS sources (SDC, CRSP, SPY/DIA, CBOE) with proper date alignment and forward-fill handling.

3. **Transparent baselines**: The report includes market-only, IPO-only, and equal 50/50 benchmarks evaluated on the identical validation period and date range. This makes the model's contribution (and limitations) clear — it beats market-only substantially but underperforms a naive 50/50 split on Sharpe.

### Areas for Improvement

1. **No out-of-sample test**: The validation set is both the evaluation set and the set used to select hyperparameters via grid search. This is methodologically equivalent to testing on training data. The reported Sharpe of 3.27 is optimistic — it reflects the best of 288 configs on this specific period, not expected future performance.

2. **Model underperforms naive 50/50**: The equal-weight baseline achieves Sharpe 3.65 vs our model's 3.27, with comparable drawdown (-7.20% vs -7.47%). The GRU adds complexity without demonstrably improving on a zero-parameter strategy. The report does not address this directly.

3. **CVaR formulation is approximate and not validated**: The `cvar_smooth` function uses a soft-sorting heuristic with a hardcoded temperature (0.1) that was never tuned or validated against the true CVaR. It is unclear whether this approximation is tight enough to meaningfully penalize tail risk, or if it degenerates to something simpler.

### Critical Risks/Assumptions

The central risk is **validation leakage through hyperparameter selection**. By choosing the config that maximizes validation Sharpe across 288 trials, we are implicitly fitting to the validation set. A Sharpe of 3.27 on a 232-day window with 288 searches amounts to substantial multiple-testing bias. Additionally, the IPO index suffers from survivorship bias (delisted stocks excluded) and the 2020–2024 period is unusually favorable for IPOs, making it unclear whether any model trained on this window would generalize to a regime with weak IPO performance.

---

## DECIDE: Concrete Next Actions

1. **Implement a train/validation/test split**: Reserve the final 60 trading days (~3 months) as a locked test set that is never used for hyperparameter selection. Tune on train/val (the remaining 80/20 split), then report final metrics exclusively on the test set. This directly addresses the validation leakage problem and requires only a change to `train_val_split` and the runner script.

2. **Benchmark against static allocations**: Add a "best static weight" baseline — find the single fixed weight $w^*$ that maximizes Sharpe on the training period and evaluate it on the same validation/test set. If the GRU cannot beat this, the model's complexity is unjustified. This directly addresses the 50/50 underperformance and can be implemented in ~20 lines in `export.py`.

3. **Validate the CVaR approximation**: Compare `cvar_smooth` output against the exact empirical CVaR (sort returns, average the worst 5%) on several batches. If the soft approximation deviates by more than 10%, tune the temperature parameter or replace with a differentiable quantile regression. This addresses the formulation rigor concern and requires adding a diagnostic function to `losses.py`.

---

## ACT: Resource Needs

To implement the test split, we need to decide on a cutoff date — the simplest approach is to use the last 60 trading days of the dataset (approximately Oct–Dec 2024), which requires no external data. For the static-weight baseline, we need a small optimization loop (scipy or brute-force over a grid of weights) applied to the training period returns, which is straightforward. For CVaR validation, we need to add a non-differentiable exact CVaR function and compare it against `cvar_smooth` across batches — this is a diagnostic, not a training change. No new libraries or WRDS access required; all three actions can be completed within one work session.
