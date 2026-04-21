# Gamma Presentation Prompt

Create a professional, modern slide deck for a university research project progress presentation. The course is **STAT 4830 at the University of Pennsylvania, Spring 2026**. The project title is **"IPO Portfolio Optimization with GRU-Based Allocation."** The tone should be technical but accessible, suitable for a graduate-level statistics/data science audience. Use a clean, modern financial/quantitative theme with dark or navy tones. The presentation should follow the structure below.

---

## Slide 1: Title Slide

**Title:** IPO Portfolio Optimization with GRU-Based Allocation
**Subtitle:** STAT 4830 — Spring 2026 — University of Pennsylvania
**Author:** Shivani

---

## Slide 2: Problem Statement

We optimize **daily portfolio weights** between two asset classes:
- A **market index** (82% S&P 500, 18% Dow Jones, proxied via SPY and DIA from CRSP)
- A **custom IPO index** — a market-cap-weighted basket of recent IPOs sourced from SDC Platinum, where each stock is held for 180 trading days post-IPO

**Why this matters:** IPOs exhibit 3–4× higher volatility than the broad market, with mean reversion in the first six months and information asymmetry between institutional and retail participants. Static allocations ignore regime shifts (COVID crash 2020, IPO boom 2021, correction 2022). A systematic, data-driven allocator that adjusts IPO tilt based on market conditions can improve risk-adjusted returns.

**Constraints:** Long-only, fully invested, no shorting or leverage, daily rebalancing.

---

## Slide 3: Data Sources

Show a table of the four institutional data sources:

| Source | Content | Scale |
|--------|---------|-------|
| SDC Platinum (WRDS) | IPO issue dates for each issuer | 1,136 tickers |
| CRSP Daily Stock File (WRDS) | Split-adjusted prices, shares outstanding | 770,418 rows |
| CRSP SPY + DIA | Market returns (82%/18% blend) | 1,258 trading days |
| CBOE via WRDS | Daily VIX closing level | 1,535 days |

Date range: 2020-01-01 through 2024-12-31. This institutional WRDS data (SDC + CRSP + CBOE) provides a rigorous, reproducible foundation — far superior to yfinance-based prototyping.

---

## Slide 4: Technical Approach — Model Architecture

The model is a **GRU (Gated Recurrent Unit)** neural network built in PyTorch:
- **Input:** 84-day rolling window of 4 features (market return, IPO return, 21-day rolling volatility, VIX level)
- **Architecture:** GRU (input=4, hidden=64, 1 layer) → 2-layer MLP (64→64→2) → Softmax output
- **Output:** Portfolio weights w = [w_market, w_IPO] with w_i ≥ 0, Σw_i = 1

The GRU extracts temporal patterns from the rolling window; the softmax output guarantees valid simplex-constrained weights. GRU was chosen over LSTM for fewer parameters and faster training on our small dataset (~1,100 samples).

---

## Slide 5: Technical Approach — Loss Function

The model minimizes a composite differentiable loss:

**L = −mean(r_p) + λ_cvar · CVaR + λ_vol · Var(r_p) + λ_vol_excess · max(0, σ_ann − σ_target) + λ_turnover · ‖w_t − w_{t-1}‖₁ + λ_path · ‖w_t − w_{t-1}‖₂²**

Six modular, tunable components:
- **Return maximization** (negative mean return)
- **Tail risk control** (CVaR at 5% level via soft-sorting approximation)
- **Variance penalty** and **volatility excess penalty** (target 25% annual vol)
- **Turnover penalty** (L1) and **path stability** (L2 squared)

No labels or supervised targets required — end-to-end gradient-based optimization.

---

## Slide 6: Hyperparameter Tuning

**288-configuration grid search** over all six loss penalty weights, target volatility, and window length. ~19 minutes total runtime.

Key tuning results:

| Parameter | Previous (Hardcoded) | Tuned (Best) |
|-----------|---------------------|--------------|
| λ_turnover | 0.01 | **0.0025** |
| λ_path | 0.01 | **0.01** |
| λ_vol | 0.5 | **1.0** |
| λ_cvar | 0.5–1.0 | **1.0** |
| window_len | 126 days | **84 days** |

Reducing turnover penalty from 0.01 → 0.0025 allowed more active rebalancing. Heavier risk penalties kept volatility controlled. Shorter 84-day lookback outperformed 126 days. Training converges in ~4 effective epochs (11 total with early stopping).

---

## Slide 7: Results — Performance Comparison

Show a table comparing all strategies on the validation period (Jan–Dec 2024):

| Strategy | Total Return | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|----------|-------------|-------------|----------|--------|-------------|
| **Model Portfolio** | **58.81%** | **65.27%** | **15.74%** | **3.27** | **-7.47%** |
| Market only | 19.40% | 21.24% | 12.25% | 1.63 | -7.89% |
| IPO only | 192.78% | 221.19% | 31.03% | 3.92 | -10.08% |
| Equal 50/50 | 88.52% | 99.11% | 19.39% | 3.65 | -7.20% |

The model allocates ~69% market / 31% IPO on average with avg daily turnover of 0.0031. It achieves tighter drawdown than IPO-only (-7.47% vs -10.08%) and substantially higher return than market-only (65% vs 21%).

---

## Slide 8: Results — Key Metrics Summary

Highlight four success metrics with a visual scorecard:

| Metric | Target | Achieved |
|--------|--------|----------|
| Annualized Sharpe Ratio | > 1.5 | **3.27** ✓ |
| Max Drawdown | < 15% | **-7.47%** ✓ |
| Annualized Return | > Market (SPY) | **65.27% vs 21.24%** ✓ |
| Avg Turnover | Low (< 0.01/day) | **0.0031** ✓ |

---

## Slide 9: OBSERVE — What We See

Critical observations from re-reading the report and re-running the pipeline:

- The model trains end-to-end and produces valid simplex-constrained weights (69% market / 31% IPO average)
- Sharpe of 3.27 is strong, but validation set was also used for hyperparameter selection — no configs were evaluated on truly unseen data
- Real VIX replaced the constant 20.0 placeholder; the model now receives four informative features
- Turnover and path penalties were tuned (previously hardcoded), improving Sharpe from 3.07 → 3.27
- Training converges in ~4 effective epochs, suggesting the model saturates quickly on ~930 training samples
- The IPO-only baseline has a *higher* Sharpe (3.92) than the model (3.27) — the model's value is in **drawdown reduction** (-7.47% vs -10.08%), not raw risk-adjusted return

---

## Slide 10: ORIENT — Strengths

1. **Systematic penalty tuning:** All six loss penalties are searchable hyperparameters. The 288-config grid search is reproducible and found that reducing λ_turnover from 0.01 → 0.0025 meaningfully improved performance — a finding that would not emerge from intuition alone.

2. **Real feature inputs:** Replacing the constant VIX placeholder with actual CBOE VIX data gives the GRU a genuine volatility regime signal. The data pipeline now pulls from four WRDS sources with proper date alignment and forward-fill handling.

3. **Transparent baselines:** Market-only, IPO-only, and equal 50/50 benchmarks evaluated on the identical validation period make the model's contribution and limitations clear.

---

## Slide 11: ORIENT — Areas for Improvement

1. **No out-of-sample test:** The validation set is both the evaluation set and the set used for hyperparameter selection. This is methodologically equivalent to testing on training data. The Sharpe of 3.27 is optimistic — it reflects the best of 288 configs on this specific period.

2. **Model underperforms naive 50/50:** The equal-weight baseline achieves Sharpe 3.65 vs the model's 3.27, with comparable drawdown (-7.20% vs -7.47%). The GRU adds complexity without demonstrably improving on a zero-parameter strategy.

3. **CVaR formulation is approximate and unvalidated:** The soft-sorting heuristic uses a hardcoded temperature (0.1) that was never tuned or validated against the true CVaR.

---

## Slide 12: ORIENT — Critical Risks

**Validation leakage through hyperparameter selection:** Choosing the config that maximizes validation Sharpe across 288 trials implicitly fits to the validation set. A Sharpe of 3.27 on a 232-day window with 288 searches amounts to substantial multiple-testing bias.

**Survivorship bias:** The IPO index excludes delisted/failed stocks, inflating index returns.

**Regime dependence:** The 2020–2024 period is unusually favorable for IPOs. It is unclear whether any model trained on this window would generalize to a regime with weak IPO performance.

---

## Slide 13: DECIDE — Concrete Next Actions

1. **Implement train/validation/test split:** Reserve the final 60 trading days (~3 months) as a locked test set never used for hyperparameter selection. Tune on train/val, then report final metrics exclusively on the test set. This directly addresses the validation leakage problem.

2. **Benchmark against static allocations:** Add a "best static weight" baseline — find the single fixed weight w* that maximizes Sharpe on the training period and evaluate it on the same test set. If the GRU cannot beat this, the model's complexity is unjustified.

3. **Validate the CVaR approximation:** Compare the soft-sorted CVaR output against the exact empirical CVaR (sort returns, average worst 5%) on several batches. If the approximation deviates by more than 10%, tune the temperature or replace with differentiable quantile regression.

---

## Slide 14: ACT — Resource Needs and Timeline

- **Test split:** Requires only a change to the split logic and runner script. No external data needed. Use the last 60 trading days of the dataset (~Oct–Dec 2024).
- **Static weight baseline:** A small optimization loop (scipy or brute-force grid over weights) applied to training period returns. ~20 lines of code.
- **CVaR validation:** Add a non-differentiable exact CVaR function and compare against cvar_smooth across batches. Diagnostic only — no training change.
- **No new libraries or WRDS access required.** All three actions can be completed within one work session.

---

## Slide 15: What We've Learned

1. **Penalty tuning matters:** Hardcoded regularization left performance on the table. Systematic grid search over λ_turnover and λ_path yielded Sharpe improvement (3.07 → 3.27).
2. **Feature quality matters:** Replacing a constant VIX placeholder with real CBOE data gave the model actual volatility regime information.
3. **Risk penalties are essential:** Pure return maximization drives 100% IPO allocation. The interplay between λ_vol, λ_cvar, and λ_vol_excess produces a balanced, investable portfolio.
4. **Institutional data at scale works:** WRDS (SDC + CRSP + CBOE) provides a rigorous, reproducible foundation with 1,136 IPO tickers and 1,535 days of VIX.
5. **Simple models converge fast:** The GRU learns in ~4 effective epochs. The bottleneck is data quantity, not model capacity.

---

## Slide 16: Questions & Discussion

Open floor for questions about:
- Methodology and model design choices
- Data pipeline and WRDS integration
- Next steps for out-of-sample validation
- Alternative approaches (reinforcement learning, ensemble methods, online learning)
