# Grading Criteria Checklist — build_ipo_index.py only

This checklist evaluates **only** the IPO index builder (`build_ipo_index.py`) and its direct documentation/outputs. Other files (run_pytorch, train_policy, policy network, etc.) are **not** considered.

**✅ = Yes, ⚠ = Partial, ❌ = No.**

---

## Report

*(Primary report for the index: WEEK3_DELIVERABLE.md)*

| Criterion | Status | Where / Note |
|-----------|--------|--------------|
| **Clear problem definition** | ✅ | WEEK3_DELIVERABLE § Problem Statement: what is built (market-cap weighted IPO index, 180-day window), why it matters, success metrics (fitness, vs SPY), constraints, data (dailyhistorical CSV, yfinance), what could go wrong. |
| **Well-formulated technical approach** | ✅ | WEEK3_DELIVERABLE § Technical Approach: index construction (weights, returns), fitness formula (0.4×Sharpe + 0.3×return + 0.3×(1+MDD)); optional OnlinePortfolioOptimizer (gradient ascent, simplex). |
| **Evidence of testing/validation** | ✅ | WEEK3_DELIVERABLE § Initial Results: index built, metrics (return, vol, MDD, fitness) for IPO index vs SPY; data attrition and limitations; resource usage (runtime, memory, disk). |
| **Thoughtful next steps** | ✅ | WEEK3_DELIVERABLE § Next Steps: immediate improvements, technical challenges, questions, alternative approaches, what we learned. |

---

## Implementation

*(Only build_ipo_index.py and what it uses)*

| Criterion | Status | Where / Note |
|-----------|--------|--------------|
| **Code runs end-to-end** | ✅ | `python3 build_ipo_index.py --csv src/dailyhistorical_21-26.csv --max_ipos 20` runs: loads CSV → fetches data → builds index → writes results/ipo_index.csv, ipo_weights.csv, ipo_weights_pivot.csv → prints summary and fitness. |
| **Clear objective function** | ✅ | **Market-cap mode**: weights = market_cap / total_market_cap (no optimization). **Optimizer mode** (`--use-optimizer`): gradient combines return (0.4), volatility (0.3), return again (0.3); fitness reported = 0.4×Sharpe + 0.3×annual return + 0.3×(1+MDD). Both are explicit in code and WEEK3_DELIVERABLE. |
| **Working optimization loop** | ✅ | With `--use-optimizer`: `OnlinePortfolioOptimizer` updates weights each day (gradient ascent + `project_onto_simplex`). Loop over dates in `build_ipo_index()`. Without flag: deterministic market-cap weighting (no loop to optimize). |
| **Basic validation/testing** | ⚠ | No dedicated unit tests for `build_ipo_index.py`. Validation = manual run and inspection of output CSVs and printed metrics. Script has been run successfully (e.g. 20 IPOs, 16 fetched, index built). |
| **Resource monitoring** | ⚠ | WEEK3_DELIVERABLE documents runtime (~3–5 min for 100 IPOs), memory (<500MB), disk. Script does not print runtime/memory itself; could add a one-line timer at end of main(). |

---

## Development Process

*(Only as it relates to build_ipo_index)*

| Criterion | Status | Where / Note |
|-----------|--------|--------------|
| **AI conversations show exploration** | ✅ | docs/llm_exploration/week4_log.md: dated entry on adding OnlinePortfolioOptimizer and gradient descent to build_ipo_index. |
| **Failed attempts documented** | ✅ | WEEK3_DELIVERABLE: failed fetches (delisted SPACs, no shares), data attrition. development_log: fetch failures, no live trading. |
| **Design decisions explained** | ✅ | WEEK3_DELIVERABLE and development_log: market-cap vs optimizer option, 180-day window, fitness formula, yfinance. |
| **Safety considerations** | ✅ | development_log “Safety”: no live trading, research-only. build_ipo_index uses historical/synthetic data only. |
| **Alternative approaches considered** | ✅ | WEEK3_DELIVERABLE “Alternative approaches to try” (equal-weight, momentum filter, shorter holding, etc.). build_ipo_index has `--use-optimizer` as one alternative. |

---

## Repository Structure

*(Relevant to build_ipo_index only)*

| Criterion | Status | Where / Note |
|-----------|--------|--------------|
| **Clean organization** | ✅ | build_ipo_index.py at repo root; results/ for output; docs/ for WEEK3_DELIVERABLE and other docs; data from src/dailyhistorical_21-26.csv (or --csv). |
| **Clear documentation** | ✅ | README (project structure, how to run); WEEK3_DELIVERABLE (full problem, approach, results, next steps); docstrings in build_ipo_index.py (load_ipo_dates, fetch_ipo_data, build_ipo_index, OnlinePortfolioOptimizer, etc.). |
| **Working tests** | ⚠ | No pytest for build_ipo_index. “Working” = script runs and produces expected CSVs and summary. |
| **Complete logs** | ✅ | docs/development_log.md (progress, decisions, failed attempts, safety). docs/llm_exploration/week4_log.md (dated entry re build_ipo_index + optimizer). |

---

## Summary

- **Report**: Covered by WEEK3_DELIVERABLE (problem, approach, results, next steps) for the IPO index.
- **Implementation**: build_ipo_index.py runs end-to-end, has a clear objective (fitness and/or optimizer gradient), and has an optional working optimization loop (OnlinePortfolioOptimizer). No dedicated tests; validation is manual run. Resource usage is documented but not printed by the script.
- **Development process & repo**: Logs and docs that refer to the index are in place; safety and alternatives are documented.

**Optional improvements**: (1) Add a single end-to-end test or script that runs build_ipo_index with small --max_ipos and checks output files exist. (2) Print elapsed time at end of main() in build_ipo_index.py for resource monitoring.
