# Development Log

Progress and key decisions for the IPO trading policy project.

## Structure

- **report.md**: Week 4 report (problem, approach, results, next steps).
- **notebooks/week4_implementation.ipynb**: Working implementation, validation, and docs.
- **src/model.py**: Thin wrapper around policy + objective (core optimization interface).
- **src/objective.py**, **src/backtest.py**, **src/train_policy.py**, **src/policy_network.py**: Core optimization and backtest.
- **tests/test_basic.py**: Basic validation (objective, metrics, backtest edge cases).

## Decisions

- **Objective**: Use E[R] - λ·CVaR - κ·E[Cost] - μ·MDD (no Sharpe in code yet) for consistency with `objective.score()`.
- **Optimization**: REINFORCE with Adam, mean-reward baseline, gradient clipping; backtest remains non-differentiable.
- **Validation**: Random train/val split in notebook; walk-forward by year planned as next step.

## Progress

- Week 3: IPO index builder, live data, fitness score, SPY comparison.
- Week 4: Report and notebook; test_basic; model.py wrapper; docs/llm_exploration and development_log; OnlinePortfolioOptimizer in build_ipo_index; self_critique.

## Failed attempts / notes

- Some IPO data fetches fail (delisted SPACs, ticker changes); build_ipo_index skips and reports count. test_data_meta, test_data_prices, test_data_schema expect older src.data API (load_ipo_meta, validate_columns, IPOInfo) and currently fail at import—either align src.data or skip those tests for Week 4.

## Safety

- No live trading or real capital. All code is backtest/simulation and research-only; build_ipo_index and run_pytorch use historical or synthetic data.
