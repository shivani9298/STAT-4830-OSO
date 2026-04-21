# Project File-by-File Guide

This document explains **every file** in the repo: what it does, key functions/classes, and how it fits into the project.

---

## Root / Top-level

### `README.md`
- **Purpose:** Project overview, structure, installation, quick start, and team roles.
- **Contents:** Describes the IPO trading strategy optimization framework, data sources (`synth` / `path` / yfinance), CLI args for `run_week3.py` and `run_pytorch.py`, objective formula (Score = E[R] − λ·CVaR − κ·E[Cost] − μ·MDD), policy parameters, baselines, data formats, testing, and Person A/B/C/D responsibilities.
- **Note:** Some README commands refer to an IPO random-search CLI; the current `run_week3.py` in the repo is the **stock-vs-S&P 500** comparison script (see below).

### `requirements.txt`
- **Purpose:** Python dependencies.
- **Contents:** `pandas`, `numpy`, `pytest`, `torch`, `yfinance`. Install with `pip install -r requirements.txt`.

### `.gitignore`
- **Purpose:** Ignore generated/cache files and envs.
- **Contents:** `__pycache__/`, `.pytest_cache/`, `venv/`, `results/*.json`, `results/*.jsonl`, `results/*.csv`, `results/report.md`, `results/policy_network.pt`, `data/*.csv`, `.DS_Store`, etc.

### `WEEK3_DELIVERABLE.md`
- **Purpose:** Week 3 deliverable write-up.
- **Contents:** Problem statement (optimize policy π_θ for IPO participation/sizing/timing), objective (E[R] − λ·CVaR − κ·E[C] − μ·MDD), constraints, data needs, algorithm choice (random search then optional PyTorch), validation ideas, initial results (synthetic + real data), test results, limitations, next steps, and implementation checklist.

---

## Entry points (scripts you run)

### `run_week3.py`
- **Purpose:** **Stock vs S&P 500** buy-and-hold comparison (single ticker vs SPY).
- **What it does:**
  - Takes `--ticker` (e.g. AAPL), `--benchmark` (default SPY), `--period` (e.g. 1y).
  - Fetches stock + benchmark via `fetch_stock_vs_benchmark()` from `src.data`.
  - Computes metrics: total return, volatility, Sharpe, max drawdown, beta, alpha, win rate, fitness (0.4·Sharpe + 0.3·AnnualReturn + 0.3·(1+MaxDD)).
  - Prints “BEAT / UNDERPERFORMED / MATCHED” and an ASCII price chart.
- **Key functions used:** `fetch_stock_vs_benchmark`, `calculate_metrics`, `calculate_fitness`.
- **Not:** The IPO random-search optimizer (that logic lives in `src/optimize.py` and is wired from `run_pytorch.py` for PyTorch policy; a standalone IPO random-search CLI could be added that uses `random_search()` from `src/optimize.py`).

### `run_pytorch.py`
- **Purpose:** Train the **IPO policy network** with REINFORCE (synth / path / yfinance).
- **What it does:**
  - **Data:** `--data synth` → synthetic episodes; `--data yfinance` → S&P 500 tickers from Wikipedia + prices via yfinance, then `build_episodes()`; `--data path` → `build_episodes_from_rich_csv(rich_csv)` or `load_ipo_meta(meta_csv)` + prices (dir or synthetic).
  - Splits episodes into train/val by `--val_frac`, then calls `train_reinforce()` from `src.train_policy`.
  - Saves `results/policy_network.pt` and `results/training_summary.json`.
- **Key args:** `--data`, `--rich_csv`, `--meta_csv`, `--max_tickers`, `--lookback_days`, `--n_epochs`, `--lr`, `--lr_schedule`, `--batch_size`, `--val_frac`, `--seed`, `--lam`, `--alpha`, `--kappa`, `--mu`, `--cost_bps`, `--out_dir`.

### `build_ipo_index.py`
- **Purpose:** Build a **market-cap weighted IPO index** from a daily historical CSV and compare to SPY.
- **What it does:**
  - Loads IPO dates from a CSV (e.g. `.pytest_cache/dailyhistorical_21-26.csv` or `--csv`); expects columns like `tic`, `datadate`, `prccd`, `gvkey`.
  - For each IPO, fetches 180 days of prices and shares outstanding via yfinance (`fetch_ipo_data`).
  - Builds daily index: each day, constituents are IPOs within 180 days of IPO; weights = market cap / total market cap; index return = weighted sum of daily returns.
  - Fetches SPY, merges, saves `results/ipo_index.csv`, `results/ipo_weights.csv`, `results/ipo_weights_pivot.csv`.
  - Prints index vs SPY return, fitness, and latest weights.
- **Note:** Default CSV path points to `.pytest_cache`; override with `--csv` if you use another file.

### `generate_report.py`
- **Purpose:** Turn optimization outputs into a **Markdown report**.
- **Inputs:** `results/best_params.json`, `results/trials.jsonl`, `results/results.csv` (from a run that produced these, e.g. a random-search or backtest that writes the same format).
- **What it does:** Loads best params and trials, computes participation rate, win rate, cost impact; writes `results/report.md` with parameter tables, interpretation (participation, entry timing, hold period, sizing), performance stats, trial analysis, and insights.
- **Note:** Expects the IPO-optimization result format (best_params with `participate_threshold`, `entry_day`, `hold_k`, etc.). Running it requires having run the optimizer (e.g. via code that uses `src.optimize.random_search`) so that `best_params.json` and `results.csv` exist.

---

## Source package: `src/`

### `src/__init__.py`
- Empty package marker (or short comment). No exports.

### `src/data.py`
- **Purpose:** Data loading and **episode** construction for both “stock vs benchmark” and “IPO” use cases.
- **Main types:**
  - **`Episode`:** Dataclass: `ticker`, `ipo_date` (or start date), `df` (DataFrame with `date`, `close`, optionally `volume`, `benchmark_close`), `day0_index`, `N`, optional `meta`.
- **Key functions:**
  - **`get_sp500_tickers()`:** Fetches current S&P 500 symbols from Wikipedia.
  - **`fetch_stock_data(ticker, period=..., start=..., end=...)`:** Single-ticker daily prices from yfinance → `date`, `close`, `volume`.
  - **`fetch_stock_vs_benchmark(ticker, benchmark='SPY', period='1y')`:** Stock + benchmark aligned on date → adds `benchmark_close` to the stock DataFrame.
  - **`build_episodes_from_stock(ticker, benchmark, period, N, step)`:** Rolling N-day windows from one stock vs benchmark → list of `Episode`.
  - **`load_prices_from_yfinance(meta_df, N, buffer_days, fetch_days)`:** Batch download for many tickers (e.g. S&P 500); returns `prices_map: ticker → DataFrame(date, close, volume)`.
  - **`build_episodes(meta_df, prices_map, N, short_mode)`:** From a table of (ticker, ipo_date) and per-ticker price DataFrames, builds one `Episode` per row; `short_mode` = `'skip'` or `'truncate'` when data is shorter than N.
  - **`generate_synthetic_prices(ticker, ipo_date, N, initial_price, volatility, rng)`:** Random walk prices + volume for testing.
- **Note:** `run_pytorch.py` and some tests also expect `load_ipo_meta` and `build_episodes_from_rich_csv`; if those are not present in `data.py`, those call paths will need to be restored or stubbed (see Tests below).

### `src/policy.py`
- **Purpose:** **Rule-based** IPO trading policy (participate/skip, entry, exit, weight).
- **Main types:**
  - **`PolicyParams`:** participate_threshold, entry_day, hold_k, w_max, raw_weight, use_volume_cap, vol_cap_mult, optional min_offer_amount / max_offer_amount (from meta).
- **Key functions:**
  - **`sample_params(rng, max_hold)`:** Random policy params for optimization.
  - **`decide_trade(episode, params)`:** Returns dict `participate`, `entry_day`, `exit_day`, `weight`. Uses early price (or excess vs benchmark) vs `participate_threshold`; applies meta filters (offer amount) if `episode.meta` exists; computes weight with optional volume cap and clips to w_max.

### `src/backtest.py`
- **Purpose:** Simulate strategy execution on episodes.
- **Key functions:**
  - **`backtest_episode(episode, decision, cost_bps)`:** One episode: if not participate → zeros; else entry/exit prices, gross_ret, benchmark_ret (if column present), excess_ret, cost (cost_bps/1e4 * weight), net_ret = weight * excess_ret − cost, pnl.
  - **`backtest_all(episodes, params, cost_bps)`:** For each episode, `decide_trade(episode, params)` then `backtest_episode`; returns `(results_df, equity_series)` (equity = cumulative product of (1 + net_ret)).
  - **`backtest_all_with_decisions(episodes, decisions, cost_bps)`:** Same but uses provided decision dicts (e.g. from policy network); returns same shape.

### `src/metrics.py`
- **Purpose:** Risk and cost metrics.
- **Key functions:**
  - **`cvar(returns, alpha=0.9)`:** CVaR (expected loss in worst (1−α) tail).
  - **`max_drawdown(equity_curve)`:** Max drawdown from equity series.
  - **`summarize_costs(results_df)`:** total_cost, avg_cost_per_trade, total_trades.
  - **`turnover(results_df)`:** Sum of absolute weights over participating trades.

### `src/objective.py`
- **Purpose:** **Score** and metrics for optimization.
- **Key function:** **`score(results_df, equity, lam, alpha, kappa, mu)`**  
  - E[R] = mean(net_ret) over all episodes; E[R]_per_trade and n_trades for participations; CVaR, E[Cost], MDD.  
  - **Score = E[R] − λ·CVaR − κ·E[Cost] − μ·MDD.**  
  Returns `(score_value, metrics_dict)`.

### `src/optimize.py`
- **Purpose:** **Random search** over rule-based policy and baselines.
- **Key functions:**
  - **`random_search(episodes, n_trials, seed, objective_kwargs, out_dir)`:** Samples `PolicyParams` with `sample_params`, runs `backtest_all`, `score`, appends to `trials.jsonl`, keeps top-5 leaderboard; saves best params to `best_params.json`, best backtest to `results.csv`; returns best params, score, metrics, leaderboard.
  - **`baseline_always_skip(episodes)`**, **`baseline_always_participate(episodes, weight)`**, **`baseline_fixed_hold_k(episodes, hold_k, weight)`:** Fixed policies, each returns name, params, score, metrics.

### `src/logging_utils.py`
- **Purpose:** Trial logging.
- **Key function:** **`log_trial(path, trial_record)`:** Appends one JSON object per line to a JSONL file (e.g. `trials.jsonl`).

### `src/features.py`
- **Purpose:** **Feature vector** for the policy network (episode → fixed-size vector).
- **Constants:** `N_FEATURES` (20): 8 base (from `episode.df`: day0/day1 close, return, vol proxy, volume, N, const) + 12 meta (offer price, shares, offer amount, employees, sector/industry hashes, CEO, firstday open/close, first-day return).
- **Key functions:**
  - **`episode_to_features(episode)`:** Returns length-20 float32 array.
  - **`episodes_to_tensor(episodes, device)`:** Stack features → (B, N_FEATURES) tensor.

### `src/policy_network.py`
- **Purpose:** **PyTorch policy** (contextual bandit): features → participate / entry / hold / weight.
- **Classes:** **`IPOPolicyNetwork`:** MLP → heads for participate (logit), entry_day (2 choices), hold_k (1..9), weight (sigmoid × w_max).
- **Key functions:**
  - **`sample_actions`** / **`sample_and_log_prob`:** Sample actions and log probs for REINFORCE.
  - **`policy_network_to_decision_list`:** Run policy on episodes (no grad), return list of decision dicts.

### `src/train_policy.py`
- **Purpose:** **REINFORCE** training (Adam, optional LR schedule, validation).
- **Key function:** **`train_reinforce(episodes, val_episodes, n_epochs, lr, lr_schedule, cost_bps, lam, alpha, kappa, mu, batch_size, entropy_coef, seed, device, out_dir)`:**  
  Minibatches of episodes → features → sample actions → backtest → reward = net_ret (or batch score); loss = −(log_prob × advantage).mean(); optional cosine/step LR; logs train/val score; saves `policy_network.pt` and `training_summary.json`; prints interpretation.

---

## Tests: `tests/`

- **`tests/__init__.py`:** Package marker.
- **`tests/test_data_schema.py`:** Schema/validation tests (e.g. required columns).
- **`tests/test_data_meta.py`:** **`load_ipo_meta`** and **`validate_columns`** (expect these in `src.data`; if removed, these tests will fail until restored or skipped).
- **`tests/test_data_prices.py`:** **`load_prices_dir`**, **`validate_columns`** (same dependency on `data.py`).
- **`tests/test_build_episodes.py`:** **`build_episodes`**, **`Episode`** (build episodes from meta + prices_map, short_mode, etc.).

Run: `pytest tests/ -v`.

---

## Docs: `docs/`

- **`RESULTS_GLOSSARY.md`:** Defines score, E[R], CVaR, E[Cost], MDD, n_trades, and why score is “per opportunity”; explains why numbers are small (sparse bars, fractional weights, conservative policies).
- **`COURSE_CONCEPTS.md`:** Maps course topics (e.g. autodiff, SGD/Adam, validation) to code (policy network, train_policy, etc.).
- **`TRAINING_INTERPRETATION.md`:** How to interpret training summary and validation scores.
- **`GET_GVKEY_MAPPING.md`:** How to get GVKEY (e.g. from WRDS/Compustat) for tickers.
- **`IPOnames_vs_IPO_count_explanation.md`:** Why IPO counts in deal databases differ from “US IPO” counts.
- **`IPOnames_mitigation.md`:** Mitigations for name/ID issues in IPO data.

---

## Data flow (high level)

1. **Stock vs S&P 500:**  
   `run_week3.py` → `data.fetch_stock_vs_benchmark` → `calculate_metrics` / `calculate_fitness` → print.

2. **IPO rule-based optimization:**  
   Episodes (from data.py or CSV) → `optimize.random_search` → `policy.decide_trade` → `backtest.backtest_all` → `objective.score` → save best_params, results.csv, trials.jsonl.  
   (No dedicated CLI in repo right now; you can call `random_search()` from a small script or add a CLI in `run_week3.py`.)

3. **IPO PyTorch policy:**  
   Episodes → `run_pytorch.py` → `train_policy.train_reinforce` → `policy_network` + `features` + `backtest_all_with_decisions` + `objective.score` → save policy_network.pt, training_summary.json.

4. **Report:**  
   `generate_report.py` reads best_params.json, trials.jsonl, results.csv → writes results/report.md.

---

## Summary table

| File | Role |
|------|------|
| `run_week3.py` | Stock vs S&P 500 comparison (single ticker vs SPY) |
| `run_pytorch.py` | Train IPO policy network (REINFORCE, synth/path/yfinance) |
| `build_ipo_index.py` | Build market-cap weighted IPO index vs SPY |
| `generate_report.py` | Markdown report from optimization results |
| `src/data.py` | Episodes, S&P 500 list, yfinance fetch, build_episodes, synthetic prices |
| `src/policy.py` | Rule-based policy (PolicyParams, decide_trade, sample_params) |
| `src/backtest.py` | backtest_episode, backtest_all, backtest_all_with_decisions |
| `src/metrics.py` | CVaR, MDD, costs, turnover |
| `src/objective.py` | score() = E[R] − λ·CVaR − κ·E[Cost] − μ·MDD |
| `src/optimize.py` | random_search, baselines |
| `src/logging_utils.py` | log_trial (JSONL) |
| `src/features.py` | episode → feature vector (20 dims) |
| `src/policy_network.py` | IPOPolicyNetwork (PyTorch), action sampling |
| `src/train_policy.py` | train_reinforce (REINFORCE + Adam) |

This is the full file-by-file picture of the repo as it stands.
