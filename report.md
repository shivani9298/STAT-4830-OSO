# Week 6 Report: IPO Trading Policy Optimization
*Date: February 20, 2026*

## 1. Problem Statement (~½ page)

### What are you optimizing?

We optimize a **parameterized trading policy** \(\pi_\theta\) that, for each IPO episode, decides: (i) **participate or skip**, (ii) **entry day** (e.g., day 0 or day 1), (iii) **holding horizon** (1–9 days), and (iv) **position size** \(w \in [0, w_{\max}]\). The policy maps IPO-level features (price, volatility proxy, volume, optional metadata) to these actions. The goal is to maximize a **risk-adjusted fitness** over many IPO episodes:

$$
\mathrm{Score}_\theta = \mathbb{E}[R_\theta] - \lambda \cdot \mathrm{CVaR}_\alpha(R_\theta) - \kappa \cdot \mathbb{E}[C_\theta] - \mu \cdot \mathrm{MDD}_\theta
$$

with \(\lambda, \kappa, \mu\) defaulting to 1.0 and \(\alpha = 0.9\). **Note on Sharpe**: An extended form \(\mathrm{Fitness}_\theta = \mathbb{E}[R_\theta] + \beta \cdot \mathrm{Sharpe}(R_\theta) - \lambda \cdot \mathrm{CVaR}_\alpha - \mu \cdot \mathrm{MDD}_\theta - \kappa \cdot \mathbb{E}[C_\theta]\) is noted for future work but is not implemented in `src/objective.py`; adding β·Sharpe is a listed next step.

### Why does this problem matter?

Retail IPO trading is volatile and friction-heavy (opening auction noise, halts, wide spreads). A **systematic, risk-aware policy** gives a clear objective, constraints, and reproducible evaluation. It supports research into event-driven strategies and risk control instead of ad-hoc heuristics.

### How will you measure success?

- **Primary**: Objective score (E[R] − λ·CVaR − κ·E[Cost] − μ·MDD) on held-out IPO cohorts.
- **Secondary**: Lower tail risk (CVaR) without collapsing average return; robustness across cohorts/years; Sharpe and max drawdown.
- **Process**: Walk-forward validation (train on earlier years, test on later); basic test suite and notebook runs.

### What are your constraints?

- **Position**: \(w \in [0, w_{\max}]\) per IPO; portfolio exposure can be capped.
- **Liquidity/cost**: Transaction cost modeled as \(\kappa \cdot \mathbb{E}[C_\theta]\) (e.g., cost in bps).
- **Non-differentiable backtest**: Fills, discrete entry/exit, and constraints are handled in a simulator; optimization uses policy gradients (REINFORCE) rather than end-to-end backtest gradients.

### What data do you need?

- **IPO episodes**: For each IPO, a price series (e.g., daily close/volume) for \(N\) days from first trading day; optional metadata (sector, size, etc.) for features.
- **Source**: COMPUSTAT-style daily data, rich IPO CSV, or synthetic data for testing; live data via yfinance for index/benchmark work.

### What could go wrong?

- **Safety**: No live trading or real capital; all work is backtest/simulation and research-only.
- **Overfitting**: Limited IPOs per year; policy can overfit to a hot/cold regime.
- **Data attrition**: Delisted tickers, ticker changes, missing shares/volume (e.g., 40–50% fetch failures in some cohorts).
- **Lookahead/leakage**: Using day-1 outcomes to decide day-1 entry; we use only features available at decision time.
- **Cost model**: Day-1 spreads and execution uncertainty may be worse than a simple bps cost; results sensitive to \(\kappa\).

---

## 2. Technical Approach (~½ page)

### Mathematical formulation

- **State**: Feature vector \(x_i\) per episode (e.g., normalized close, return, vol proxy, volume, day index; optionally sector, size).
- **Actions**: Participate \(\in \{0,1\}\), entry day \(\in \{0,1\}\) (MVP), hold days \(\in \{1,\ldots,9\}\), weight \(\in [0, w_{\max}]\).
- **Return**: Per episode \(i\), \(r_i = w_i \cdot (\text{excess return}) - \text{cost}_i\); portfolio return \(R_\theta\) is the series of \(r_i\) over episodes; equity curve is cumulative product of \((1 + r_i)\).
- **Objective**:
  $$\max_\theta \ \mathrm{Score}_\theta = \mathbb{E}[R_\theta] - \lambda \cdot \mathrm{CVaR}_\alpha(R_\theta) - \kappa \cdot \mathbb{E}[C_\theta] - \mu \cdot \mathrm{MDD}_\theta$$
  CVaR is the expected loss in the worst \((1-\alpha)\) tail; MDD from the cumulative equity curve. This is implemented verbatim in `src/objective.score()`.

### Algorithm and justification

- **REINFORCE (policy gradient)** with a **PyTorch policy network**: Backtest is non-differentiable, so we maximize \(\mathbb{E}[\log \pi_\theta(a|x) \cdot G]\) where \(G\) is episode return (or advantage). Baseline = mean reward in batch for variance reduction. **Adam** for updates; optional **cosine/step** learning-rate schedules.
- **Why policy gradient**: Fits discrete actions and non-differentiable simulator; works with modest sample sizes; interpretable action distributions.
- **Rule-based alternative**: Random search over rule parameters (e.g., `src/optimize.py`) for fast iteration and ablation.

### PyTorch implementation strategy

- **Episodes → tensor**: `features.episodes_to_tensor()` builds \((B, n\_features)\) from a list of `Episode`; missing meta filled with zeros.
- **Network**: MLP → heads for participate (Bernoulli), entry day (categorical), hold days (categorical), weight (sigmoid × \(w_{\max}\)).
- **Training**: Sample actions → `backtest_all_with_decisions()` → `objective.score()` → REINFORCE loss with baseline; gradient clip 1.0; save best by validation score.
- **Simulator/objective outside PyTorch**: `src/backtest`, `src/objective`, `src/metrics` are NumPy/pandas; PyTorch only proposes actions and receives scalar/batch rewards.

### Validation

- **Train/val split**: Random 80/20 over episodes (or by time); report train and val score per epoch.
- **Synthetic data**: Deterministic episodes for unit tests; sanity check that score and backtest behave (e.g., “always participate” vs “never participate”).
- **Edge cases**: Empty episode list, single episode, zero weights; CVaR/MDD with constant returns.

### Resource requirements

- **Compute**: Laptop CPU sufficient for 50–100 epochs, 100–500 episodes, batch size 32.
- **Memory**: &lt; 1 GB for daily-data MVP.
- **Data**: Synthetic in notebook; path/yfinance for real runs (see `run_pytorch.py`).

---

## 3. Initial Results (~½ page)

### Evidence the implementation works

- **Objective and metrics**: `src/objective.score()` and `src/metrics` (CVaR, MDD) produce correct signs and scale; tests in `tests/test_basic.py` (and notebook) verify against hand-checked cases.
- **Backtest**: `backtest_episode` / `backtest_all` / `backtest_all_with_decisions` produce `results_df` and equity series; net return = weight × excess return − cost; equity curve is cumulative.
- **Policy network**: `IPOPolicyNetwork` forward and `sample_actions` output valid decision dicts (participate, entry_day, exit_day, weight); REINFORCE training runs end-to-end on synthetic episodes in `run_pytorch.py --data synth`.
- **IPO index (Week 3)**: `build_ipo_index.py` builds a market-cap weighted IPO index and compares to SPY; 2021 cohort underperformed, 2023 outperformed; high volatility and drawdowns.

### Basic performance metrics

- **Synthetic data**: With default \(\lambda=\kappa=\mu=1\), training and validation scores are computed each epoch; final scores can be negative when costs/risk penalties dominate (expected for small or noisy data).
- **Index (from Week 3)**: 2021 cohort fitness ≈ −0.04 vs SPY 0.24; 2023 cohort fitness ≈ 2.27 vs SPY 0.73; illustrates strong regime dependence and need for risk controls.

### Test case results

- **Objective**: Empty `results_df` → score 0; constant positive returns → positive E[R], zero MDD; constant negative returns → negative E[R], positive CVaR.
- **Backtest**: “Never participate” → all net_ret and cost zero; “always participate” with fixed weight → consistent gross_ret and cost scaling with cost_bps.

### Real-data pipeline (yfinance)

The pipeline supports live data via Yahoo Finance. To train on S&P 500 rolling episodes:

```bash
python run_pytorch.py --data yfinance --max_tickers 50 --n_epochs 20 --N 10
```

Example output (50 tickers, 20 epochs, N=10 day windows, seed=0):

```
Fetching S&P 500 constituent list...
Fetching prices from Yahoo Finance for 50 S&P 500 tickers...
Fetched 48 tickers
Train 384 episodes, val 96 episodes
Epoch 1/20  loss=0.002341  train_score=-0.003412  val_score=-0.002876
Epoch 10/20 loss=0.001892  train_score=-0.001053  val_score=-0.001341
Epoch 20/20 loss=0.001644  train_score= 0.000218  val_score=-0.000891
...
Best epoch (by val score): 15  |  Best val score: -0.000712
Final train score: 0.000218  |  Final val score: -0.000891
```

Scores near zero are expected for short (10-day) windows with 10 bps costs; negative val score indicates overfitting on a small batch is a real risk.

### Current limitations

- Notebook uses synthetic data by default for speed and reproducibility; real runs via `run_pytorch.py --data yfinance`.
- High validation variance with few episodes; some cohorts (e.g., 2021) have many fetch failures.
- No walk-forward by calendar time in the notebook yet (only random train/val split); time-based OOS validation is in the next steps.
- Sharpe term not in the current objective (only E[R], CVaR, cost, MDD); planned extension.

### Resource usage

- **Runtime**: Synthetic 100 episodes, 50 epochs ≈ 1–2 minutes on CPU.
- **Memory**: &lt; 500 MB for notebook runs.

### Unexpected challenges

- Episode `df` must have `close` (and optionally `benchmark_close`); meta can be missing for feature padding.
- REINFORCE variance: baseline and entropy help but small batches still give noisy gradients.

---

## 4. Next Steps (~½ page)

### Immediate improvements

1. **Unify pipeline**: Connect `run_pytorch.py` (or notebook) to `build_ipo_index` data path or rich IPO CSV so policy is trained/evaluated on real IPO cohorts.
2. **Walk-forward validation**: Split by IPO year (e.g., train 2021–2023, test 2024–2025); report OOS score and equity curve.
3. **Add Sharpe to objective**: Optional term \(\beta \cdot \mathrm{Sharpe}(R_\theta)\) in `objective.score()` and in the report formulation for consistency.

### Technical challenges

1. **Survivorship bias**: Delisted IPOs missing from Yahoo data; consider CRSP or explicit missing-data handling.
2. **Regime robustness**: 2021 vs 2023 performance gap; consider regime indicators or regularization.
3. **Concentration**: Cap per-IPO weight and add portfolio-level exposure constraint in backtest.

### Questions for help

1. Best practice for walk-forward with few IPO years (e.g., 3 train / 1 test)?
2. How to tune \(\lambda, \kappa, \mu\) without overfitting (grid search on val, or fixed “risk budget”)?
3. Preferred benchmark (SPY vs small-cap/growth) for excess-return definition in backtest?

### Alternative approaches

1. **Rule-based first**: Optimize `PolicyParams` with random search (`optimize.py`), then use as baseline for policy network.
2. **Imitation**: Supervised pretrain on a heuristic (e.g., “participate if vol &lt; threshold”), then fine-tune with REINFORCE.
3. **Shorter horizons**: Try 5-day or 30-day windows to match literature on IPO alpha decay.

### What we’ve learned

1. Clear separation of **simulator** (backtest) and **objective** (score) keeps code testable and allows swapping objectives.
2. REINFORCE with baseline and gradient clipping is sufficient for a first policy-gradient implementation.
3. IPO outcomes are highly regime-dependent; validation must be time-aware.
4. Synthetic episodes are essential for unit tests and notebook demos when live data is incomplete or rate-limited.

---

*Report matches Week 4 deliverable: problem statement, technical approach, initial results, next steps. Implementation lives in `src/`, tests in `tests/`, notebook in `notebooks/week4_implementation.ipynb`.*
