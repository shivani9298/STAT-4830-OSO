# IPO Portfolio Optimization — Project Report

**Course:** STAT 4830 · **Date:** April 10, 2026 · **Repository:** `STAT-4830-OSO`

---

## 1. Problem Statement

### What we are optimizing

We learn **daily portfolio weights** on a simplex between (i) a **broad market** sleeve (CRSP-based blend of S&P 500 and Dow proxies, e.g. SPY/DIA) and (ii) an **IPO** sleeve. In **sector mode** (our main setup), GICS-style sector labels group IPOs into baskets; a shared recurrent or attention encoder feeds **one two-way softmax head per sector** (market vs that sector’s IPO basket), and sleeve returns are aggregated for training and evaluation. The trainable mapping is from rolling windows of past returns/features **\(X\)** to nonnegative weights **\(w\)** summing to one.

### Why this matters

IPOs differ systematically from the broad market (volatility, horizon effects, information asymmetry). A learned allocator that reacts to recent history can tilt exposure toward or away from IPO risk in a disciplined way compared with fixed mixes—relevant for research and for communicating **position scale** and policy-style rules to a retail audience.

### How we measure success

- **Training:** minimize a **combined differentiable loss** (mean return terms, CVaR-style tail penalty, volatility and diversification terms, etc.—see `src/losses.py`).
- **Validation:** same combined loss for early stopping and LR schedules; we have extended the stack with optional **path-level metrics** on chronological validation returns (compound return, Sharpe, Sortino, max drawdown) and a **retail composite** objective for checkpoint selection when enabled.
- **Reporting:** validation cumulative wealth vs a **50/50 market/IPO** benchmark, per-sector Sharpes where applicable, and exported weights/JSON for audit.

**Important caveat:** `val_loss` is **not** the same as out-of-sample Sharpe. A model with strong `val_loss` can still underperform a fixed 50/50 in live-like conditions if loss weights are miscalibrated—so we treat path metrics and holdout behavior as essential, not just loss.

### Constraints

- **Long-only, fully invested:** softmax over two assets per head (or multi-sector simplex in alternate setups).
- **Data and compute:** WRDS access; sector index construction is CPU-heavy; full-sample runs can take tens of minutes.
- **Temporal discipline:** rolling splits with embargo so train/val/test windows do not leak overlapping input spans.

### Data required

- **SDC / LSEG-style IPO deals** for offer dates and identifiers.
- **CRSP** daily prices and shares for return and market-cap-weighted IPO baskets.
- **Sector labels** (e.g. Compustat GICS, or alternatives such as Yahoo/CCM chain—see `docs/SECTOR_CCM_WORKFLOW.md`).
- **Sample window:** we **shortened history to start at 2010** after experiments with very long history (e.g. from 1970) produced misleading or unstable behavior; the current defaults align with 2010–2024-style panels.

### What could go wrong

Survivorship and listing bias; regime change; overfitting to validation; misalignment between **loss** and **economic** performance; heavy tails in IPO returns; and operational risk (WRDS quotas, long transformer runs).


## 2. Technical Approach

### Mathematical formulation (sketch)

Let **\(r_t\)** be realized returns for market and IPO sleeves, and **\(w_t\)** be model weights. Portfolio return is **\(\pi_t = w_t^\top r_t\)**. The loss combines:

- **Mean / growth terms** (level and optional **log-growth**-style proxies),
- **Tail risk (CVaR)** and **volatility** penalties,
- Optional **diversification / min-weight** and **turnover / path** terms,

with configurable **\(\lambda\)** weights. Constraints are enforced by architecture (softmax outputs) and by penalty terms.

**Recent extensions** (where wired in the codebase): `lambda_log_return`, **segment auxiliary loss** (`train_segment_len`, `lambda_segment_log`) to nudge longer-horizon behavior without abandoning batch training; and **selection metrics** tied to **path statistics** on validation returns (`val_sharpe`, `val_sortino`, `val_retail_composite`, etc.).

### Algorithm choices

- **GRU and LSTM** (`AllocatorNet` + sector heads): strong default for sequential windows; we compared them directly on the same splits.
- **Transformer** (sector multi-head): promising in smaller experiments but **slow** and, with extended **1970s-onward** data, **degraded** in ways we did not trust; we **deprioritized** transformer on the long-history path and focused on **GRU/LSTM** with cleaner 2010+ panels.
- **Gradient clipping:** successful stabilizer for training.
- **Context window:** we tried multiple window lengths; **effects were modest** relative to data and objective choices.
- **Sectors:** **helped somewhat** versus a single IPO index by exposing separate heads per basket.

### PyTorch strategy

Models in `src/model.py`; training loops in `src/train.py` with Adam, optional **ReduceLROnPlateau** / cosine / exponential schedules, early stopping on validation loss or on a **path-based selection scalar** when configured. **Gradient clipping** is supported. Sector batches use `combined_loss_sector_heads`.

### Validation

Time-based splits; embargo between train and validation; optional **walk-forward** evaluation via `scripts/walk_forward_train_wrds.py` (anchors for test start, validation window before anchor, metrics copied per run).

### Resources

WRDS + local CPU/GPU PyTorch; sector IPO index construction is **CPU-heavy** (minutes per full refresh). Transformers are the main **latency** risk.

---

## 3. Initial Results

### Evidence the pipeline works

End-to-end WRDS pulls, rolling tensors, training, export of weights and **validation** cumulative-return plots under `figures/ipo_optimizer/<model>/`. **GRU vs LSTM** comparisons (loss curves and side-by-side validation figures) are generated from saved histories without retraining (`scripts/plot_ipo_gru_lstm_comparison.py`).

### Basic metrics

On recent runs with tuned configs, sector-level **validation Sharpes** vary by sleeve (some strong, some weak); **Materials**-like thin sleeves can look poor—consistent with data sparsity. **Train/val loss gaps** suggest **overfitting risk**, motivating regularization and walk-forward work.

### Tests

`tests/test_loss_balance.py` passes. Legacy `tests/test_basic.py` targets an older API and is **not** aligned with the current `src/model.py` (known technical debt).

### Limitations

- Log-utility and segment losses are **not** full trajectory optimizers of terminal wealth.
- **Val loss** ≠ economic dominance vs 50/50.
- **Dropout** in config may not fully propagate into all RNN depths in every code path—worth verifying when raising dropout.
- **Weight decay** at 1e-5 may be weak for parameter count; increasing needs care (Adam vs AdamW semantics).

### Unexpected challenges

Transformer runtime; long-history **1970** experiments behaving badly enough to **discard** that design choice; reconciling **early transformer optimism** with **larger-sample** reality.

---

## 4. Next Steps

### Immediate improvements

1. **Regularization sweep:** increase **dropout** (e.g. 0.1→0.2) and **weight decay** (e.g. 1e-5→1e-4) where wired; consider smaller **hidden size** (64→32) if train/val gap remains.
2. **Selection criterion:** use **path metrics** / `val_retail_composite` consistently for checkpointing when the goal is economic performance, not raw loss alone.
3. **Walk-forward:** run `scripts/walk_forward_train_wrds.py` on a small anchor grid to assess **regime stability** (accept **~K×** training cost).

### Technical challenges

Ensuring **all** hyperparameters in JSON actually reach `build_model` / RNN dropout; aligning **AdamW** if we lean on weight decay; managing **compute** for multi-anchor walk-forward.

### Alternatives

Ensemble across folds; simpler **linear** or **risk-parity** baselines for ablation; lighter **temporal convolution** instead of full transformer.

### What we have learned

**Data era matters** as much as architecture; **GRU/LSTM** remain dependable baselines; **sectors** add structure; **grad clipping** matters; and **metrics for selection** must match the **question** (trading vs statistical loss).


