# IPO Portfolio Optimizer (OGD baselines + learned allocators)

**STAT 4830 | Spring 2026 | University of Pennsylvania**

This repo contains **three related things**:

- **Precomputed OGD (online) allocation benchmark** time series in `results/recent/ipo_180day_mcap_returns.csv` (columns like `OGD_Portfolio` vs `Equal_Weight`). The OGD table below covers the full file range.
- **WRDS end-to-end training** for a **GRU / LSTM / Transformer** daily allocator: `scripts/run_ipo_optimizer_wrds.py` trains on rolling windows, exports weights to `results/recent/ipo_optimizer_weights.csv`, and writes plots under `figures/recent/ipo_optimizer/<model>/` (enable extras with `IPO_SAVE_LOSS_PLOTS=1`).
- **Compound-only loss experiments**: offline and online models trained with a pure log-growth objective (no CVaR / vol / turnover penalties), revealing near-100% IPO allocation policies. Scripts in `scripts/compound_only/`, results in `results/`, figures in `figures/experiments/`.

**Data**: IPO + market return construction uses **WRDS** (SDC + CRSP) in the training scripts; the benchmark CSV is treated as a fixed artifact in-repo.

## Key Results

### OGD baseline vs 50/50 (`results/recent/ipo_180day_mcap_returns.csv`)

Full file: **2020-07-06 → 2025-01-14** (1 082 trading days).

| Strategy | Total Return | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|----------|--------------|-------------|----------|--------|--------------|
| **Learned (OGD)** | **193.46%** | **28.50%** | 20.06% | **1.42** | **-26.17%** |
| Equal 50/50 | 577.70% | 56.15% | 35.03% | 1.60 | -51.33% |
| SPY only | 86.12% | 15.57% | 16.11% | 0.97 | -24.50% |
| IPO only | 1699.28% | 96.03% | 61.36% | 1.56 | -73.12% |

The OGD allocator keeps conservative IPO exposure, trading high absolute return for substantially lower drawdown and volatility vs the 50/50 benchmark.

### Online compound-only GRU (`results/recent/ipo_optimizer_summary.txt`)

Online model trained with pure compound-growth objective (no regularization), evaluated on **2021-07-19 → 2024-12-31** (870 trading days):

| Strategy | Total Return | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|----------|--------------|-------------|----------|--------|--------------|
| **Online compound (net)** | **85.94%** | **18.69%** | 19.24% | **0.99** | **-30.97%** |
| Offline static (same dates) | 108.98% | 22.59% | 18.31% | 1.20 | -26.07% |
| Best static (train) | 165.50% | 30.97% | 21.19% | 1.38 | -29.88% |
| Equal 50/50 | 298.22% | 46.49% | 27.46% | 1.52 | -36.30% |
| Market only | 45.98% | 11.02% | 16.04% | 0.73 | -23.69% |
| IPO only | 860.47% | 86.84% | 45.22% | 1.59 | -52.92% |

Removing all regularization collapses the policy toward near-100% IPO allocation, producing high returns with high volatility. The offline static model consistently outperforms the online version on this window, suggesting online updates add noise rather than signal under this objective. See `figures/online_evaluation/` and `figures/experiments/` for allocation trajectory and return comparison plots.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [**uv**](https://docs.astral.sh/uv/) (recommended) or `pip`
- **WRDS account** (CRSP + SDC)
- `WRDS_USERNAME` and `WRDS_PASSWORD` environment variables (or interactive login)

### 1. Clone and Install

```bash
git clone https://github.com/shivani9298/STAT-4830-OSO.git
cd STAT-4830-OSO
```

**Using uv (recommended)**

Install uv if needed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
uv venv .venv
source .venv/bin/activate
```

Install **PyTorch** (choose one command for your platform):

```bash
# macOS (Apple Silicon or Intel)
uv pip install torch
```

```bash
# Linux/Windows + NVIDIA CUDA 12.4 (example)
# Use pytorch.org to pick the exact CUDA index URL for your machine.
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Install WRDS + data stack:

```bash
uv pip install -r requirements.txt
```

**Using pip only**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install **PyTorch** (choose one command for your platform):

```bash
# macOS (Apple Silicon or Intel)
pip install torch
```

```bash
# Linux/Windows + NVIDIA CUDA 12.4 (example)
# Use pytorch.org to pick the exact CUDA index URL for your machine.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Install WRDS + data stack:

```bash
pip install -r requirements.txt
```

Why this matters: `wrds==3.1.6` declares `sqlalchemy<2`, which conflicts with this repo's SQLAlchemy 2.x requirement in `src/wrds_data.py`.

### 2. Run the Optimizer

```bash
python scripts/run_ipo_optimizer_wrds.py
```

This script:

1. Connects to WRDS and loads IPO data from SDC + CRSP
2. Builds the IPO index and market returns
3. Trains a learned allocator (`model_type` is typically **`gru`**, but can be **`lstm` / `transformer` / `hybrid`** via `results/recent/ipo_optimizer_best_config.json`, a local override JSON, or `IPO_MODEL_TYPE=...` — see the header of `scripts/run_ipo_optimizer_wrds.py`)
4. Exports weights to `results/recent/ipo_optimizer_weights.csv` and a summary to `results/recent/ipo_optimizer_summary.txt`
5. Saves figures under `figures/recent/ipo_optimizer/<model>/` (see `IPO_SAVE_LOSS_PLOTS` in `scripts/run_ipo_optimizer_wrds.py`)

**Runtime**: ~2–3 minutes.

### 3. Run Compound-Only Loss Experiments (Optional)

```bash
python scripts/compound_only/run_sensitivity_test_compound_only.py
python scripts/compound_only/experiment_loss_subset_offline.py
```

These scripts train offline/online models with the loss restricted to the log-growth (compound return) term only — zeroing CVaR, vol, and turnover penalties. Results land in `results/` and figures in `figures/experiments/`.

### 4. Interactive Demo (Streamlit)

```bash
pip install streamlit plotly  # if not already installed
streamlit run scripts/demo.py
```

Launches a browser UI with cumulative performance charts, allocation-weight evolution, model vs benchmark comparisons, and scenario toggles (Historical / Asset Rally / Risk-Off).

### 5. Hyperparameter Tuning (Optional)

```bash
python notebooks/tune_hyperparameters_wrds.py
```

Grid search over window length, volatility penalties, CVaR, etc. Saves the best config to `results/recent/ipo_optimizer_best_config.json`; `scripts/run_ipo_optimizer_wrds.py` will use it on the next run.

**Runtime**: ~1–3 hours depending on grid size.

---

## Technical Approach

### Model Architecture

#### Learned neural allocator (default training path: GRU)

- **Input**: Rolling window of past returns (e.g., 84–252 days × features, configurable)
- **GRU (or LSTM / Transformer / hybrid, configurable)** → last hidden state → **MLP** → **softmax** → weights on **[market, IPO]**
- **Output constraints**: long-only, fully invested (simplex)
- **Training loop**: `src/train.py` optimizes a multi-term portfolio objective in `src/losses.py` (return, tail risk, turnover/path penalties, etc.)
- **Sector-head mode** (default in `scripts/run_ipo_optimizer_wrds.py`): shared encoder + one market-vs-sector-IPO head per sector using `src/multisector_data.py` and `src/multi_sector_setup.py`

<img width="476" height="209" alt="image" src="https://github.com/user-attachments/assets/0b7a71dd-3994-4263-9c6f-3381f50a23f9" />

#### OGD baseline (`results/recent/ipo_180day_mcap_returns.csv`)

- Classic **online gradient descent** allocator; weights updated each day via a projected-gradient step on the portfolio return.
- The OGD table above is computed directly from the precomputed daily return series (2020-07-06 → 2025-01-14).

#### Compound-only loss experiments (`scripts/compound_only/`)

- **Offline**: the multi-term objective is stripped to `log(1 + r)` only (no CVaR, no vol penalty, no turnover). The offline GRU converges to a near-constant ~100% IPO weight — the unconstrained optimal policy when only log-growth matters.
- **Online**: the same objective applied in an online update loop. The online model tracks the IPO sleeve aggressively but is outperformed by the static offline model, suggesting online re-fitting adds noise on this window.
- Loss-subset ablations (with/without diversification penalty, with/without RL-style discounted log-return) are in `results/loss_subset_*` and `figures/experiments/loss_subset/`.


### Data Sources

| Source        | Content                                   |
|---------------|-------------------------------------------|
| **SDC**       | IPO dates (`sdc.wrds_ni_details`)         |
| **CRSP**      | Daily prices, shares (split-adjusted)     |
| **CRSP SPY/DIA** | Market returns (82% / 18%)            |

- IPO index: market-cap weighted, 180 trading days per IPO
- Date range: 2020–2024 (CRSP lag)

---

## Repository Structure

```
.
├── README.md                         # This file
├── scripts/
│   ├── run_ipo_optimizer_wrds.py    # Main WRDS training + export (GRU/LSTM/TF/...)
│   ├── demo.py                      # Streamlit interactive demo
│   ├── compound_only/               # Compound-growth-only loss experiments
│   │   ├── experiment_loss_subset_offline.py
│   │   ├── run_sensitivity_test_compound_only.py
│   │   ├── run_sensitivity_test_offline.py
│   │   └── plot_online_compound_vs_offline_compound.py
│   └── ...                          # Plotting + analysis CLIs
├── notebooks/
│   ├── tune_hyperparameters_wrds.py # Hyperparameter grid search
│   ├── week4_implementation.ipynb
│   ├── final_project_demo_colab.ipynb
│   └── test_wrds.ipynb
├── src/
│   ├── model.py                     # GRU/LSTM/Transformer allocators
│   ├── losses.py                    # Differentiable loss components
│   ├── train.py                     # Training loop
│   ├── export.py                    # Predict, stats, export
│   ├── data_layer.py                # Rolling windows, splits
│   ├── multisector_data.py          # Sector IPO basket construction
│   ├── multi_sector_setup.py        # Multi-asset windowing + exports
│   ├── wrds_data.py                 # WRDS data loading
│   └── policy_layer.py              # Position scaling, policy rules
├── results/
│   ├── recent/                      # Latest WRDS run artifacts
│   ├── loss_subset_*/               # Compound-only loss ablation results (CSV + TXT)
│   ├── sensitivity_test_*/          # Sensitivity analysis (compound-only vs full objective)
│   ├── online_*/                    # Online compound model results
│   └── older/                       # Historical artifacts grouped by commit hash
├── figures/
│   ├── experiments/
│   │   ├── loss_subset/             # Training curves + return plots for loss ablations
│   │   └── sensitivity/             # Hyperparameter sensitivity heatmaps
│   ├── online_evaluation/           # Online compound return trajectories + IPO weights
│   ├── architecture/                # Pipeline diagrams (offline GRU, online GRU)
│   ├── recent/                      # Latest WRDS run figures
│   └── older/                       # Historical figures grouped by commit hash
├── docs/
│   ├── reports/                     # Date-first reports (see 2026-05-05-final-written-report.md)
│   ├── self_critiques/              # Date-first self-critiques
│   └── slides/
└── tests/
    └── test_*.py
```

---

## Key Parameters

| Parameter            | Default | Description                          |
|----------------------|---------|--------------------------------------|
| `model_type`         | `gru`   | `gru` / `lstm` / `transformer` / `hybrid` (see `src/model.py` + `scripts/run_ipo_optimizer_wrds.py`) |
| `window_len`         | 126     | Days of history per prediction        |
| `val_frac`           | 0.2     | Fraction of dates for validation     |
| `lambda_vol_excess`  | 1.0     | Penalty when vol exceeds target       |
| `target_vol_annual`  | 0.25    | Target max annual vol (25%)           |
| `lambda_diversify`   | 0.0     | Diversification penalty (optional)    |
| `hidden_size`        | 64      | GRU hidden dimension                 |

---

## Known Limitations

1. **No true out-of-sample test** – Metrics are on the validation set; no held-out test period
2. **Stability / near-constant weights (sometimes)** – Depending on the objective/penalties, the learned allocator can become **nearly static** day-to-day; this is a behavior to check against baselines, not a guaranteed property.
3. **Survivorship bias** – IPO index excludes delisted stocks
4. **Turnover display** – Very small turnover (~1e-5) rounds to 0.0000 in the summary
5. **Two “headline” result sources** – The OGD table is from `results/recent/ipo_180day_mcap_returns.csv` (precomputed series); the online compound model table is from `results/recent/ipo_optimizer_summary.txt` (WRDS PyTorch run, different window and objective).
6. **Compound-only policy collapse** – With all regularization removed, the offline model learns near-100% IPO allocation. This is optimal under log-growth alone but ignores tail risk entirely.

See `docs/reports/2026-05-05-final-written-report.md` and `docs/self_critiques/` for full writeups.

---

## License

Developed for STAT 4830 at the University of Pennsylvania.
