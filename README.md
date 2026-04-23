# IPO Portfolio Optimizer (OGD baselines + learned allocators)

**STAT 4830 | Spring 2026 | University of Pennsylvania**

This repo contains **two related things**:

- **Precomputed OGD (online) allocation benchmark** time series in `results/ipo_180day_mcap_returns.csv` (columns like `OGD_Portfolio` vs `Equal_Weight`). The “Key Results” table below is from this file (it is **not** the WRDS PyTorch run unless you re-generate the CSV with the OGD pipeline you used for the diagram).
- **WRDS end-to-end training** for a **GRU / LSTM / Transformer** daily allocator: `scripts/run_ipo_optimizer_wrds.py` trains on rolling windows, exports weights to `results/ipo_optimizer_weights.csv`, and writes plots under `figures/ipo_optimizer/<model>/` (enable extras with `IPO_SAVE_LOSS_PLOTS=1`).

**Data**: IPO + market return construction uses **WRDS** (SDC + CRSP) in the training scripts; the benchmark CSV is treated as a fixed artifact in-repo for the diagram-aligned headline numbers.

## Key Results (OGD “learned” vs 50/50 baseline)

These numbers match the project diagram / `results/ipo_180day_mcap_returns.csv` for **2022-09-30 → 2024-01-22** (271 trading days): **Learned = `OGD_Portfolio`**, **50/50 = `Equal_Weight`**, with single-asset sleeves **SPY** and **IPO** for context.

| Strategy | Total Return | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|----------|--------------|-------------|----------|--------|--------------|
| **Learned (OGD)** | **49.25%** | **45.12%** | 16.82% | **2.30** | **-10.47%** |
| Equal 50/50 | 82.90% | 75.32% | 30.86% | 1.97 | -25.99% |
| SPY only | 27.07% | 24.95% | 14.36% | 1.62 | -9.97% |
| IPO only | 145.92% | 130.89% | 54.95% | 1.79 | -41.13% |

Over that window the learned allocator keeps **~11%** average weight in the IPO sleeve (vs **50%** for the static baseline): lower cumulative return than 50/50, but **meaningfully better tail risk** (max drawdown about **-10%** vs **-26%**).

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

**Using uv (recommended, including VM/GPU)**

Install uv if needed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
uv venv .venv
source .venv/bin/activate
```

Install **PyTorch** for your CUDA build from the official index (pick the URL that matches your driver; examples below—see [pytorch.org](https://pytorch.org) for current wheels):

```bash
# Example: CUDA 12.4 wheels (adjust cu118/cu121/cu124/cpu per your VM)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

WRDS-friendly stack (pandas/sqlalchemy pins that work with `wrds`):

```bash
uv pip install pandas==2.1.4 sqlalchemy==1.4.54 wrds==3.1.6 python-dotenv matplotlib numpy
```

**Using pip only**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# Then install torch from pytorch.org, then:
pip install pandas==2.1.4 sqlalchemy==1.4.54 wrds==3.1.6 python-dotenv matplotlib numpy
```

### 2. Run the Optimizer

```bash
python scripts/run_ipo_optimizer_wrds.py
```

This script:

1. Connects to WRDS and loads IPO data from SDC + CRSP
2. Builds the IPO index and market returns
3. Trains a learned allocator (`model_type` is typically **`gru`**, but can be **`lstm` / `transformer` / `hybrid`** via `results/ipo_optimizer_best_config.json`, a local override JSON, or `IPO_MODEL_TYPE=...` — see the header of `scripts/run_ipo_optimizer_wrds.py`)
4. Exports weights to `results/ipo_optimizer_weights.csv` and a summary to `results/ipo_optimizer_summary.txt`
5. Saves figures under `figures/ipo_optimizer/<model>/` (see `IPO_SAVE_LOSS_PLOTS` in `run_ipo_optimizer_wrds.py`)

**Runtime**: ~2–3 minutes.

### 3. Hyperparameter Tuning (Optional)

```bash
python notebooks/tune_hyperparameters_wrds.py
```

Grid search over window length, volatility penalties, CVaR, etc. Saves the best config to `results/ipo_optimizer_best_config.json`; `scripts/run_ipo_optimizer_wrds.py` will use it on the next run.

**Runtime**: ~1–3 hours depending on grid size.

### 4. Jupyter Notebook

```bash
jupyter notebook notebooks/week4_implementation.ipynb
```

Step-by-step notebook with problem setup, implementation, and validation.

---

## Technical Approach

### Model Architecture

#### Learned neural allocator (default training path: GRU)

- **Input**: Rolling window of past returns (e.g., 84–252 days × features, configurable)
- **GRU (or LSTM / Transformer / hybrid, configurable)** → last hidden state → **MLP** → **softmax** → weights on **[market, IPO]**
- **Output constraints**: long-only, fully invested (simplex)
- **Training loop**: `src/train.py` optimizes a multi-term portfolio objective in `src/losses.py` (return, tail risk, turnover/path penalties, etc.)

<img width="476" height="209" alt="image" src="https://github.com/user-attachments/assets/0b7a71dd-3994-4263-9c6f-3381f50a23f9" />

#### OGD baseline (diagram / `ipo_180day_mcap_returns.csv`)

- The repo also includes an **online convex optimization** style allocator (see `docs/development_log.md` and `tests/test_basic.py` references to `OnlineOGDAllocator`).
- The headline **OGD vs 50/50** table above is taken directly from the **precomputed** daily return series in `results/ipo_180day_mcap_returns.csv`.


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
├── report.md                         # Week 4 report (problem, approach, results)
├── self_critique.md                 # OODA self-assessment
├── scripts/
│   ├── run_ipo_optimizer_wrds.py   # Main WRDS training + export (GRU/LSTM/TF/...)
│   └── ...                         # Plotting + analysis CLIs
├── notebooks/
│   ├── tune_hyperparameters_wrds.py  # Hyperparameter grid search
│   ├── week4_implementation.ipynb   # Main notebook
│   ├── ipo_optimizer_2025_wrds.ipynb
│   └── test_wrds.ipynb
├── src/
│   ├── model.py                     # GRU/MLP allocator
│   ├── losses.py                    # Differentiable loss components
│   ├── train.py                     # Training loop
│   ├── export.py                    # Predict, stats, export
│   ├── data_layer.py                # Rolling windows, splits
│   ├── wrds_data.py                 # WRDS data loading
│   ├── utils.py                     # Misc helpers (includes OGD-related utilities)
│   └── policy_layer.py              # Position scaling, policy rules
├── results/
│   ├── ipo_180day_mcap_returns.csv  # OGD vs baselines (diagram headline table)
│   ├── ipo_optimizer_weights.csv    # Learned model daily weights (WRDS run)
│   ├── ipo_optimizer_summary.txt    # Learned model summary
│   └── ipo_optimizer_best_config.json  # Best tuning config
├── figures/
│   ├── ipo_optimizer/<model>/       # Plots for WRDS training runs
│   └── ...                          # Validation analysis plots (optional scripts)
├── docs/
│   └── ipo_concentration_diagnosis.md
└── tests/
    └── test_basic.py
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
5. **Two “headline” result sources** – The README’s OGD table is from `results/ipo_180day_mcap_returns.csv` (a checked-in series); WRDS run metrics come from `results/ipo_optimizer_summary.txt` and may differ by window + objective.

See `report.md` and `self_critique.md` for details.

---

## License

Developed for STAT 4830 at the University of Pennsylvania.
