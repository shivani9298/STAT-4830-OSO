# IPO Trading Strategy Optimization

A complete trading strategy optimization framework for IPO retail participation with risk-aware objectives.

## Overview

This project implements a constrained trading optimization system that learns/optimizes an event-conditioned retail IPO trading policy to maximize risk-adjusted PnL under allocation uncertainty and realistic trading frictions.

## Project Structure

```
.
├── README.md
├── report.md                    # Week 4 report (problem, approach, results, next steps)
├── notebooks/
│   └── week4_implementation.ipynb  # Week 4 working implementation & validation
├── src/
│   ├── model.py           # Core optimization interface (re-exports policy, objective, training)
│   ├── data.py            # Data loading and episode building (Person A)
│   ├── policy.py          # Rule-based trading logic (Person B)
│   ├── backtest.py        # Backtest engine (Person C)
│   ├── metrics.py         # Performance metrics (Person C)
│   ├── objective.py       # Objective function (Person D)
│   ├── optimize.py        # Random search and baselines (Person D)
│   ├── logging_utils.py   # Trial logging (Person D)
│   ├── features.py        # Feature extraction for policy network
│   ├── policy_network.py  # PyTorch contextual bandit policy
│   └── train_policy.py    # REINFORCE training (SGD/Adam, validation)
├── tests/
│   ├── test_basic.py      # Basic validation (objective, metrics, backtest)
│   └── ...                # Other tests
├── docs/
│   ├── COURSE_CONCEPTS.md
│   ├── llm_exploration/
│   │   └── week4_log.md   # Week 4 AI conversation log
│   └── development_log.md # Progress & decisions
├── results/               # Output artifacts (generated)
├── run_week3.py           # CLI: random search
├── run_pytorch.py         # CLI: PyTorch policy training
└── generate_report.py    # Generate markdown reports
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Random search (rule-based policy)

### Synthetic Data (Testing)

```bash
python run_week3.py \
  --data synth \
  --N 10 \
  --trials 500 \
  --seed 0 \
  --lam 1.0 --alpha 0.9 --kappa 1.0 --mu 1.0 --cost_bps 10
```

### PyTorch policy (REINFORCE + Adam)

Uses course concepts: autodiff, SGD/Adam, step-size schedules, validation.

```bash
python run_pytorch.py --data synth --n_epochs 50 --lr 1e-3 --batch_size 32 --val_frac 0.2
```

Trained policy is saved to `results/policy_network.pt`. See `docs/COURSE_CONCEPTS.md` for how course lectures map to this code.

### Real Data (Archive-3)

```bash
python run_week3.py \
  --data path \
  --meta_csv archive-3/ipo_clean_2010_2018.csv \
  --N 10 \
  --trials 500 \
  --seed 0 \
  --lam 1.0 --alpha 0.9 --kappa 1.0 --mu 1.0 --cost_bps 10
```

## Command-Line Arguments

### Data Arguments
- `--data`: Data source (`synth` or `path`)
- `--prices_dir`: Directory containing price CSV files (optional, generates synthetic if not provided)
- `--meta_csv`: Path to IPO metadata CSV file
- `--N`: Number of days in episode window

### Optimization Arguments
- `--trials`: Number of optimization trials
- `--seed`: Random seed for reproducibility

### Objective Function Arguments
- `--lam`: CVaR penalty weight (λ)
- `--alpha`: CVaR confidence level (α)
- `--kappa`: Cost penalty weight (κ)
- `--mu`: Maximum drawdown penalty weight (μ)
- `--cost_bps`: Transaction cost in basis points

### Output Arguments
- `--out_dir`: Output directory (default: `results/`)

## Output Artifacts

The optimization produces three artifacts in the output directory:

1. **`best_params.json`**: Best policy parameters found
2. **`trials.jsonl`**: All trial records (one JSON object per line)
3. **`results.csv`**: Backtest results DataFrame with columns:
   - `ticker`, `entry_day`, `exit_day`, `weight`
   - `entry_px`, `exit_px`, `gross_ret`, `cost`, `net_ret`, `pnl`

### Generate Report

After running optimization, generate a markdown report:

```bash
python3 generate_report.py
```

This creates `results/report.md` with detailed analysis and interpretations.

## Objective Function

The optimization maximizes:

```
Score = E[R] - λ·CVaR_α - κ·E[Cost] - μ·MDD
```

Where:
- `E[R]`: Expected return (mean of net returns)
- `CVaR_α`: Conditional Value at Risk at confidence level α
- `E[Cost]`: Expected transaction cost
- `MDD`: Maximum drawdown

## Policy Parameters

The policy optimizes:
- `participate_threshold`: Threshold for participation decision
- `entry_day`: Entry day (0 = day0, 1 = day1, etc.)
- `hold_k`: Hold period in days
- `w_max`: Maximum position weight
- `raw_weight`: Base position size
- `use_volume_cap`: Whether to use volume-based position capping
- `vol_cap_mult`: Volume cap multiplier

## Baselines

The system includes several baselines:
- `always_skip`: Never participate
- `always_participate`: Always participate with fixed weight
- `fixed_hold_k`: Fixed hold period strategies

## Data Format

### IPO Metadata CSV
Required columns:
- `ticker` (or `Symbol`): Stock ticker symbol
- `ipo_date` (or `Date Priced`): IPO date

### Price Data CSV
Each ticker CSV file should contain:
- `date`: Trading date
- `close`: Closing price (required)
- `volume`: Trading volume (optional)

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Team Responsibilities

- **Person A (Shivani)**: Data pipeline (`src/data.py` + tests)
- **Person B (Olivia)**: Policy module (`src/policy.py` + tests)
- **Person C (Aaron)**: Backtest + Metrics (`src/backtest.py`, `src/metrics.py` + tests)
- **Person D (Oceana)**: Objective + Optimizer + Logging + CLI (`src/objective.py`, `src/optimize.py`, `src/logging_utils.py`, `run_week3.py` + tests)

## License

This project is for educational purposes.
