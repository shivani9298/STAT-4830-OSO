# STAT 4830 Final Report (Practical Layout)

## 1) Executive Summary

- We built a risk-aware GRU allocator between market and IPO sleeves, then extended it to online updates with cadence/confidence gates.
- Offline GRU achieved strong risk-adjusted performance on the main validation period, but online gating did not beat static high-IPO benchmarks during IPO-led regimes.
- A 12-day validation window was too noisy for stable model selection; larger validation slices and explicit scheduler control were more reliable.

## 2) Data + Model Pipeline Diagram

- Offline pipeline figure: `figures/architecture/offline_gru_static_pipeline_visual.png`
- Online pipeline figure: `figures/architecture/online_gru_adaptive_pipeline_clean.png`

Interpretation:
- The same cleaned return features feed both modes.
- Offline mode trains once then holds policy fixed.
- Online mode inserts scheduled retraining + gate logic before applying updated weights.

## 3) Baseline Results Table (Model vs Market vs IPO vs 50/50)

Source summary: `results/ipo_optimizer_summary.txt`

| Strategy | Total Return | Ann. Return | Ann. Vol | Sharpe | Max DD |
|---|---:|---:|---:|---:|---:|
| Model | 87.52% | 97.97% | 19.26% | 3.65 | -7.20% |
| Market only | 19.40% | 21.24% | 12.25% | 1.63 | -7.89% |
| IPO only | 192.78% | 221.19% | 31.03% | 3.92 | -10.08% |
| Equal 50/50 | 88.52% | 99.11% | 19.39% | 3.65 | -7.20% |

### Architecture Testing (Transformer, GRU, LSTM)

From project presentation evidence (`docs/STAT 4830 Final Project-3.pdf`) and experiment artifacts:

| Architecture | Validation/Selection Evidence | Performance Signal | Decision |
|---|---|---|---|
| GRU | Stable loss behavior in core offline runs | Strong Sharpe with controlled drawdown in baseline table | Keep as default |
| LSTM | Competitive vs GRU under matched setup (`results/ipo_optimizer_gru_lstm_metrics_w126.txt`) | GRU 78.35% total / 3.47 Sharpe vs LSTM 87.63% / 3.55 | Competitive fallback |
| Transformer | Ablation sensitivity high (`results/wrds_transformer_ablation_2020_2024.json`) | Baseline transformer run had weaker positive losses; best run required specific regularization and still inconsistent | Do not use as primary |

Supporting figures for model-selection decision:
- `figures/ipo_optimizer_gru_vs_lstm_loss_w126.png`
- `figures/ipo_optimizer/comparison/gru_vs_lstm_val_loss.png`
- `figures/ipo_optimizer/comparison/gru_vs_lstm_validation_returns_side_by_side.png`
- `figures/online_evaluation/val_loss_hypothesis_experiments.png`

Interpretation:
- LSTM and GRU are close under controlled settings, but GRU remained the most robust default through the broader pipeline experiments.
- Transformer benefited from narrow hyperparameter pockets but showed higher fragility and complexity cost.

## 4) Online A/B Gate Results Figure

- Figure: `figures/online_evaluation/ab_update_gate_results.png`
- Data: `results/ab_update_gate_results.csv`

Interpretation:
- Cadence gate had higher return/Sharpe in this test matrix.
- Confidence gate reduced update count materially (lower accept rate), but did not outperform on net risk-adjusted return.

## 5) Return Trajectories (Cadence/Confidence/50-50 + Market + IPO)

- Figure: `figures/online_evaluation/online_return_trajectories_all_benchmarks.png`
- Paths: `results/online_path_cadence_lb504.csv`, `results/online_path_confidence_lb504.csv`

Final cumulative return readout:
- Cadence (net): 91.45%
- Confidence (net): 85.94%
- Static 50/50: 298.22%
- Market-only: 45.98%
- IPO-only: 860.47%

Interpretation:
- Gates changed update behavior but did not close the return gap to high-IPO static benchmarks in this regime.

## 6) IPO Weight Trajectory (Underweighting Evidence)

- Figure: `figures/online_evaluation/online_ipo_weight_trajectory.png`

Interpretation:
- Online policy average IPO weight remained low relative to static 50/50 during key periods.
- This exposure mismatch is a main driver of underperformance vs IPO-heavy benchmarks.

## 7) Validation-Loss Diagnostics (Hypothesis Experiment Chart)

- Figure: `figures/online_evaluation/val_loss_hypothesis_experiments.png`
- Table: `results/val_loss_hypothesis_experiments.csv`

Interpretation:
- Validation behavior is sensitive to split design (`val_frac`) and LR schedule choice.
- Constant-LR variants showed different end-state loss behavior from scheduled variants.
- Very small validation slices increased variance and degraded stability.

## 8) 12-Day Validation Window Case Study (IPO)

### IPO model case

Files:
- Loss plots:
  - `figures/ipo_optimizer_replots/loss_semilogy_gru_offline_true_val12_adaptive.png`
  - `figures/ipo_optimizer_replots/loss_linear_gru_offline_true_val12_adaptive.png`
  - `figures/ipo_optimizer_replots/loss_rolling_gru_offline_true_val12_adaptive.png`
- Returns:
  - `figures/ipo_optimizer_replots/returns_val12_adaptive_vs_5050.png`
- Weights:
  - `figures/ipo_optimizer_replots/ipo_weights_val12_adaptive.png`
- Summary:
  - `results/ipo_optimizer_summary_val12_adaptive.txt`

Key results:
- Model total return: -2.78%
- 50/50 total return: -3.22%
- Market-only: -2.77%
- IPO-only: -3.68%
- Average IPO weight: 1.36%

Interpretation:
- A 12-day validation window is highly fragile; conclusions are dominated by short-horizon noise.
- Adaptive LR was active in this run, but small validation horizon still produced unstable selection signals.

## 9) Commodities Case Study

Files:
- `results/commodity_optimizer_summary.txt`
- `figures/commodity_optimizer/gru/validation_returns_vs_equal_weight.png`
- `figures/commodity_optimizer/gru/weight_evolution_over_time.png`

Key results:
- Model total return: 58.43%
- Equal 50/50: 40.36%
- Delta total return vs 50/50: +18.07%

Interpretation:
- Dynamic allocation added value in commodities, indicating method portability when signal-to-noise and regime structure differ from IPO sleeve behavior.

## 10) Lessons Learned / Limitations

- Update gates control update frequency, not objective alignment; they cannot fix low-exposure policy bias alone.
- Validation split design (size and chronology) changes conclusions substantially.
- Scheduler behavior must be explicitly wired in training calls; otherwise runs may silently default to constant LR.
- Small validation windows (12 days) are useful for stress testing, not for primary model selection.
- Transformer complexity increased engineering/runtime cost without robust realized-path gains in this project regime.

## 11) Future Work

- Promote scheduler controls to first-class config in all training entrypoints.
- Use larger, walk-forward validation windows for selection (e.g., rolling 1-3 month validation blocks).
- Add richer IPO-entry features (issue-level metadata, regime indicators) to improve exposure targeting.
- Evaluate objective variants that better align with forward cumulative wealth (not just short-horizon validation loss deltas).
- Extend online gate criteria with forward-utility proxies and explicit exposure constraints/floors.
