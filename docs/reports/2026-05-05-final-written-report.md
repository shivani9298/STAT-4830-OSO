# STAT 4830 — Final Written Report

**Course:** STAT 4830 · **Repository:** `STAT-4830-OSO` · **Report date:** May 5, 2026

---

## 1. Executive summary

- **Offline GRU allocation** shows strong risk-adjusted validation behavior on WRDS panels, but **online** behavior remains sensitive to **sleeve exposure**: IPO-heavy benchmarks can dominate simple 50/50 or IPO-only paths when the learned weights **underweight** the IPO sleeve.
- **Architecture comparison** (GRU vs LSTM vs Transformer) favors **GRU/LSTM** for stability; the Transformer path is usable but **hyperparameter-sensitive** and was not adopted as the default production recipe (see §3 and project deck PDF).
- **Cross-asset portability is uneven**: dynamic weights add clearer value in **commodities** (vs 50/50) than in the IPO-online narrative alone; **bonds** remain a useful stress case with modest Sharpe in the exported summaries.

---

## 2. Data + model pipeline diagram

WRDS-backed daily panels feed rolling-window features into a recurrent **allocator** (`AllocatorNet`-style) trained with a **multi-term differentiable objective** (growth, tail risk, volatility/diversification, optional turnover/path terms—see `src/losses.py`). Offline training selects checkpoints via validation loss and/or path metrics; online deployment wraps the same backbone with **periodic retraining** and optional **update-gate** logic before exporting daily weights and realized portfolio returns.

**Figures (conceptual pipeline):**

| Mode | Figure |
|------|--------|
| Offline (single train / validation selection) | `figures/architecture/offline_gru_static_pipeline_visual.png` |
| Online (rolling schedule + updates) | `figures/architecture/online_gru_adaptive_pipeline_clean.png` |

**Interpretation:** Both modes share **feature construction** and **loss structure**; online mode adds **when** the model is refit and **which** checkpoint is live—so gaps between offline metrics and online paths can come from **update policy** as well as representation quality.

---

## 3. Baseline results table (model vs market vs IPO vs 50/50)

**Sources (summaries on disk):**

- IPO (offline GRU, sector multi-head): `results/recent/ipo_optimizer_weights_val_sector_mean.csv` + `results/recent/ipo_optimizer_returns_val.csv` (mean across 11 GICS sector heads, 2022-01-25 → 2024-12-31, 738 days — matches the final slides and Streamlit demo Tab 1)
- Bonds: `results/bonds/bond_optimizer_summary.txt`
- Commodities: `results/older/40539ec/commodity_optimizer_summary.txt`

### 3.1 Aggregated metrics

| Asset class | Strategy | Total return | Ann. return | Ann. vol | Sharpe | Max DD |
|-------------|----------|--------------|-------------|----------|--------|--------|
| IPO | Model (offline GRU, sector mean) | 49.25% | 45.12% | 16.82% | 2.30 | -10.47% |
| IPO | Market only | 27.07% | 24.95% | 14.36% | 1.62 | -9.97% |
| IPO | IPO only | 145.92% | 130.89% | 54.95% | 1.79 | -41.13% |
| IPO | Equal 50/50 | 82.90% | 75.32% | 30.86% | 1.97 | -25.99% |
| Bonds | Model | 18.16% | 5.84% | 11.66% | 0.55 | -18.13% |
| Bonds | Market only | 34.64% | n/a | n/a | 0.67 | n/a |
| Bonds | Bond only | -4.62% | n/a | n/a | -0.19 | n/a |
| Bonds | Equal 50/50 | 14.57% | n/a | n/a | 0.50 | n/a |
| Commodities | Model | 58.43% | 17.24% | 13.65% | 1.23 | -18.42% |
| Commodities | Market only | 35.10% | 10.96% | 16.77% | 0.70 | -21.21% |
| Commodities | Commodity only | 41.61% | 12.78% | 15.15% | 0.87 | -20.94% |
| Commodities | Equal 50/50 | 40.36% | 12.43% | 12.42% | 1.01 | -17.44% |

*Notes:* Bond summary exports omit some annualized fields in the text artifact—cells marked n/a reflect the source file. All figures are **artifact snapshots** and should be regenerated after reruns.

### 3.2 Model architecture testing (Transformer, GRU, LSTM)

**Primary narrative deck / write-up:** `docs/STAT 4830 Final Project-3.pdf` (architecture comparison slides and discussion).

**Supporting repo artifacts:**

| Architecture | Representative artifacts | Readout |
|--------------|-------------------------|---------|
| **GRU** | `results/ipo_optimizer_gru_lstm_metrics_w126.txt`; `figures/ipo_optimizer_gru_vs_lstm_loss_w126.png`; `figures/ipo_optimizer/comparison/gru_vs_lstm_validation_returns_side_by_side.png` | Stable training curves; strong validation Sharpe in standard windows—**default choice**. |
| **LSTM** | Same comparison bundle as GRU | **Near-parity** with GRU on matched splits; occasionally better on total return but similar risk—reasonable **fallback**. |
| **Transformer** | `results/wrds_transformer_ablation_2020_2024.json`; deck discussion in PDF above | **Sensitive** to regularization and windowing; higher complexity without consistent robust lift—**not** primary deploy architecture here. |

**Practical conclusion:** GRU remains the **go-to** allocator; LSTM stays as backup; Transformer stays in the **experiment** bucket unless narrow tuning budgets are available.

---

## 4. Online A/B gate results figure

**Artifacts:**

- Data: `online_training_work/results/ab_update_gate_results.csv`
- Figure: `figures/online_evaluation/ab_update_gate_results.png`

**Takeaway:** In the evaluated matrix, **cadence-style** update gating tends to outperform **confidence-only** gating on realized outcomes; cutting updates via confidence does not automatically improve net Sharpe when exposure to the IPO sleeve is already conservative.

---

## 5. Return trajectories (cadence / confidence / 50-50 + market + IPO)

**Artifacts:**

- Figure: `figures/online_evaluation/online_return_trajectories_all_benchmarks.png`
- Paths: `online_training_work/results/online_path_cadence_lb504.csv`, `online_training_work/results/online_path_confidence_lb504.csv`

**Representative cumulative returns (exported snapshot):**

| Series | Cumulative return |
|--------|-------------------|
| Cadence (net) | 91.45% |
| Confidence (net) | 85.94% |
| Static 50/50 | 298.22% |
| Market-only | 45.98% |
| IPO-only | 860.47% |

**Interpretation:** Changing **when** the model updates shifts the net path but does not remove the structural tension that **IPO-only** and **50/50** benchmarks can massively outperform when the learned policy **does not track** full IPO exposure—pointing to **allocation level** more than update frequency alone.

---

## 6. IPO weight trajectory (underweighting)

**Artifact:** `figures/online_evaluation/online_ipo_weight_trajectory.png`

The online allocator’s **IPO weight** remains **below** what a naïve IPO-max benchmark would imply during stretches where IPO returns dominate—consistent with the objective penalizing concentration/tail risk. That **systematic underweight** helps explain lag vs IPO-heavy references even when the recurrent mapping is “working” in loss terms.

---

## 7. Validation-loss diagnostics (hypothesis experiment chart)

**Artifacts:**

- Data: `results/val_loss_hypothesis_experiments.csv`
- Figure: `figures/online_evaluation/val_loss_hypothesis_experiments.png`

Validation loss and early-stopping behavior swing with **split geometry** and **scheduler policy**. Thin or poorly placed validation slices can send misleading “improvement” signals even with adaptive learning rates—supporting **longer, chronologically coherent** validation blocks for model selection.

---

## 8. Twelve-day validation window case study (loss + return + weights)

**IPO — ultra-short validation window**

| Artifact type | Path |
|---------------|------|
| Loss (semilogy / linear / rolling) | `figures/ipo_optimizer_replots/loss_semilogy_gru_offline_true_val12_adaptive.png`, `figures/ipo_optimizer_replots/loss_linear_gru_offline_true_val12_adaptive.png`, `figures/ipo_optimizer_replots/loss_rolling_gru_offline_true_val12_adaptive.png` |
| Returns vs 50/50 | `figures/ipo_optimizer_replots/returns_val12_adaptive_vs_5050.png` |
| Weights | `figures/ipo_optimizer_replots/ipo_weights_val12_adaptive.png` |
| Summary | `results/ipo_optimizer_summary_val12_adaptive.txt` |

**Key numbers (from summary artifact):**

- Model total return: **-2.78%** vs 50/50 **-3.22%** (marginal edge in a bad tape)
- Average IPO weight ≈ **1.36%** — extremely **bonded-to-market** behavior driven by validation noise

**Interpretation:** A **12-day** validation window is better treated as a **stress/diagnostic** than a primary selector—it induces unstable, ultra-conservative weights.

---

## 9. Case studies — commodities and bonds

### 9.1 Commodities

**Artifacts:** `results/commodity_optimizer_summary.txt`; `figures/commodity_optimizer/gru/validation_returns_vs_equal_weight.png`; `figures/commodity_optimizer/gru/weight_evolution_over_time.png`

**Snapshot:** Model total return **58.43%** vs equal-weight **40.36%** (~**+18.07%** advantage in the exported summary).

**Takeaway:** The same encoder–softmax recipe can **add value** when commodity vs equity regimes separate cleanly—dynamic sleeve tilting shows up in both returns and weight paths.

### 9.2 Bonds

**Artifacts:** `results/bonds/bond_optimizer_summary.txt`; bond optimizer outputs under `results/bonds/` (e.g. `bonbond_optimizer_returns.csv` consumed by the demo).

**Snapshot (from summary table in §3):** Model Sharpe **0.55** vs 50/50 **0.50** with materially negative max drawdown on the bond sleeve experiment—illustrating **lower Sharpe, crisis-sensitive** behavior relative to equities/commodities runs.

**Takeaway:** Bonds act as a **regime and duration** stress test: the pipeline runs end-to-end, but economic dominance vs simple blends is **not** guaranteed—consistent with rate/credit dynamics not captured in a short differentiable objective alone.

---

## 10. Lessons learned / limitations

- **Update gates** tune *when* learning refreshes; they do not fix a mismatch between **target economic exposure** and **learned weights** if the loss still penalizes aggressive IPO concentration.
- **Validation design dominates** apparent quality: tiny windows (e.g. 12 days) or leaky splits can swamp architecture gains.
- **Schedulers and clipping** must be **explicitly logged** per run; silent defaults across scripts invalidate comparisons.
- **Transformer** depth without disciplined regularization can lose to **GRU/LSTM** on realized paths despite flexibility on paper.
- **Benchmark storytelling** must separate **IPO sleeve beta** from **model alpha**: strong IPO-index years can make any diversified policy look “wrong” in raw return space.

---

## 11. Loss-function adjustment experiment — compound-growth-only objective

We trained and evaluated variants that **turn down** auxiliary penalties (CVaR, vol, diversification, etc.) and emphasize **compound-growth / log-growth-style** terms—mirroring `compound_only_loss` / `IPO_COMPOUND_ONLY_LOSS` style configurations in `run_ipo_optimizer_wrds.py`.

**Offline sensitivity (IPO weight perturbations under loss ablations):**

- CSV: `results/sensitivity_test_compound_only.csv`, `results/sensitivity_test_full_objective.csv`
- Figures: `figures/experiments/sensitivity/sensitivity_compound_only_top_mean_abs_delta_ipo_weight.png`; `figures/experiments/sensitivity/returns_full_vs_compound_only.png`

**Online vs offline compound comparator (same calendar dates):**

- Table: `results/online_compound_vs_offline_compound_same_dates.csv`
- Figures: `figures/online_evaluation/online_compound_returns_vs_benchmarks_and_offline_compound.png`; `figures/online_evaluation/online_compound_ipo_weight_trajectory.png`
- Generator script: `scripts/plot_online_compound_vs_offline_compound.py`

**Policy readout:** With **compound-only** pressure, the allocator **tilts harder toward growth** (often via higher IPO exposure in the online path) relative to the multi-term **full** objective—at the cost of abandoning the explicit tail/vol/diversify guardrails. Side-by-side plots separate **static offline compound training** from **sequential online compound training**, illustrating how **retraining** plus growth-only rewards reshapes the weight path. Interactive exploration: `scripts/demo.py` (Streamlit).

---

## 12. Future work

- **Unify** optimizer defaults: one scheduler + gradient-control profile across `run_ipo_optimizer_wrds.py` and sensitivity scripts.
- **Walk-forward validation blocks** (monthly/quarterly anchors) for checkpoint selection instead of single fragile validation slices.
- **Sleeve-aware objectives:** explicit floors/ceilings or penalty terms on deviation from target IPO beta during labeled regimes.
- **Feature enrichment** for IPO/bond/commodity sleeves (liquidity, issuance waves, curve factors).
- **Demo hardening:** extend `scripts/demo.py` into reproducible scenario batches (seeded stress paths, frozen artifact hashes for grading).

---

## Appendix — figure index (quick lookup)

| Topic | Path |
|-------|------|
| Architecture | `figures/architecture/offline_gru_static_pipeline_visual.png`, `figures/architecture/online_gru_adaptive_pipeline_clean.png` |
| A/B gates | `figures/online_evaluation/ab_update_gate_results.png` |
| Online trajectories | `figures/online_evaluation/online_return_trajectories_all_benchmarks.png` |
| IPO weights | `figures/online_evaluation/online_ipo_weight_trajectory.png` |
| Val-loss diagnostics | `figures/online_evaluation/val_loss_hypothesis_experiments.png` |
| Compound-only evaluation | `figures/online_evaluation/online_compound_returns_vs_benchmarks_and_offline_compound.png`, `figures/online_evaluation/online_compound_ipo_weight_trajectory.png` |
| Sensitivity | `figures/experiments/sensitivity/returns_full_vs_compound_only.png` |

---

*End of report.*
