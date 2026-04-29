# Self-Critique (OODA) - IPO Optimizer Status

## Observe
The repository is currently stable and runnable: sector-capable code is in place, 126-day comparison artifacts are preserved, and the 252-day outputs are available in `figures/` and `results/`.

From the 252-day figures, the pattern is mixed. Training loss drops quickly at first, but validation improvement slows and becomes noisy, with a worse convergence floor than the 126-day setup. The semilog plots show early progress followed by flattening, and `validation_objective` looks regime-sensitive rather than steadily improving.

We also added a global multisector allocator (single softmax over market + sector sleeves). It trains end-to-end and generates clean outputs, but current results are still moderate rather than clearly superior. In addition, we tested a transformer-based architecture as an alternative model class; it did not show meaningfully better convergence efficiency, and validation loss did not improve enough to justify switching from the current baseline.

## Orient

### Strengths
- The 2010-2024 pipeline runs end-to-end with sector portfolios and exports complete outputs. Overall validation loss looks better than previous training on only 2020-2024 data.
- Multiple model variants were tested (GRU baseline, transformer trial, and global multisector allocator), so comparisons are now more informative.
- Branch cleanup reduced clutter from cache artifacts and made the training state easier to review.

### Areas for Improvement
- Figure interpretation is mostly qualitative; we need a small set of standard numeric diagnostics tied to each plot.
- Sector metadata quality is not tracked as a first-class metric, even though it affects downstream behavior.
- Transformer and multisector experiments need a tighter evaluation protocol so we can clearly explain when a new architecture is actually better.

### Critical Risks/Assumptions
We still assume missing or stale sector labels do not materially bias sector sleeves, which may be false for older/delisted names. We also assume objective values are directly comparable across context windows even when volatility regimes differ. If these assumptions are wrong, model comparisons can look better or worse for the wrong reasons.

## Decide

### Concrete Next Actions
- Train/learn the sector labels and attempt to classify undetermined names.
- Add three mandatory diagnostics after every run: best epoch, validation-loss stability (rolling std), and sector-coverage summary.
- Add a short comparison script that reports deltas between 126-day and 252-day runs before accepting new results.

## Act

### Resource Needs
As we expand our dataset, we plan to leverage our GPU credits during the training process.
