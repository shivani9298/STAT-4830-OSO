# Self-Critique (OODA) - IPO Optimizer Normalization

## Observe
I reviewed the current `olivia` branch after rerunning the sector-head pipeline with the 2010-2024 sample and a forced 252-day context window. The run completed and regenerated training artifacts, but the branch accumulated mixed code, caches, and outputs from multiple debugging cycles. My first reaction is that we now have a working configuration, but reproducibility and change hygiene lag behind model iteration speed.

## Orient

### Strengths
- Achieved an end-to-end sector-capable run on the expanded 2010-2024 data with updated loss plots and sector summaries.
- Recovered compatibility across runner and `src/*` APIs after branch drift, enabling training without ad hoc manual edits.
- Preserved historical context by keeping 126-day comparison artifacts alongside the 252-day setup.

### Areas for Improvement
- Configuration control is too implicit: context window and split behavior can still drift between code defaults, tuning outputs, and runtime overrides.
- Data/feature assumptions are under-tested: sector assignment quality (many ticker lookup misses) and index-construction robustness need explicit validation checks.
- Repo cleanliness is inconsistent: generated artifacts and cache churn obscure intentional code changes and complicate review.

### Critical Risks/Assumptions
The current approach assumes Yahoo sector metadata is sufficiently stable and that missing sectors do not bias allocation behavior; this may not hold across older or delisted IPO tickers. It also assumes objective values are comparable across context windows despite differing volatility regimes and sample composition. Without a fixed experiment manifest, we risk attributing performance changes to model choices when they are partly data-pipeline differences.

## Decide

### Concrete Next Actions
- Add a single experiment manifest file (window length, date bounds, split points, sector mode, loss weights) that is logged into `results/` for every run.
- Implement pre-training data QA checks (sector coverage %, missing-ticker report, per-sector sample counts) and fail fast when thresholds are violated.
- Split outputs into `tracked reports` vs `ignored runtime artifacts`, then enforce with a lightweight pre-commit check for cache/pyc pollution.

## Act

### Resource Needs
I need one focused pass on experiment management patterns (Hydra-style config discipline or an equivalent lightweight JSON/YAML registry) to eliminate hidden parameter drift. I also need a short validation checklist for financial time-series experiments (coverage diagnostics, embargo checks, leakage tests) to standardize run acceptance criteria. The only external help needed is quick alignment on which artifacts are intended to be versioned versus reproducible outputs regenerated on demand.
