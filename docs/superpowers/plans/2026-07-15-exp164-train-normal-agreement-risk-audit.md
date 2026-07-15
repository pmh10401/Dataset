# Exp164 TRAIN-Normal Agreement Risk Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Implement every behavior with RED-GREEN-REFACTOR.

**Goal:** Measure the normal false-review risk of Exp163's three-source 2-of-3 and 3-of-3 agreement using only cross-fitted TRAIN-normal pseudo-runs.

**Architecture:** A TRAIN-only SQLite bridge returns raw normal traces. A pure cross-fit core rebuilds the frozen Exp162 feature sources within each fold and emits per-run source flags. A checkpointed seven-worker runner writes hash-validated dataset caches and label-free risk summaries; an independent reaggregator verifies every count.

**Tech Stack:** Python 3, NumPy, scikit-learn, SQLite, `concurrent.futures`, `unittest`.

## Global Constraints

- Database remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Population is the frozen 1,117 Exp162 datasets.
- Audit code must not read TEST values, TEST labels, anomaly positions, family names, or Exp160/163 lane results.
- `ALPHAS = (0.005, 0.01, 0.02, 0.05)`.
- Variants are `train_audit_2of3` and `train_audit_3of3`.
- Risk targets are `(0.005, 0.01, 0.02)` per normal run and are all sensitivity outputs.
- `n_train < 5` is complete abstention.
- No TEST metric selects an alpha, variant, threshold, source, or risk target.
- No Hard/Priority lane is produced.
- Workers default to 7 and inner numeric threads to 1.

---

### Task 1: Pure Fold, Threshold, Agreement, And Risk Contracts

**Files:**
- Create: `experiments/exp164/__init__.py`
- Create: `experiments/exp164/train_normal_risk_core.py`
- Create: `experiments/exp164/test_exp164_train_normal_risk_audit.py`

**Interfaces:**
- Produces: `fold_plan(n_train, seed)`, `higher_threshold(fit_scores, alpha)`, `source_candidate_flags(query_scores, threshold)`, `agreement_flags(source_flags, required_sources)`, `wilson_upper_bound(false_reviews, runs, confidence)`, and `risk_sensitivity_rows(...)`.

- [ ] Write failing tests for LOO/5-fold coverage, query-fit exclusion, deterministic folds, strict tie rejection, 2-of-3/3-of-3 flags, n_train abstention, and Wilson bounds.
- [ ] Run `python3 -m unittest experiments/exp164/test_exp164_train_normal_risk_audit.py` and confirm missing APIs fail.
- [ ] Implement only the pure helpers required by the tests.
- [ ] Re-run the test module and confirm all core tests pass.

### Task 2: TRAIN-Only Raw-Series Bridge And Cross-Fit Source Audit

**Files:**
- Create: `experiments/exp164/exp164_train_bridge.py`
- Create: `experiments/exp164/crossfit_sources.py`
- Modify: `experiments/exp164/test_exp164_train_normal_risk_audit.py`

**Interfaces:**
- Consumes: Exp162 `spectral_features`, `shapelet_features`, `interval_features`, `robust_score_pair`, and `resample_rows`.
- Produces: `load_train_normal_raw(dataset_name) -> list[np.ndarray]`, `score_fold(dataset_name, fit_raw, query_raw)`, and `audit_dataset(dataset_name, raw_series)`.

- [ ] Write failing AST/SQL tests proving only TRAIN `values_blob` is queried and neither TEST nor label appears in executable SQL.
- [ ] Write failing tests proving target length is derived from fit rows only and query rows never enter fitting.
- [ ] Implement raw TRAIN loading, per-fold alignment, the three frozen feature sources, and per-alpha source/variant flags.
- [ ] Verify every pseudo-run is scored exactly once and all arrays are finite.

### Task 3: Atomic Cache, Checkpoint, And Seven-Worker Runner

**Files:**
- Create: `experiments/exp164/exp164_cache_io.py`
- Create: `experiments/exp164/run_experiment_164_train_normal_agreement_risk_audit.py`
- Create: `experiments/exp164/test_exp164_structure.py`
- Create: `docs/EXP164_PREREGISTRATION_KO.md`
- Create: `docs/EXP164_RUNBOOK_KO.md`

**Interfaces:**
- Produces per-dataset NPZ caches, `01_audit_contract.json`, `02_cache_manifest.csv`, `03_errors.csv`, `04_dataset_alpha_summary.csv`, `05_global_summary.csv`, `06_risk_target_sensitivity.csv`, `07_source_dependence.csv`, `CHECKPOINT.jsonl`, and manifests.

- [ ] Write failing tests for required cache arrays/metadata, forbidden TEST/evaluation fields, atomic writes, tamper rejection, checkpoint config drift, and direct CLI execution.
- [ ] Implement dataset-level seven-worker scheduling and resumable cache verification.
- [ ] Aggregate source/variant rates, Wilson bounds, threshold resolution, n_train bins, and source dependence without TEST inputs.
- [ ] Document exact smoke, full, resume, and independent-validation commands.

### Task 4: Antigravity Smoke And Codex Review

**Files:**
- Generate: `outputs/exp164_train_normal_agreement_risk_audit_smoke/`
- Generate: `outputs/antigravity_collaboration/exp164_implementation_agy.log`

- [ ] Run compile and both test modules.
- [ ] Run a 10-dataset smoke with 2 workers and checkpoint reset.
- [ ] Verify cache coverage, pseudo-run coverage, no TEST fields, no missing folds, and deterministic two-run hashes.
- [ ] Stop before full execution so Codex can review the query path, fold leakage, source parity, and checkpoint contract.

### Task 5: Full TRAIN-Normal Audit

**Files:**
- Generate: `outputs/exp164_train_normal_agreement_risk_audit/`

- [ ] Run all 1,117 datasets with 7 workers.
- [ ] Require 1,117 manifest rows, zero missing datasets, zero errors, and 162 expected `n_train < 5` abstentions unless the frozen population data proves a different count.
- [ ] Resume once without reset and verify all valid cache hashes are reused.
- [ ] Verify one-worker and seven-worker sample replays produce identical cache content hashes.

### Task 6: Independent Reaggregation And Reporting

**Files:**
- Create: `experiments/exp164/independent_reaggregate_exp164.py`
- Generate: `outputs/exp164_train_normal_agreement_risk_audit/VERIFICATION_REPORT.md`
- Generate: `outputs/exp164_train_normal_agreement_risk_audit/MANIFEST.csv`
- Generate: `outputs/exp164_train_normal_agreement_risk_audit/MANIFEST.md`

- [ ] Independently read all caches and recompute pseudo-run, source, agreement, alpha, and risk-target summaries.
- [ ] Compare independent summaries with runner outputs and fail on any mismatch.
- [ ] Report every alpha, variant, and risk target without selecting a primary policy.
- [ ] State that the experiment is TRAIN-normal only, retrospective, virtual-run based, not end-to-end provenance complete, and not prospectively validated.
