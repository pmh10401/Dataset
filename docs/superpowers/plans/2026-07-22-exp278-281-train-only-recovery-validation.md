# Exp278-281 TRAIN-only Recovery Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and sequentially execute four label-safe recovery validation experiments while keeping Exp253 Hard alerts unchanged.

**Architecture:** A focused `experiments/exp278_281_train_only_recovery/` package reads the frozen Exp272 and Exp270 artifacts, builds label-free audits and an Exp281 prediction JSONL, then evaluates only after verifying its SHA256. A thin wrapper registers four stages in the existing sequential runner and dashboard.

**Tech Stack:** Python 3.14, NumPy, SQLite read-only evaluation, CSV/JSONL/SHA256 contracts, unittest, existing rank dashboard.

## Global Constraints

- Use all 1,117 datasets; do not silently exclude failures.
- Keep Exp253 autonomous Hard alerts immutable.
- Do not use TEST labels, anomaly positions, family names, TEST length, or TEST order in prediction policy.
- Freeze `alpha=0.005`, full-reference-size bootstrap seeds `20260722..20260726`, risk targets `0.005/0.01/0.02`, and Exp281 variants before evaluation.
- Do not merge adjacent TEST indices because each index is a separate series instance.
- Keep every new candidate review-only and retrospective.

---

### Task 1: Contracts and risk-resolution audit

**Files:**
- Create: `experiments/exp278_281_train_only_recovery/contracts.py`
- Create: `experiments/exp278_281_train_only_recovery/risk_audit.py`
- Test: `experiments/exp278_281_train_only_recovery/tests/test_risk_audit.py`

**Interfaces:**
- Consumes: Exp272 frozen prediction, source caches, Exp270 TRAIN-risk CSV.
- Produces: `run_risk_resolution_audit(output_root: Path) -> tuple[Path, Path]`.

- [ ] Write tests proving fixed constants, Wilson-risk lookup, and zero TEST-label fields.
- [ ] Run `python3 -m unittest experiments.exp278_281_train_only_recovery.tests.test_risk_audit -v` and confirm the tests fail before implementation.
- [ ] Implement all-population candidate evidence reconstruction and write `exp278_risk_resolution_dataset.csv` plus `exp278_summary.csv`.
- [ ] Re-run the test and confirm it passes.
- [ ] Commit the independently testable audit.

### Task 2: Calibration-reference stability

**Files:**
- Create: `experiments/exp278_281_train_only_recovery/stability.py`
- Test: `experiments/exp278_281_train_only_recovery/tests/test_stability.py`

**Interfaces:**
- Consumes: verified source cache arrays and Exp272 candidate support.
- Produces: `run_calibration_stability(output_root: Path) -> tuple[Path, Path]`.

- [ ] Write deterministic tests for five full-size TRAIN OOF bootstrap samples and identical-seed output hashes.
- [ ] Run the focused test and verify failure before implementation.
- [ ] Implement empirical p-value recurrence per source group without TEST labels.
- [ ] Write `exp279_candidate_stability.csv` and `exp279_summary.csv` with explicit `calibration_reference_stability` terminology.
- [ ] Run the focused and risk-audit tests and commit.

### Task 3: Event-merging applicability audit

**Files:**
- Create: `experiments/exp278_281_train_only_recovery/event_audit.py`
- Test: `experiments/exp278_281_train_only_recovery/tests/test_event_audit.py`

**Interfaces:**
- Consumes: DB schema metadata and Exp272 lane indices.
- Produces: `run_event_applicability_audit(output_root: Path) -> tuple[Path, Path]`.

- [ ] Write tests that reject adjacent-index event merging for instance-level candidate indices.
- [ ] Verify the test fails before implementation.
- [ ] Query SQLite in read-only mode to document instance grain and audit duplicate/lane-overlap keys.
- [ ] Write `exp280_event_applicability.csv` and `exp280_summary.csv`; do not alter candidate indices.
- [ ] Run the focused tests and commit.

### Task 4: Exp281 frozen two-stage verifier

**Files:**
- Create: `experiments/exp278_281_train_only_recovery/verifier.py`
- Create: `experiments/exp278_281_train_only_recovery/evaluation.py`
- Test: `experiments/exp278_281_train_only_recovery/tests/test_verifier.py`

**Interfaces:**
- Consumes: Exp272 lanes, Exp278 candidate risks, Exp279 stability.
- Produces: `freeze_verifier_predictions(...)` and `evaluate_verifier_predictions(...)`.

- [ ] Write tests for the three frozen variants, minimum two continuous groups, lane exclusivity, and forbidden evaluation fields.
- [ ] Write a tamper test proving TEST labels are not loaded before prediction hash verification.
- [ ] Implement the label-free prediction freeze as `exp281_frozen_predictions.jsonl` and `exp281_prediction_manifest.json`.
- [ ] Implement separate evaluation outputs `exp281_dataset_results.csv` and `exp281_summary.csv`.
- [ ] Run all package tests and commit.

### Task 5: Pipeline, queue, dashboard, and full execution

**Files:**
- Create: `experiments/exp278_281_train_only_recovery/run_pipeline.py`
- Create: `scratch/run_exp278_281_train_only_recovery.py`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Test: `experiments/exp278_281_train_only_recovery/tests/test_registration.py`

**Interfaces:**
- Consumes: stages `278`, `279`, `280`, `281`.
- Produces: queue command `queue-exp278-281` and live status for all four stages.

- [ ] Add registration tests requiring exact sequential order and separate Hard/review metric scopes.
- [ ] Implement status publishing and runner/dashboard registration.
- [ ] Run `python3 -m unittest discover -s experiments/exp278_281_train_only_recovery/tests -v` and existing Exp267/274 dashboard tests.
- [ ] Restart the dashboard and verify `http://127.0.0.1:8765` returns HTTP 200.
- [ ] Queue and execute all stages one at a time.
- [ ] Independently reaggregate Exp281 from the frozen prediction and read-only DB labels.
- [ ] Write `RESULT_REPORT.md`, verify queue completion and output hashes, then commit only code/docs changes.
