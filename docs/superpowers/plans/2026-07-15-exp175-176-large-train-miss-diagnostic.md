# Exp175/176 Large-TRAIN Miss Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain the 2,107 Exp137 Hard misses in datasets with at least 20 TRAIN-normal instances and separate policy/budget misses from score-representation misses.

**Architecture:** A prediction-safe loader freezes four existing TEST score vectors and their TRAIN OOF references without labels. A separate evaluator then loads labels, computes anomaly-level rank and conformal-tail diagnostics, aggregates dataset-level concentration, and independently reaggregates all reported totals.

**Tech Stack:** Python 3, NumPy, SQLite, CSV/JSON, unittest.

## Global Constraints

- Do not overwrite existing Exp137 or score-cache artifacts.
- Do not use TEST results to select alpha, K, source, threshold, or routing.
- Keep the alpha grid fixed at `(0.005, 0.01, 0.02, 0.05)`.
- Keep rank-depth sensitivity fixed at `(1, 2, 3, 5, 8, 10)`.
- Keep autonomous Hard metrics separate from human-assisted review metrics.
- Do not use tail replacement.
- Mark every result retrospective and non-prospective.

---

### Task 1: Diagnostic Core

**Files:**
- Create: `experiments/exp175_176/__init__.py`
- Create: `experiments/exp175_176/diagnostic_core.py`
- Test: `experiments/exp175_176/test_exp175_176_diagnostic_core.py`

**Interfaces:**
- Produces: `upper_tail_pvalue(train_scores, value) -> float`, `descending_ranks(scores) -> np.ndarray`, `classify_miss(...) -> str`, and fixed-grid support helpers.

- [ ] Write tests for tie-stable descending ranks, finite-sample p-values, mutually exclusive cause categories, and fixed alpha/K constants.
- [ ] Run the tests and confirm they fail because the module is absent.
- [ ] Implement the minimal core functions.
- [ ] Run the tests and confirm they pass.

### Task 2: Frozen Score Loader and Evaluator

**Files:**
- Create: `experiments/exp175_176/run_experiment_175_176_large_train_miss_diagnostic.py`
- Test: `experiments/exp175_176/test_exp175_176_structure.py`

**Interfaces:**
- Consumes: Exp137 CSV, score-cache manifest/NPZ files, SQLite labels.
- Produces: label-free frozen score records followed by dataset-level and anomaly-level retrospective diagnostics.

- [ ] Write structural tests that reject label fields in frozen score records and verify cache/source length contracts.
- [ ] Run the tests and confirm they fail before the runner exists.
- [ ] Implement manifest resolution, score freezing, delayed label loading, and CSV/report writers.
- [ ] Run unit and structural tests.

### Task 3: Independent Reaggregation

**Files:**
- Create: `experiments/exp175_176/independent_reaggregate_exp175_176.py`
- Test: `experiments/exp175_176/test_exp175_176_reaggregation.py`

**Interfaces:**
- Consumes: generated Exp175/176 CSV files.
- Produces: `independent_reaggregation.json` with count and category reconciliation.

- [ ] Write a failing fixture test for totals and mutually exclusive categories.
- [ ] Implement read-only reaggregation.
- [ ] Verify the fixture and full test suite pass.

### Task 4: Full Diagnostic Run

**Files:**
- Create under: `outputs/exp175_176_large_train_miss_diagnostic/`

**Interfaces:**
- Produces: frozen score manifest, Exp175 dataset CSV, Exp176 anomaly CSV, alpha/K summaries, report, log, and manifest.

- [ ] Run the full 639-dataset diagnostic.
- [ ] Verify 639 datasets, 3,440 anomalies, and 2,107 Hard misses.
- [ ] Run independent reaggregation and compare hashes/totals.
- [ ] Run `git diff --check` and the focused unit suite.

### Task 5: Result Interpretation

**Files:**
- Create: `outputs/exp175_176_large_train_miss_diagnostic/09_result_analysis.md`

**Interfaces:**
- Consumes: verified aggregate outputs only.
- Produces: a Korean operational interpretation separating policy/budget and representation bottlenecks.

- [ ] Report cause counts and anomaly concentration without selecting a TEST-optimal policy.
- [ ] State whether increasing candidate depth alone is likely to recover a material fraction.
- [ ] Propose a next experiment only from the diagnosed bottleneck and preserve false-alarm-first constraints.
