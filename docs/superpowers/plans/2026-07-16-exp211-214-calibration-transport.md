# Exp211-214 Calibration Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine and repair the OOF-to-TEST calibration transport failure observed in Exp208/209 without changing the frozen Exp137 autonomous Hard-alert policy.

**Architecture:** Exp211 performs a label-free source and fusion transport audit using frozen Exp206 caches. Exp212 rebuilds TEST scores with the same five fold-fitted models used for TRAIN OOF calibration and freezes candidate predictions before evaluation. Exp213 applies pre-registered TRAIN-only stability criteria to the same fold artifacts. Exp214 independently reaggregates all outputs and records operational boundaries.

**Tech Stack:** Python 3, NumPy, SQLite, CSV/JSONL, unittest, existing sequential runner and dashboard.

## Global Constraints

- Population is the frozen 251 datasets with `n_train >= 100`.
- Existing Exp137 and Exp206-210 outputs remain unchanged.
- TEST labels and anomaly positions are unavailable to prediction functions.
- Family or dataset names cannot control source selection or thresholds.
- TEST length cannot control threshold, candidate budget, or source selection.
- Exp137 Hard alerts remain autonomous and unchanged.
- New candidates are retrospective research or review diagnostics only.
- No TEST metric may select a source, alpha, aggregation rule, or stability threshold.
- Fixed alpha grid remains `(0.005, 0.01, 0.02, 0.05)`.
- Sequential execution uses seven workers only for the fold-artifact build stage.

---

### Task 1: Calibration transport core and tests

**Files:**
- Create: `experiments/exp211_214/__init__.py`
- Create: `experiments/exp211_214/core.py`
- Create: `experiments/exp211_214/test_exp211_214_core.py`

**Interfaces:**
- Consumes: Exp206 score vectors and `experiments.exp206_210.core` p-value helpers.
- Produces: `transport_metrics()`, `aggregate_fold_pvalues()`, `source_stability()`, and prediction payload validation.

- [ ] Write failing unit tests for deterministic fold p-value aggregation, transport metrics, stability bounds, and forbidden evaluation fields.
- [ ] Run `python3 -m unittest experiments.exp211_214.test_exp211_214_core` and confirm failure.
- [ ] Implement the minimal core functions and constants.
- [ ] Run the unit tests and confirm success.

### Task 2: Fold-matched source artifacts

**Files:**
- Create: `experiments/exp211_214/fold_transport.py`
- Modify: `experiments/exp211_214/test_exp211_214_core.py`

**Interfaces:**
- Consumes: TRAIN and TEST values plus Exp206 alternative source implementations.
- Produces: source-specific OOF scores, full-fit TRAIN/TEST scores, five fold calibration vectors, five fold TEST p-value vectors, and deterministic config hashes.

- [ ] Add failing tests proving query changes cannot change calibration scores and repeated runs are identical.
- [ ] Implement five-fold source scoring with held-out TRAIN calibration and TEST scoring under the same fold model.
- [ ] Verify every TRAIN index is held out exactly once and all output arrays are finite.
- [ ] Run all Exp211-214 core tests.

### Task 3: Exp211 audit and Exp212 prediction/evaluation

**Files:**
- Create: `experiments/exp211_214/run_experiment_211_214.py`
- Create: `experiments/exp211_214/test_exp211_214_structure.py`

**Interfaces:**
- Consumes: Exp137 baseline, Exp206 cache/manifest, raw TRAIN/TEST values.
- Produces: Exp211 transport CSV, Exp212 fold artifact manifest, frozen predictions, delayed evaluation results, and summaries.

- [ ] Add structural tests that prediction functions do not accept `y_test` and frozen rows contain no TP/FP/F1.
- [ ] Implement Exp211 as a read-only, label-free audit of OOF/full-fit TRAIN and unlabeled TEST score transport.
- [ ] Implement Exp212 fold-artifact generation with seven workers and resumable SHA256-verified caches.
- [ ] Freeze Exp212 predictions for all source and fold-aggregation variants before loading labels.
- [ ] Evaluate frozen predictions without changing any rules.
- [ ] Verify hashes, row counts, lane overlap, and candidate index ranges.

### Task 4: Exp213 stability guard and Exp214 reaggregation

**Files:**
- Modify: `experiments/exp211_214/run_experiment_211_214.py`
- Modify: `experiments/exp211_214/test_exp211_214_structure.py`

**Interfaces:**
- Consumes: Exp211 TRAIN-only stability evidence and Exp212 frozen fold p-values.
- Produces: Exp213 frozen guarded predictions/evaluation and Exp214 comparison frontier.

- [ ] Freeze the TRAIN-only stability contract: rank correlation `>= 0.90`, robust median shift `<= 0.25`, and MAD ratio in `[0.80, 1.25]`.
- [ ] Report unguarded, stable-source 2-of-N, and all-stable-source variants without choosing a winner from TEST metrics.
- [ ] Preserve zero-source and one-source cases as abstentions instead of relaxing rules.
- [ ] Independently reaggregate Exp211-213 outputs and compare with the frozen Exp137 baseline.
- [ ] Mark all new candidate metrics retrospective and non-autonomous.

### Task 5: Runner, dashboard, verification, and Agy review

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py` only if automatic runner metadata discovery does not expose Exp211-214.
- Create: `outputs/exp211_214_validation/agy_review.md`

**Interfaces:**
- Consumes: Exp211-214 CLI stages.
- Produces: a one-at-a-time queue visible in the existing dashboard.

- [ ] Register Exp211-214 with expected dataset count 251 and seven workers only for Exp212.
- [ ] Run `py_compile` and all Exp211-214 tests.
- [ ] Run a three-dataset smoke sequence through Exp211-214.
- [ ] Ask Agy to review prediction/evaluation separation, calibration matching, and queue safety.
- [ ] Apply only contract-level fixes found before the full run.
- [ ] Queue Exp211, Exp212, Exp213, and Exp214 in order and launch the first item.
- [ ] Verify the dashboard API reports the active experiment and remaining queue.

## Interpretation Contract

- Exp211 can diagnose but cannot claim causality from unlabeled TEST drift alone.
- Exp212 tests whether scoring-mechanism alignment improves calibration transport.
- Exp213 tests a pre-registered TRAIN-only source guard; no TEST-selected routing is permitted.
- Exp214 is retrospective comparison only.
- Even successful results do not establish end-to-end strict TRAIN-only provenance or prospective deployment validity.
