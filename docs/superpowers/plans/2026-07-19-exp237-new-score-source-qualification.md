# Exp237 New Score Source Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible TRAIN-only qualification audit for five new anomaly score sources across all 1,117 datasets.

**Architecture:** A dataset worker loads label-free TRAIN/TEST values, aligns them using a TRAIN-derived target length, computes deterministic five-fold OOF and full-TRAIN TEST scores, and writes a hashed cache. A separate prediction phase converts scores to empirical p-value candidates, freezes label-free JSONL, and only then evaluates with TEST labels.

**Tech Stack:** Python 3, NumPy, SciPy/scikit-learn where already installed, SQLite read-only access, `unittest`, CSV/JSONL/NPZ artifacts.

## Global Constraints

- Seed is `20260719`; alpha grid is `(0.005, 0.01, 0.02, 0.05)`.
- No TEST label, anomaly position, family name, or prior family TEST performance may affect prediction.
- Do not modify Exp236 outputs or promote Exp237 candidates into automatic alerts.
- Approximate elastic and subsequence methods must not be described as exact DTW or exact Matrix Profile.

---

### Task 1: Core contracts and source unit tests

**Files:**
- Create: `experiments/exp237_new_score_source_qualification/__init__.py`
- Create: `experiments/exp237_new_score_source_qualification/config.py`
- Create: `experiments/exp237_new_score_source_qualification/core.py`
- Create: `experiments/exp237_new_score_source_qualification/score_sources.py`
- Test: `experiments/exp237_new_score_source_qualification/test_exp237.py`

**Interfaces:**
- Produces: `deterministic_fold_plan(n_rows, seed)`, `fit_score_source(name, train)`, `score_rows(model, rows)`, `build_score_bundle(dataset, train, test)`.

- [ ] Write failing tests for fold disjointness, determinism, finite scores, fit/query separation, and five source names.
- [ ] Run `python3 -m unittest experiments.exp237_new_score_source_qualification.test_exp237` and confirm failure.
- [ ] Implement minimal deterministic score sources and validation.
- [ ] Re-run the unit test and confirm all tests pass.

### Task 2: Hashed resumable cache and label-free prediction

**Files:**
- Create: `experiments/exp237_new_score_source_qualification/cache.py`
- Create: `experiments/exp237_new_score_source_qualification/prediction.py`
- Modify: `experiments/exp237_new_score_source_qualification/test_exp237.py`

**Interfaces:**
- Consumes: `build_score_bundle`.
- Produces: atomic per-dataset NPZ/JSON manifests, `freeze_predictions`, `load_verified_predictions`.

- [ ] Add failing tests for hash tamper detection and forbidden evaluation fields.
- [ ] Run the focused tests and confirm failure.
- [ ] Implement atomic cache/prediction persistence and empirical p-value candidates.
- [ ] Run tests and confirm pass.

### Task 3: Runner, evaluation, and independent aggregation

**Files:**
- Create: `experiments/exp237_new_score_source_qualification/run_experiment_237.py`
- Create: `experiments/exp237_new_score_source_qualification/reporting.py`
- Create: `experiments/exp237_new_score_source_qualification/independent_reaggregate.py`
- Modify: `experiments/exp237_new_score_source_qualification/test_exp237.py`

**Interfaces:**
- Produces CLI stages `cache`, `predict`, `evaluate`, `all`; result/summary/qualification CSVs; heartbeat and status files.

- [ ] Add tests proving evaluation happens only after prediction hash verification and labels are absent from prediction payloads.
- [ ] Implement the runner with `--dataset-limit`, `--workers`, `--reset-cache`, and resumable cache behavior.
- [ ] Implement source/alpha summary and independent reaggregation.
- [ ] Run py_compile and all Exp237 unit tests.

### Task 4: Smoke, full run, dashboard, and report

**Files:**
- Create: `outputs/exp237_new_score_source_qualification/RESULT_REPORT.md`
- Create: `outputs/exp237_new_score_source_qualification/MANIFEST.csv`
- Modify: dashboard experiment catalog/status mapping only if required by the existing dashboard architecture.

**Interfaces:**
- Consumes frozen artifacts under `/Users/minho/Documents/Dataset/exp237_new_score_source_qualification`.
- Produces verified 1,117-dataset summaries and dashboard visibility.

- [ ] Run a 10-dataset smoke with two workers and verify no labels/metrics in prediction JSONL.
- [ ] Run all 1,117 datasets with seven workers and zero missing/error datasets.
- [ ] Run independent reaggregation and compare exact summary totals.
- [ ] Add dashboard metadata without marking Exp237 comparison-eligible.
- [ ] Record commands, hashes, limitations, and results; run `git diff --check` and relevant tests.
