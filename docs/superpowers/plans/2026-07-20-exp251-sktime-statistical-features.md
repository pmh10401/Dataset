# Exp251 sktime Statistical Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable sktime 1.0.1 statistical-feature qualification experiment over the frozen 1,117-dataset Exp236 population.

**Architecture:** A dedicated package extracts four deterministic label-free feature blocks once per dataset, freezes TRAIN-only non-finite replacements, and scores five individual/combined variants with five-fold OOF KNN-3 local-gap. Prediction, evaluation, and independent reaggregation remain separate phases.

**Tech Stack:** Python 3.14, sktime 1.0.1, NumPy, scikit-learn, SQLite, unittest, existing Exp249/250 lane and dashboard contracts.

## Global Constraints

- Keep Exp236 and Exp249/250 code and results unchanged.
- Use exactly the five feature variants and four alpha values in the design.
- Prediction must not read TEST labels or evaluation metrics.
- No family, dataset-name, TEST-length, tail-position, or past TEST-performance policy.
- Use `/Users/minho/Documents/Dataset/venvs/tsai-sktime-py314/bin/python`.
- Run with seven dataset workers only after Exp250 completes.

---

### Task 1: Feature and TRAIN-only sanitization contract

**Files:**
- Create: `experiments/exp251_sktime_statistical_features/config.py`
- Create: `experiments/exp251_sktime_statistical_features/features.py`
- Test: `experiments/exp251_sktime_statistical_features/tests/test_features.py`

**Interfaces:**
- Produces: `extract_feature_blocks(rows, target_length) -> dict[str, np.ndarray]` and `fit_train_sanitizer(train_blocks)`.
- Consumes: aligned two-dimensional instance arrays.

- [ ] Write failing tests for exact block names/dimensions, `requires_y=False`, deterministic output, and TRAIN-only replacement of TEST non-finite values.
- [ ] Run `python -m unittest experiments.exp251_sktime_statistical_features.tests.test_features -v` and confirm missing-module failure.
- [ ] Implement the four fixed sktime blocks, adaptive TRAIN-length Hurst lag, and combined block.
- [ ] Run focused tests and confirm all pass.
- [ ] Commit the feature contract.

### Task 2: OOF local-gap cache

**Files:**
- Create: `experiments/exp251_sktime_statistical_features/scoring.py`
- Create: `experiments/exp251_sktime_statistical_features/cache_pipeline.py`
- Test: `experiments/exp251_sktime_statistical_features/tests/test_scoring.py`

**Interfaces:**
- Produces: per-dataset NPZ score cache and JSON provenance.
- Consumes: five frozen feature matrices and deterministic cross-fit plans.

- [ ] Write failing tests proving fold-query rows are excluded from scaler/KNN fit and `n_train < 5` abstains with all variants retained.
- [ ] Run the focused test and confirm expected failure.
- [ ] Implement five-fold OOF KNN-3 local-gap, full TRAIN-fit TEST scoring, atomic cache writes, package-version config hashes, and seven-worker coverage checks.
- [ ] Run focused tests and py_compile.
- [ ] Commit scoring and cache.

### Task 3: Label-free prediction and post-freeze evaluation

**Files:**
- Create: `experiments/exp251_sktime_statistical_features/prediction.py`
- Create: `experiments/exp251_sktime_statistical_features/evaluation.py`
- Create: `experiments/exp251_sktime_statistical_features/independent_reaggregate.py`
- Create: `experiments/exp251_sktime_statistical_features/reporting.py`
- Test: `experiments/exp251_sktime_statistical_features/tests/test_pipeline.py`

**Interfaces:**
- Produces: frozen prediction JSONL, `results.csv`, `summary.csv`, report, and mismatch manifest.
- Consumes: hash-verified caches and frozen Exp236 lane indices.

- [ ] Write failing tests for forbidden prediction fields, four alpha rows per variant, prediction hash tamper rejection, gate formulas, and independent summary matching.
- [ ] Run tests and confirm expected failures.
- [ ] Implement prediction freeze, post-freeze label evaluation, qualification gates, and independent reaggregation.
- [ ] Run all package tests.
- [ ] Commit prediction and evaluation.

### Task 4: Runner, queue, dashboard, and smoke

**Files:**
- Create: `experiments/exp251_sktime_statistical_features/run_pipeline.py`
- Create: `scratch/run_exp251_after_current.py`
- Create: `scratch/exp251_smoke_datasets.txt`
- Modify: `serve_rank_dashboard.py`
- Test: `experiments/exp251_sktime_statistical_features/tests/test_queue_dashboard.py`

**Interfaces:**
- Produces: canonical queued/running/completed state and dashboard-visible status.
- Consumes: the isolated sktime Python executable and canonical sequential state.

- [ ] Write failing tests that Exp251 is registered, uses the isolated interpreter, waits for the current experiment, and reports frozen Exp236 alerts unchanged.
- [ ] Implement runner and dashboard catalog entry.
- [ ] Run the package tests and relevant dashboard registration test.
- [ ] Run a 10-dataset smoke with two workers and verify 200 prediction rows, no errors, no forbidden fields, and zero reaggregation mismatches.
- [ ] Register Exp251 behind Exp250 and launch the deferred runner.
- [ ] Verify the dashboard shows Exp250 running and Exp251 queued.
- [ ] Commit runner/dashboard integration.
