# Exp249-250 Official MiniROCKET/MultiROCKET Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resumable official aeon MiniROCKET and MultiROCKET TRAIN-normal score-source qualification experiments over the frozen 1,117-dataset Exp236 population.

**Architecture:** A dedicated package cross-fits the transformer, robust feature scaler, and KNN-3 local-gap score for every seed/configuration. Prediction freezes label-free candidate rows and Exp236 lane intersections before a separate evaluation and independent reaggregation phase reads TEST labels.

**Tech Stack:** Python 3.14, NumPy, aeon 1.5.0, scikit-learn-compatible collection transforms, SQLite, unittest, existing Exp239-248 dashboard/state contracts.

## Global Constraints

- Keep Exp236 and Exp239-248 code/results unchanged.
- Use the exact transformer, seed, alpha, score, and lane contracts from the design document.
- Use no TEST labels, positions, lengths for budgets, family routing, or past family TEST performance during prediction.
- Run Exp249 before Exp250 with seven outer workers and one inner aeon thread.
- Do not claim prospective or deployment validation.

---

### Task 1: Configuration and transform contract

**Files:**
- Create: `experiments/exp249_250_official_mini_multirocket/config.py`
- Create: `experiments/exp249_250_official_mini_multirocket/transforms.py`
- Test: `experiments/exp249_250_official_mini_multirocket/tests/test_transforms.py`

**Interfaces:**
- Produces: `configuration(experiment: int, requested_kernels: int, seed: int)` and `fit_transform_scores(bundle, config)`.
- Consumes: shared `DatasetBundle`, deterministic folds, and Exp239 robust scaling utilities.

- [ ] Write tests that assert exact requested kernels, actual feature dimensions, fixed seeds, and no label argument.
- [ ] Run the tests and confirm failure because the package does not exist.
- [ ] Implement official aeon transform construction and TRAIN-only cross-fitting.
- [ ] Run transform tests and the existing Exp239-248 test suite.
- [ ] Commit the transform contract.

### Task 2: Resumable cache and frozen prediction

**Files:**
- Create: `experiments/exp249_250_official_mini_multirocket/cache_pipeline.py`
- Create: `experiments/exp249_250_official_mini_multirocket/prediction.py`
- Test: `experiments/exp249_250_official_mini_multirocket/tests/test_prediction.py`

**Interfaces:**
- Produces: per-dataset score NPZ/JSON caches and sorted `frozen_predictions.jsonl`.
- Consumes: two seed score vectors per requested-kernel configuration.

- [ ] Write tests for all four variants per experiment, explicit abstention rows, NumPy scalar normalisation, seed intersection, and forbidden evaluation fields.
- [ ] Run tests and confirm they fail for missing cache/prediction APIs.
- [ ] Implement atomic hash-verified caches and label-free prediction freezing.
- [ ] Run focused and existing tests.
- [ ] Commit cache and prediction.

### Task 3: Evaluation, reports, and independent reaggregation

**Files:**
- Create: `experiments/exp249_250_official_mini_multirocket/evaluation.py`
- Create: `experiments/exp249_250_official_mini_multirocket/reporting.py`
- Create: `experiments/exp249_250_official_mini_multirocket/independent_reaggregate.py`
- Test: `experiments/exp249_250_official_mini_multirocket/tests/test_evaluation.py`

**Interfaces:**
- Produces: `results.csv`, `summary.csv`, manifests, Korean result-ready report, and independent mismatch report.
- Consumes: a verified frozen prediction SHA256 and TEST labels only after verification.

- [ ] Write tests for hash tamper rejection, disjoint lane accounting, range checks, and gate formulas.
- [ ] Run tests and confirm expected failures.
- [ ] Implement post-freeze evaluation and independent reaggregation.
- [ ] Run focused and complete tests.
- [ ] Commit evaluation and reporting.

### Task 4: Runner, queue, and dashboard

**Files:**
- Create: `experiments/exp249_250_official_mini_multirocket/run_pipeline.py`
- Create: `scratch/run_exp249_250_sequential.py`
- Create: `scratch/exp249_250_smoke_datasets.txt`
- Modify: `serve_rank_dashboard.py`
- Test: `experiments/exp249_250_official_mini_multirocket/tests/test_queue_dashboard.py`

**Interfaces:**
- Produces: audit/cache/predict/evaluate/all CLI stages and canonical dashboard queue state.
- Consumes: the existing dashboard heartbeat and external-live-run JSON contract.

- [ ] Write failing tests for Exp249-before-Exp250 order and dashboard catalog names.
- [ ] Implement direct-script path bootstrap, sequential lock/state updates, and dashboard entries.
- [ ] Run syntax, unit, and `git diff --check` verification.
- [ ] Commit queue/dashboard support.

### Task 5: Smoke and full launch

**Files:**
- Generate externally: `/Users/minho/Documents/Dataset/exp249_250_official_mini_multirocket_smoke/`
- Generate externally: `/Users/minho/Documents/Dataset/exp249_250_official_mini_multirocket/`

**Interfaces:**
- Produces: 10-dataset smoke evidence, then the 1,117-dataset sequential run.
- Consumes: frozen source code, config hashes, DB, and Exp236 baseline hash.

- [ ] Run all tests and py_compile.
- [ ] Run Exp249 and Exp250 smoke on the fixed 10-dataset list.
- [ ] Verify all variants have 10 rows per alpha, no missing datasets, no forbidden prediction fields, and zero independent mismatches.
- [ ] Confirm no other experiment process or canonical queue is active.
- [ ] Start the full sequential queue and verify the dashboard shows Exp249 running and Exp250 queued.

