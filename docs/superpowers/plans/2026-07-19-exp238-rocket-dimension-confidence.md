# Exp238 ROCKET Dimension Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable TRAIN-only 256/512/1,024-kernel ROCKET qualification pipeline that labels frozen Exp236 alerts by confidence without changing lanes.

**Architecture:** Dataset workers generate nested ROCKET features once, fit KNN-3 local-gap score models, and write hashed label-free score caches. A separate prediction phase combines frozen Exp236 lanes, ROCKET stability, and frozen Exp237 independent evidence into disjoint confidence and review-priority strata; evaluation runs only after the prediction hash is frozen.

**Tech Stack:** Python 3, NumPy, scikit-learn, SQLite, JSONL/NPZ/CSV, unittest.

## Global Constraints

- Kernel counts `(256, 512, 1024)`, seeds `(20260706, 20261706)`, and alpha grid `(0.005, 0.01, 0.02, 0.05)` are frozen before evaluation.
- Exp236 Hard, Standard, and Priority candidate sets must remain exactly unchanged.
- TEST labels, family names, and TEST performance cannot affect prediction.

### Task 1: Core score and confidence contracts

**Files:**
- Create: `experiments/exp238_rocket_dimension_confidence/core.py`
- Create: `experiments/exp238_rocket_dimension_confidence/test_exp238.py`

- [ ] Write failing tests for nested dimensions, stable evidence, disjoint confidence partitions, and frozen lane preservation.
- [ ] Implement minimal core functions and run tests green.

### Task 2: Resumable score cache and frozen prediction

**Files:**
- Create: `experiments/exp238_rocket_dimension_confidence/run_experiment_238.py`
- Modify: `experiments/exp238_rocket_dimension_confidence/test_exp238.py`

- [ ] Add failing tests for label-free prediction and SHA tamper detection.
- [ ] Implement seven-worker cache, prediction, evaluation, heartbeat, and independent reaggregation.
- [ ] Run compile and unit tests.

### Task 3: Smoke, full execution, and reporting

**Files:**
- Create: `outputs/exp238_rocket_dimension_confidence/RESULT_REPORT.md`
- Modify: `serve_rank_dashboard.py`

- [ ] Run a 10-dataset smoke with two workers.
- [ ] Run 1,117 datasets with seven workers and verify zero missing/error datasets.
- [ ] Independently reaggregate metrics and verify exact lane preservation.
- [ ] Add a non-comparison-eligible dashboard history entry and record hashes and limitations.
