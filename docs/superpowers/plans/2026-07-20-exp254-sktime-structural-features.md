# Exp254 sktime Structural Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible 1,117-dataset score-source experiment for RandomInterval, MatrixProfile, SAX, and SFA structural features.

**Architecture:** Reuse Exp252's database, TRAIN-only sanitizer, KNN local-gap, conformal prediction, separated evaluation, and dashboard contracts. Add a fitted structural feature extractor that downsamples aligned series to at most 256 points and produces four fixed-width blocks plus a combined block.

**Tech Stack:** Python 3.14, sktime 1.0.1, NumPy, pandas, scikit-learn, unittest, local dashboard.

## Global Constraints

- Experiment ID is `experiment_254_sktime_structural_features`.
- Use all 1,117 frozen Exp236 datasets and fail on incomplete coverage.
- TRAIN normal data alone determines transforms, score reference, and calibration.
- Prediction must contain no TEST labels or evaluation metrics.
- Use seed 20260723, KNN-3, 5-fold OOF, and alpha 0.005/0.01/0.02/0.05.
- Keep Exp236 and Exp253 alert lanes unchanged.
- Mark all results retrospective and not deployment eligible.

---

### Task 1: Structural Feature Contract

**Files:**
- Create: `experiments/exp254_sktime_structural_features/config.py`
- Create: `experiments/exp254_sktime_structural_features/features.py`
- Test: `experiments/exp254_sktime_structural_features/tests/test_features.py`

**Interfaces:**
- Produces `StructuralFeatureExtractor.fit_transform_train(train) -> dict[str, np.ndarray]`.
- Produces `StructuralFeatureExtractor.transform(test) -> dict[str, np.ndarray]`.

- [ ] Write failing tests for fixed widths 224/16/72/128/440, deterministic output, TRAIN-fitted SFA, and MatrixProfile windows.
- [ ] Run the feature tests and confirm missing-module failures.
- [ ] Implement fixed-length interpolation, RandomIntervals, multi-scale MatrixProfile summaries, SAX histograms/transitions, SFA dictionary hashing, and combined features.
- [ ] Run the tests and verify finite identical-width TRAIN/TEST blocks.
- [ ] Commit with `feat: add Exp254 structural feature extractor`.

### Task 2: Cache and TRAIN-only Scoring

**Files:**
- Create: `experiments/exp254_sktime_structural_features/cache_pipeline.py`
- Create: `experiments/exp254_sktime_structural_features/scoring.py`
- Test: `experiments/exp254_sktime_structural_features/tests/test_scoring.py`

**Interfaces:**
- Produces per-dataset NPZ with `variant__train_oof` and `variant__test_scores`.
- Produces complete cache manifest or fails with dataset-level errors.

- [ ] Write failing tests for deterministic cross-fit, no TEST labels, and cache width checks.
- [ ] Confirm failures before implementation.
- [ ] Reuse Exp252 sanitizer and local-gap logic; fit every stateful transform on TRAIN only.
- [ ] Add atomic cache writes, input/config hashes, 7-worker progress, and coverage validation.
- [ ] Run tests and commit with `feat: add Exp254 TRAIN-only score cache`.

### Task 3: Frozen Prediction and Evaluation

**Files:**
- Create: `experiments/exp254_sktime_structural_features/prediction.py`
- Create: `experiments/exp254_sktime_structural_features/evaluation.py`
- Create: `experiments/exp254_sktime_structural_features/independent_reaggregate.py`
- Test: `experiments/exp254_sktime_structural_features/tests/test_evaluation.py`

**Interfaces:**
- Produces deterministic `frozen_predictions.jsonl` before labels are loaded.
- Produces results and summary for candidate, Hard confirmation, Standard, Priority, and outside-lane views.

- [ ] Write failing tests for hash tamper rejection, lane disjointness, and zero denominators.
- [ ] Confirm failures before implementation.
- [ ] Adapt Exp252 prediction/evaluation with experiment-specific paths and leakage flags.
- [ ] Add Exp253 remaining-Standard post-analysis without modifying prediction lanes.
- [ ] Implement read-only independent reaggregation and run tests.
- [ ] Commit with `feat: add Exp254 separated evaluation`.

### Task 4: Runner, Reporting, and Dashboard

**Files:**
- Create: `experiments/exp254_sktime_structural_features/run_pipeline.py`
- Create: `experiments/exp254_sktime_structural_features/reporting.py`
- Create: `scratch/run_exp254_sktime_structural_features.py`
- Modify: `serve_rank_dashboard.py`
- Test: `experiments/exp254_sktime_structural_features/tests/test_pipeline.py`

**Interfaces:**
- CLI stages: audit, cache, predict, evaluate, all.
- Dashboard ID: `experiment_254_sktime_structural_features`.

- [ ] Write failing CLI, queue, and dashboard registration tests.
- [ ] Confirm failures.
- [ ] Implement stage status, heartbeat, canonical queue state, Korean report, and manifest hashes.
- [ ] Run a 10-dataset smoke test and inspect prediction leakage/lane invariants.
- [ ] Run all tests and commit with `feat: add Exp254 pipeline and dashboard`.

### Task 5: Full Sequential Experiment

**Files:**
- Generate: `/Users/minho/Documents/Dataset/exp254_sktime_structural_features/*`

**Interfaces:**
- Produces full 1,117-dataset cache and 22,340 frozen prediction rows.

- [ ] Register Exp254 in the canonical one-at-a-time queue with 7 workers.
- [ ] Verify dashboard visibility and dataset-level progress.
- [ ] Complete audit, cache, prediction, evaluation, and independent reaggregation.
- [ ] Run all tests and `git diff --check`.
- [ ] Report all variants without selecting a deployment policy from TEST metrics.
