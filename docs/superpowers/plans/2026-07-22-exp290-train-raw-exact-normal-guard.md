# Exp290 TRAIN Raw Exact Normal Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Exp253 Hard candidates that are byte-identical to same-dataset TRAIN normal instances, with prediction frozen before label evaluation.

**Architecture:** A read-only blob matcher verifies Exp253 and partitions each frozen Hard lane without labels. A separate evaluator verifies the new prediction hash, then loads TEST labels and independently reports autonomous Hard changes.

**Tech Stack:** Python 3.14, SQLite read-only URI, SHA256, JSONL/CSV, unittest.

## Global Constraints

- Baseline coverage is 1,117 datasets and 2,253 Exp253 Hard alerts.
- Exact means byte-for-byte `values_blob` equality only.
- Prediction must not SELECT or receive TEST labels.
- Do not modify existing experiment artifacts.
- Run once through the sequential dashboard queue.

---

### Task 1: Exact-match prediction contract

**Files:**
- Create: `experiments/exp290_train_raw_exact_normal_guard/__init__.py`
- Create: `experiments/exp290_train_raw_exact_normal_guard/contracts.py`
- Create: `experiments/exp290_train_raw_exact_normal_guard/prediction.py`
- Create: `experiments/exp290_train_raw_exact_normal_guard/tests/test_prediction.py`

- [x] Write failing tests for exact partitioning, lane invariants, and label-free prediction schema.
- [x] Verify the tests fail for missing production modules.
- [x] Implement Exp253 hash verification, read-only blob loading, and Hard partitioning.
- [x] Run tests and commit the prediction contract.

### Task 2: Frozen evaluation and reporting

**Files:**
- Create: `experiments/exp290_train_raw_exact_normal_guard/evaluation.py`
- Create: `experiments/exp290_train_raw_exact_normal_guard/run_pipeline.py`
- Create: `experiments/exp290_train_raw_exact_normal_guard/tests/test_evaluation.py`
- Create: `scratch/run_exp290_train_raw_exact_normal_guard.py`

- [x] Write failing tests for prediction hash tamper rejection and metric aggregation.
- [x] Implement separate `predict` and `evaluate` phases plus an `all` orchestration phase.
- [x] Write prediction/evaluation manifests, dataset results, summary, removed-detail, and status artifacts.
- [x] Run all Exp290 tests and commit.

### Task 3: Dashboard registration and official run

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

- [x] Add and run a failing focused registration test.
- [x] Register Exp290 as an autonomous retrospective ablation with separate baseline and guarded metrics.
- [x] Run the focused test, commit, restart dashboard, queue, and execute once.

### Task 4: Independent verification

**Files:**
- Create outside git: `/Users/minho/Documents/Dataset/exp290_train_raw_exact_normal_guard/RESULT_REPORT.md`
- Modify: `docs/superpowers/plans/2026-07-22-exp290-train-raw-exact-normal-guard.md`

- [x] Recompute exact-match removals directly from SQLite blobs.
- [x] Recompute baseline/guarded TP, FP, FN, precision, recall, and mean dataset F1.
- [x] Verify hashes, coverage, lane partition, and zero errors.
- [x] Record the result without tuning the rule and commit the completed plan.

## Execution Result

- Completed: 2026-07-22 11:08:01 KST
- Coverage: 1,117 / 1,117 datasets
- Baseline Hard: 2,253 = 1,915 TP + 338 FP
- Guarded Hard: 2,218 = 1,915 TP + 303 FP
- Automatic normal decisions: 35 = 0 TP + 35 FP
- FP reduction: 35 / 338 (10.355%)
- Precision: 84.9978% -> 86.3390% (+1.3413 percentage points)
- Mean dataset Hard F1: 0.608659 -> 0.608983
- Recall: unchanged
- Prediction labels/metrics: absent
- Errors: 0
- Interpretation: strong retrospective operating candidate; not prospective or end-to-end strict TRAIN-only validation
