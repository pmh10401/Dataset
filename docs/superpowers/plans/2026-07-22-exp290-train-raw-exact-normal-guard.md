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

- [ ] Write failing tests for exact partitioning, lane invariants, and label-free prediction schema.
- [ ] Verify the tests fail for missing production modules.
- [ ] Implement Exp253 hash verification, read-only blob loading, and Hard partitioning.
- [ ] Run tests and commit the prediction contract.

### Task 2: Frozen evaluation and reporting

**Files:**
- Create: `experiments/exp290_train_raw_exact_normal_guard/evaluation.py`
- Create: `experiments/exp290_train_raw_exact_normal_guard/run_pipeline.py`
- Create: `experiments/exp290_train_raw_exact_normal_guard/tests/test_evaluation.py`
- Create: `scratch/run_exp290_train_raw_exact_normal_guard.py`

- [ ] Write failing tests for prediction hash tamper rejection and metric aggregation.
- [ ] Implement separate `predict` and `evaluate` phases plus an `all` orchestration phase.
- [ ] Write prediction/evaluation manifests, dataset results, summary, removed-detail, and status artifacts.
- [ ] Run all Exp290 tests and commit.

### Task 3: Dashboard registration and official run

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

- [ ] Add and run a failing focused registration test.
- [ ] Register Exp290 as an autonomous retrospective ablation with separate baseline and guarded metrics.
- [ ] Run the focused test, commit, restart dashboard, queue, and execute once.

### Task 4: Independent verification

**Files:**
- Create outside git: `/Users/minho/Documents/Dataset/exp290_train_raw_exact_normal_guard/RESULT_REPORT.md`
- Modify: `docs/superpowers/plans/2026-07-22-exp290-train-raw-exact-normal-guard.md`

- [ ] Recompute exact-match removals directly from SQLite blobs.
- [ ] Recompute baseline/guarded TP, FP, FN, precision, recall, and mean dataset F1.
- [ ] Verify hashes, coverage, lane partition, and zero errors.
- [ ] Record the result without tuning the rule and commit the completed plan.
