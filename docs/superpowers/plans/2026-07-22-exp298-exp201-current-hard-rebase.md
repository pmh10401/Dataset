# Exp298 Exp201 Candidate Rebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-evaluate the frozen Exp201 large-dataset candidate set against the current Exp292 automatic Hard baseline without changing the deployed alert policy.

**Architecture:** Prediction reads only the hash-pinned Exp201 and Exp292 prediction artifacts, removes candidates already present in current Hard, and freezes a 251-dataset residual candidate file. Evaluation verifies that frozen hash before loading TEST labels and reports added-candidate and counterfactual-union metrics for `100-199`, `200+`, and `100+` TRAIN-size strata.

**Tech Stack:** Python 3, JSONL/CSV artifacts, NumPy, unittest.

## Global Constraints

- Existing Exp201, Exp292, and Exp297 artifacts remain unchanged.
- Prediction must not read TEST labels, anomaly positions, family names, or past family performance.
- Current Exp292 automatic Hard remains the autonomous baseline; Exp298 is retrospective shadow evaluation only.
- The 80% added-precision threshold is a reporting gate, not a TEST-selected threshold.
- Results must be visible through the Vercel snapshot exporter without starting the local dashboard.

---

### Task 1: Frozen candidate rebasing

**Files:**
- Create: `experiments/exp298_exp201_current_hard_rebase/core.py`
- Test: `experiments/exp298_exp201_current_hard_rebase/test_exp298.py`

**Interfaces:**
- Consumes: Exp201 `candidate_indices`, Exp292 `automatic_hard_indices` and review lanes.
- Produces: `rebase_candidate_row(exp201_row, exp292_row) -> dict`.

- [ ] Write tests proving current-Hard overlap is removed, review overlap is retained and classified, and forbidden evaluation fields are absent.
- [ ] Run the test and confirm failure because the package is not implemented.
- [ ] Implement the minimum rebase logic with index and dataset contract checks.
- [ ] Run the test and confirm it passes.

### Task 2: Prediction/evaluation boundary and reporting

**Files:**
- Create: `experiments/exp298_exp201_current_hard_rebase/run_pipeline.py`
- Create: `experiments/exp298_exp201_current_hard_rebase/__init__.py`
- Create: `scratch/run_exp298_exp201_current_hard_rebase.py`
- Test: `experiments/exp298_exp201_current_hard_rebase/test_exp298.py`

**Interfaces:**
- Consumes: hash-pinned frozen input JSONL files and SQLite labels only during evaluation.
- Produces: frozen predictions, detailed CSV, summary CSV, manifest, status and result report.

- [ ] Write tests for TP/FP/FN, Zero-F1 repair and stratum aggregation.
- [ ] Confirm the tests fail before implementation.
- [ ] Implement separate `predict` and `evaluate` phases with prediction SHA verification.
- [ ] Confirm focused tests and compilation pass.

### Task 3: Vercel monitoring registration and full run

**Files:**
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`
- Modify: `run_rank_experiments_sequential.py`

**Interfaces:**
- Consumes: Exp298 status/detail/summary artifacts.
- Produces: Vercel snapshot queue/result visibility.

- [ ] Add a failing registration test for Exp298 metadata.
- [ ] Register Exp298 in the runner and dashboard catalog.
- [ ] Run unit tests, smoke prediction/evaluation, and then the full 251-dataset replay.
- [ ] Verify input/output hashes, zero errors, 251/251 coverage, and Vercel exporter heartbeat.
