# Exp303 Expanded TRAIN-only Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare direct four-expert consensus with diversity and calibration-reliability TRAIN-only Gates over the frozen Standard candidate universe.

**Architecture:** Build Gate choices from hash-verified TRAIN artifacts, merge them with Exp294 candidate support without labels, freeze one prediction row per dataset, and evaluate labels only after hash verification.

**Tech Stack:** Python, NumPy, unittest, existing Exp293 artifact loaders and freeze contracts.

## Global Constraints

- TEST labels and metrics never enter prediction or Gate selection.
- Precision 80% is not a pass condition.
- No variant is selected from TEST results.
- Existing operational automatic alerts remain unchanged.
- New Gates reuse four verified sources and are not described as new independent evidence.

---

### Task 1: TRAIN-only Gate selection

**Files:**
- Create: `experiments/exp303_expanded_train_only_gates/gates.py`
- Create: `experiments/exp303_expanded_train_only_gates/tests/test_gates.py`

- [ ] Write failing tests for rank representatives, diversity pair choice, calibration rank choice, and abstention.
- [ ] Implement minimal deterministic Gate functions.
- [ ] Verify focused tests pass.

### Task 2: Candidate routing and frozen pipeline

**Files:**
- Create: `experiments/exp303_expanded_train_only_gates/routing.py`
- Create: `experiments/exp303_expanded_train_only_gates/pipeline.py`
- Create: `experiments/exp303_expanded_train_only_gates/run_pipeline.py`
- Create: `scratch/run_exp303_expanded_train_only_gates.py`
- Create: `experiments/exp303_expanded_train_only_gates/tests/test_pipeline.py`

- [ ] Write failing candidate parity, label separation, and metric tests.
- [ ] Implement eight fixed rules and prediction freeze.
- [ ] Implement post-freeze evaluation and reports.
- [ ] Run compile and package tests.

### Task 3: Dashboard and execution

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

- [ ] Add failing Exp303 catalog test.
- [ ] Register the experiment as retrospective and comparison-ineligible.
- [ ] Run all 1,117 datasets and independently verify hashes, coverage, and metrics.
- [ ] Restart the local dashboard and confirm Exp303 in `/api/overview`.
