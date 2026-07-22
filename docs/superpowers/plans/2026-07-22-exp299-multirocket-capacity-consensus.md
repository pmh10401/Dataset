# Exp299 MultiROCKET Capacity Consensus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether frozen official MultiROCKET 512- and 1024-kernel capacity blocks jointly identify precise residual candidates in the 251 large TRAIN-normal datasets.

**Architecture:** Prediction hash-verifies Exp250 official MultiROCKET and Exp298 current-Hard rebase artifacts. For each frozen alpha, it intersects the seed-intersection candidates from the two capacity blocks, removes current Exp292 Hard, and freezes component plus consensus indices before evaluation reads labels.

**Tech Stack:** Python 3, NumPy, JSONL, CSV, unittest.

## Global Constraints

- Primary alpha is frozen at `0.005`; `0.01`, `0.02`, and `0.05` are sensitivity outputs only.
- No TEST metric selects alpha, capacity, threshold, or routing.
- Current Exp292 automatic Hard remains unchanged.
- Exp299 is retrospective shadow evaluation, not prospective validation.
- Exp250 and Exp298 artifacts are read-only and hash-pinned.

---

### Task 1: Capacity consensus contract

**Files:**
- Create: `experiments/exp299_multirocket_capacity_consensus/core.py`
- Create: `experiments/exp299_multirocket_capacity_consensus/test_exp299.py`

- [ ] Write failing tests for exact 512/1024 intersection, current-Hard exclusion, component provenance, and range validation.
- [ ] Implement the minimum label-free prediction row builder.
- [ ] Run tests and verify green.

### Task 2: Frozen prediction and evaluation

**Files:**
- Create: `experiments/exp299_multirocket_capacity_consensus/run_pipeline.py`
- Create: `scratch/run_exp299_multirocket_capacity_consensus.py`

- [ ] Write failing tests for component/consensus TP-FP metrics and zero-F1 repair.
- [ ] Implement hash-pinned prediction, prediction-SHA-verified evaluation, summaries and report.
- [ ] Run focused tests and compilation.

### Task 3: Vercel registration and execution

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

- [ ] Add failing Exp299 catalog test.
- [ ] Register the 251-dataset experiment and queue command.
- [ ] Execute the frozen replay, independently verify arithmetic, and confirm Vercel recent-completion visibility.
