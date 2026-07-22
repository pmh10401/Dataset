# Exp297 Bootstrap Confidence Stability Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure TRAIN-bootstrap stability of Exp296 confidence tiers and show frozen alert confidence in the dashboard.

**Architecture:** A deterministic bootstrap core recomputes TRAIN-only expert audits and Gate choices, then freezes one stability row per dataset. A dashboard snapshot reads only the frozen artifact and renders aggregate tiers plus a compact candidate table.

**Tech Stack:** Python, NumPy, unittest, existing frozen-artifact contracts, HTML/CSS/vanilla JavaScript dashboard.

## Global Constraints

- Bootstrap count is 32 and seed is 20260722.
- Hard alert set remains 2,218 and unchanged.
- TEST labels are evaluation-only.
- Confidence is explanatory; no lane change.

---

### Task 1: Deterministic bootstrap core

**Files:**
- Create: `experiments/exp297_bootstrap_confidence_stability/core.py`
- Create: `experiments/exp297_bootstrap_confidence_stability/tests/test_core.py`

- [ ] Write failing tests for exact fast LOO p-values, deterministic samples, Gate reselection, tier stability, and n_train below five.
- [ ] Implement the minimum deterministic core.
- [ ] Run focused tests.

### Task 2: Frozen Exp297 pipeline

**Files:**
- Create: `experiments/exp297_bootstrap_confidence_stability/config.py`
- Create: `experiments/exp297_bootstrap_confidence_stability/pipeline.py`
- Create: `experiments/exp297_bootstrap_confidence_stability/run_pipeline.py`
- Create: `experiments/exp297_bootstrap_confidence_stability/tests/test_pipeline.py`
- Create: `scratch/run_exp297_bootstrap_confidence_stability.py`

- [ ] Write failing prediction/evaluation separation and Hard parity tests.
- [ ] Implement hash-verified bootstrap prediction and separate evaluation.
- [ ] Register and execute Exp297 for all 1,117 datasets.

### Task 3: Dashboard confidence panel

**Files:**
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

- [ ] Write failing snapshot and registration tests.
- [ ] Add the frozen confidence snapshot to detailed API status.
- [ ] Add tier cards, filter, and alert table to the dashboard.
- [ ] Verify API fields, all dashboard tests, and local browser rendering.
