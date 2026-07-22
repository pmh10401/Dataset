# Exp295 MoE Hard Alert Confidence Tiering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frozen retrospective experiment that annotates unchanged Exp291 Hard alerts with TRAIN-only MoE confidence tiers.

**Architecture:** A focused Exp295 package loads hash-verified Exp291, Exp293, and source-support artifacts. Prediction assigns mutually exclusive confidence tiers without labels, freezes rows, and evaluation loads labels only after hash verification.

**Tech Stack:** Python standard library, NumPy, unittest, existing experiment contracts and dashboard catalog.

## Global Constraints

- Existing Hard alerts remain unchanged.
- Review candidates are never promoted.
- All three Gate variants are reported; no TEST-based primary selection.
- Existing experiment artifacts are read-only.

---

### Task 1: Confidence classification contract

**Files:**
- Create: `experiments/exp295_moe_hard_confidence_tiering/routing.py`
- Create: `experiments/exp295_moe_hard_confidence_tiering/tests/test_routing.py`

**Interfaces:**
- Produces: `build_hard_confidence_records(dataset, hard_indices, source_support, selected_experts)`

- [ ] Write tests proving the three support counts map to exclusive tiers and preserve every Hard index.
- [ ] Run the tests and confirm they fail because the package does not exist.
- [ ] Implement the minimal routing function.
- [ ] Run the tests and confirm they pass.

### Task 2: Frozen prediction and retrospective evaluation

**Files:**
- Create: `experiments/exp295_moe_hard_confidence_tiering/config.py`
- Create: `experiments/exp295_moe_hard_confidence_tiering/run_pipeline.py`
- Create: `experiments/exp295_moe_hard_confidence_tiering/tests/test_pipeline.py`
- Create: `scratch/run_exp295_moe_hard_confidence_tiering.py`

**Interfaces:**
- Consumes: frozen Exp291 Hard rows, Exp293 Gate choices, and four source support maps.
- Produces: frozen prediction JSONL, manifests, result CSVs, summary CSV, and status JSON.

- [ ] Write tests for Hard parity, forbidden prediction fields, exclusive tiers, and evaluation-after-freeze.
- [ ] Run the tests and confirm the missing implementation fails.
- [ ] Implement prediction, freeze, evaluation, reporting, and CLI stages.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Dashboard registration and complete verification

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Produces: visible Exp295 history/status/results using the existing dashboard catalog.

- [ ] Write a dashboard registration test and confirm it fails.
- [ ] Register the runner and dashboard metadata.
- [ ] Run Exp295 prediction and evaluation over all 1,117 datasets.
- [ ] Run focused tests, dashboard tests, hash checks, and an independent aggregate comparison.
