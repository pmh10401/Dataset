# Exp296 Cross-Gate Hard Confidence Consensus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the three frozen Exp295 Gate annotations into one mutually exclusive confidence tier per unchanged Hard alert.

**Architecture:** A focused package verifies Exp295, groups its three Gate rows by dataset, and creates one label-free consensus prediction row. Evaluation is a separate hash-verified phase that reads labels only after prediction freeze.

**Tech Stack:** Python standard library, NumPy, unittest, existing frozen-artifact contracts and dashboard catalog.

## Global Constraints

- Hard alert set remains exactly unchanged.
- No review promotion or Hard suppression.
- No TEST-based Gate or tier selection.
- Existing Exp291-295 artifacts remain read-only.

---

### Task 1: Consensus routing contract

**Files:**
- Create: `experiments/exp296_cross_gate_hard_confidence_consensus/routing.py`
- Create: `experiments/exp296_cross_gate_hard_confidence_consensus/tests/test_routing.py`

**Interfaces:**
- Produces: `build_consensus_records(dataset, gate_records_by_variant)`

- [ ] Write failing tests for all four tiers, Gate coverage, and Hard parity.
- [ ] Run the tests and verify the missing module fails.
- [ ] Implement minimal consensus routing.
- [ ] Run the tests and verify they pass.

### Task 2: Frozen prediction and evaluation

**Files:**
- Create: `experiments/exp296_cross_gate_hard_confidence_consensus/config.py`
- Create: `experiments/exp296_cross_gate_hard_confidence_consensus/pipeline.py`
- Create: `experiments/exp296_cross_gate_hard_confidence_consensus/run_pipeline.py`
- Create: `experiments/exp296_cross_gate_hard_confidence_consensus/tests/test_pipeline.py`
- Create: `scratch/run_exp296_cross_gate_hard_confidence_consensus.py`

**Interfaces:**
- Consumes: hash-verified Exp295 predictions.
- Produces: frozen predictions, manifests, result CSVs, status, and report.

- [ ] Write failing tests for one-row-per-dataset prediction and label-separated evaluation.
- [ ] Implement prediction, freeze, evaluation, reporting, and CLI.
- [ ] Run focused tests.

### Task 3: Dashboard and full verification

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Produces: completed Exp296 dashboard history and artifacts.

- [ ] Write and fail the registration test.
- [ ] Register Exp296 in the runner and dashboard.
- [ ] Execute all 1,117 datasets.
- [ ] Verify hashes, tier exclusivity, totals, unit tests, and dashboard API visibility.
