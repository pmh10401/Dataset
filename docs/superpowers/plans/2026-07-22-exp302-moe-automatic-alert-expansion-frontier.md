# Exp302 MoE Automatic Alert Expansion Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate four nested evidence-based automatic-alert expansion rules across all 1,117 datasets without an 80% pass threshold.

**Architecture:** Merge the three frozen Exp294 Gate records by dataset and Standard candidate index, classify every candidate using four fixed nested rules, freeze predictions before labels, and evaluate each counterfactual lane independently.

**Tech Stack:** Python, NumPy, unittest, existing SHA256 freeze contracts, existing dashboard catalog.

## Global Constraints

- Exp294 frozen predictions are read-only and hash-verified.
- Prediction never reads TEST labels or evaluation metrics.
- No global Top-N, TEST-length, family, or dataset-name routing.
- No rule is selected from TEST results.
- Existing automatic policy remains unchanged outside counterfactual outputs.

---

### Task 1: Nested candidate policy

**Files:**
- Create: `experiments/exp302_moe_automatic_alert_expansion_frontier/routing.py`
- Create: `experiments/exp302_moe_automatic_alert_expansion_frontier/tests/test_routing.py`

- [ ] Write failing tests for nested rules, Gate coverage, and candidate parity.
- [ ] Run the focused test and verify the missing implementation failure.
- [ ] Implement the minimal candidate classifier and dataset merger.
- [ ] Run the focused test and verify all assertions pass.

### Task 2: Frozen pipeline

**Files:**
- Create: `experiments/exp302_moe_automatic_alert_expansion_frontier/config.py`
- Create: `experiments/exp302_moe_automatic_alert_expansion_frontier/pipeline.py`
- Create: `experiments/exp302_moe_automatic_alert_expansion_frontier/run_pipeline.py`
- Create: `scratch/run_exp302_moe_automatic_alert_expansion_frontier.py`
- Create: `experiments/exp302_moe_automatic_alert_expansion_frontier/tests/test_pipeline.py`

- [ ] Write failing prediction/evaluation separation and metric tests.
- [ ] Implement frozen prediction, post-freeze evaluation, CSV reporting, and status publishing.
- [ ] Run package tests and compile checks.

### Task 3: Dashboard and full execution

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

- [ ] Add a failing Exp302 registration test.
- [ ] Register Exp302 as retrospective counterfactual and comparison-ineligible.
- [ ] Execute all 1,117 datasets.
- [ ] Independently verify coverage, hashes, lane nesting, and aggregate metrics.
- [ ] Confirm Exp302 appears in the local dashboard API.
