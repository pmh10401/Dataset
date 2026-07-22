# Exp306-307 Componentized Routing and Expanded MoE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze a provenance-separated Hard/Shadow router and evaluate a six-expert, method-group-aware MoE without changing the deployed automatic policy.

**Architecture:** Exp306 hash-verifies Exp236, Exp253, Exp290, and Exp291 and partitions the frozen benchmark Hard set. Exp307 reuses the Exp293 TRAIN audit for four experts, audits two new TRAIN caches, adds their frozen candidate evidence to Exp292 Standard candidates, and freezes review ranking plus Shadow diagnostics before evaluation.

**Tech Stack:** Python 3, NumPy, CSV/JSONL, unittest/pytest, existing SHA256 freeze contracts and dashboard runner.

## Global Constraints

- Dataset coverage is exactly 1,117 and Exp292 Standard candidate parity is exactly 806.
- Alpha is fixed at 0.005.
- TEST labels are unavailable to prediction and Gate selection.
- Exp236/253 research extensions and Exp307 Shadow remain non-autonomous.
- Existing results are never overwritten.
- New score computation order is fixed as `missing_repair -> train_length_alignment -> normalization`.

---

### Task 1: Exp306 component router

**Files:**
- Create: `experiments/exp306_componentized_operational_routing/`
- Test: `experiments/exp306_componentized_operational_routing/tests/`

**Interfaces:**
- Consumes: hash-pinned Exp236/253/290/291 JSONL artifacts.
- Produces: `build_componentized_row(...) -> dict[str, object]` and frozen 1,117-row predictions.

- [ ] Write tests proving lane disjointness, exact-guard application, and Exp291 union parity.
- [ ] Run tests and confirm they fail before implementation.
- [ ] Implement prediction, evaluation, manifests, and report generation.
- [ ] Run package tests and independently recompute aggregate lane counts.

### Task 1A: Shared preprocessing-order contract

**Files:**
- Modify: `experiments/exp226_235_hard_alert_quality/preprocessing.py`
- Modify: `experiments/exp239_248_diverse_score_sources/foundation.py`
- Test: `experiments/exp226_235_hard_alert_quality/tests/test_foundation.py`

**Interfaces:**
- Produces: `PREPROCESSING_ORDER` and `preprocessing_contract() -> dict[str, object]`.
- Enforces: missing repair before TRAIN-derived alignment and normalization after alignment.

- [ ] Add a failing test for the exact machine-readable order.
- [ ] Expose the immutable contract from the common preprocessing module.
- [ ] Record the contract in Exp307 prediction provenance.
- [ ] Run foundation and source-pipeline tests.

### Task 2: Exp307 six-expert audit and routing

**Files:**
- Create: `experiments/exp307_expanded_six_expert_moe/`
- Test: `experiments/exp307_expanded_six_expert_moe/tests/`

**Interfaces:**
- Consumes: Exp292 frozen review tiers, Exp293 audit CSV, Exp240/247 cache manifests and frozen predictions.
- Produces: six-expert audit CSV, gate choices JSONL, prediction JSONL, evaluation CSV, and result report.

- [ ] Test distinct-group expert selection, single-variant stability handling, ranking parity, and Shadow non-autonomy.
- [ ] Run tests and confirm failure before implementation.
- [ ] Implement hash and coverage audits for the two new experts.
- [ ] Implement three distinct-group Gate variants and candidate ranking.
- [ ] Freeze prediction before labels and evaluate fixed workloads plus 2/3-group Shadow diagnostics.
- [ ] Independently verify candidate parity, lane overlap, and aggregate metrics.

### Task 3: Queue and dashboard integration

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`
- Create: `scratch/run_exp306_componentized_operational_routing.py`
- Create: `scratch/run_exp307_expanded_six_expert_moe.py`

**Interfaces:**
- Consumes: experiment status, summary, result, and log artifacts.
- Produces: queue commands and dashboard catalog entries for Exp306/307.

- [ ] Write dashboard registration assertions.
- [ ] Register both experiments with expected dataset count 1,117.
- [ ] Run the complete dashboard operational test suite.
- [ ] Execute Exp306 then Exp307 sequentially and verify `/api/status` contains both experiments.
