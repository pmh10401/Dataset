# Exp304 Reconstruction Fifth Source Common-support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify the unique and consensus contribution of masked-reconstruction evidence on its frozen 251-dataset common-support population.

**Architecture:** Load hash-verified source support at alpha 0.005, restrict Exp294 rows to the exact Exp257 dataset set, freeze candidate-level five-source policy records, and evaluate labels only after prediction hashing.

**Tech Stack:** Python, NumPy, unittest, existing Exp292 source loaders and freeze contracts.

## Global Constraints

- Exactly 251 common-support datasets; no missing-as-empty behavior.
- Exp257 remains research-only due missing generator provenance.
- TEST labels and metrics never enter prediction.
- No mask variant, alpha, or policy is selected from TEST results.
- Operational automatic alerts remain unchanged.

---

### Task 1: Five-source routing

**Files:**
- Create: `experiments/exp304_reconstruction_fifth_source_common_support/routing.py`
- Create: `experiments/exp304_reconstruction_fifth_source_common_support/tests/test_routing.py`

- [ ] Write failing tests for k-of-5, reconstruction-confirmed, reconstruction-only, and missing-source rejection.
- [ ] Implement deterministic candidate policies.
- [ ] Verify focused tests pass.

### Task 2: Frozen prediction and evaluation

**Files:**
- Create: `experiments/exp304_reconstruction_fifth_source_common_support/config.py`
- Create: `experiments/exp304_reconstruction_fifth_source_common_support/pipeline.py`
- Create: `experiments/exp304_reconstruction_fifth_source_common_support/run_pipeline.py`
- Create: `scratch/run_exp304_reconstruction_fifth_source_common_support.py`
- Create: `experiments/exp304_reconstruction_fifth_source_common_support/tests/test_pipeline.py`

- [ ] Write failing coverage, label separation, and metric tests.
- [ ] Implement source/hash audit, prediction freeze, and post-freeze evaluation.
- [ ] Run compile and package tests.

### Task 3: Dashboard and complete run

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

- [ ] Add failing Exp304 registration test.
- [ ] Register common-support scope and provenance limitation.
- [ ] Execute 251 datasets and independently verify hashes, support coverage, and aggregates.
- [ ] Restart dashboard and confirm Exp304 is visible.
