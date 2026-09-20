# Exp391 FN/Zero-F1 Reachability Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify how much of Exp389/390's 2,003 FN is reachable by frozen existing evidence without changing the operational Hard lane.

**Architecture:** A read-only audit core verifies frozen input hashes, builds label-free candidate vote sets for six Alpha values, and only then joins TEST labels for retrospective evaluation. A runner writes dataset, rule, source, hotspot, manifest, status, and Korean report artifacts.

**Tech Stack:** Python 3 standard library, SQLite read-only URI, CSV, JSONL, SHA256, unittest.

## Global Constraints

- Automatic Hard alerts and all review lanes remain unchanged.
- Alpha values are exactly `(0.005, 0.010, 0.020, 0.050, 0.100, 0.200)`.
- Vote thresholds are exactly `(1, 2, 3)`.
- TEST metrics cannot select Alpha, vote threshold, source, family, or size rule.
- TEST labels are loaded only after frozen input verification.
- Every result is marked retrospective and not prospectively validated.

---

### Task 1: Candidate and category core

**Files:**
- Create: `experiments/exp391_fn_reachability/__init__.py`
- Create: `experiments/exp391_fn_reachability/fn_reachability_core.py`
- Test: `experiments/exp391_fn_reachability/test_exp391_fn_reachability.py`

**Interfaces:**
- Consumes: frozen Exp389 row, frozen Exp357 evidence row, TEST labels.
- Produces: `classify_baseline()`, `build_vote_candidates()`, and `evaluate_recovery()`.

- [ ] Write tests for zero-F1 category separation, exclusion of Hard/exact-normal indices, vote thresholds, and review overlap.
- [ ] Run the tests and confirm they fail because the module is absent.
- [ ] Implement the minimal pure functions.
- [ ] Run the tests and confirm they pass.

### Task 2: Read-only full audit runner

**Files:**
- Create: `experiments/exp391_fn_reachability/run_experiment_391_fn_reachability_audit.py`
- Modify: `experiments/exp391_fn_reachability/test_exp391_fn_reachability.py`

**Interfaces:**
- Consumes: Exp389 manifest/predictions, six Exp357 evidence files, Exp381 coverage CSV, read-only DB.
- Produces: Exp391 CSV, JSON, Markdown, status, and stdout artifacts.

- [ ] Add failing tests for SHA verification, label loading order flag, fixed grids, and aggregate invariants.
- [ ] Run the tests and verify the expected failures.
- [ ] Implement the runner with no policy selection.
- [ ] Run unit tests, a 10-dataset smoke test, and full 1,117-dataset audit.

### Task 3: Queue registration and verification

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Create: `docs/EXP391_PREREGISTRATION_KO.md`
- Create: `docs/EXP391_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: Exp391 runner and output paths.
- Produces: explicit `experiment_391_fn_zero_reachability_audit` queue entry.

- [ ] Add a structure test that resolves the Exp391 queue entry.
- [ ] Register Exp391 without changing legacy pending experiment status.
- [ ] Run `git diff --check`, unit tests, and independent result consistency checks.
- [ ] Queue and execute Exp391 explicitly; do not use `run-next`.
