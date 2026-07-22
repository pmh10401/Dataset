# Exp289 Label Consistency Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit exact and near-duplicate waveform conflicts for the frozen Exp287 Recovery Priority population without changing any alert policy.

**Architecture:** A deterministic comparison module defines fixed duplicate tiers and nearest-reference matching. A read-only retrospective pipeline verifies the frozen Exp286 prediction hash, loads labels only for evaluation, compares 43 candidates with TRAIN normal and opposite-label TEST traces, and writes independently aggregatable CSV artifacts.

**Tech Stack:** Python 3.14, NumPy, SQLite read-only loader, CSV/JSON, unittest.

## Global Constraints

- Frozen population: Exp286 alpha 0.05 Recovery Priority, exactly 43 candidates.
- TEST labels are retrospective evaluation inputs only.
- Do not change Exp253 Hard, Recovery lanes, thresholds, or candidate indices.
- Do not select similarity thresholds from TP/FP results.
- Dashboard execution remains sequential.

---

### Task 1: Duplicate comparison engine

**Files:**
- Create: `experiments/exp289_label_consistency_audit/__init__.py`
- Create: `experiments/exp289_label_consistency_audit/comparison.py`
- Create: `experiments/exp289_label_consistency_audit/tests/__init__.py`
- Create: `experiments/exp289_label_consistency_audit/tests/test_comparison.py`

**Interfaces:**
- Produces: `compare_pair(left_raw, right_raw, left_aligned, right_aligned) -> PairComparison`
- Produces: `duplicate_tier(comparison) -> str`

- [ ] Write tests for raw exact, aligned exact, normalized equivalent, near duplicate, distinct, and constant rows.
- [ ] Run tests and verify they fail because the comparison module is missing.
- [ ] Implement fixed metrics and classification constants.
- [ ] Run tests and verify all comparison tests pass.
- [ ] Commit the comparison engine and tests.

### Task 2: Retrospective audit pipeline

**Files:**
- Create: `experiments/exp289_label_consistency_audit/audit.py`
- Create: `experiments/exp289_label_consistency_audit/run_pipeline.py`
- Create: `experiments/exp289_label_consistency_audit/tests/test_audit.py`
- Create: `scratch/run_exp289_label_consistency_audit.py`

**Interfaces:**
- Consumes: verified Exp286 prediction rows and read-only SQLite values/labels.
- Produces: candidate detail, summary, dataset summary, exact conflict, representative case, and status artifacts.

- [ ] Write tests that require exactly 43 frozen candidates, reject index errors, and aggregate mutually exclusive duplicate tiers.
- [ ] Run tests and verify the expected failures.
- [ ] Implement candidate matching, dataset-wide exact conflict hashing, summaries, and representative waveform export.
- [ ] Verify prediction before label access and mark every artifact retrospective-only.
- [ ] Run all Exp289 tests and commit the audit pipeline.

### Task 3: Sequential dashboard execution

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Produces: one queue entry and live status for `experiment_289_recovery_priority_label_consistency_audit`.

- [ ] Add a failing runner/dashboard registration test.
- [ ] Register Exp289 with its fixed output root and diagnostic metric scope.
- [ ] Run dashboard tests and verify the registration passes.
- [ ] Commit runner and dashboard integration.

### Task 4: Official run and report

**Files:**
- Create outside git: `/Users/minho/Documents/Dataset/exp289_recovery_priority_label_consistency_audit/RESULT_REPORT.md`
- Modify: `docs/superpowers/plans/2026-07-22-exp289-label-consistency-audit.md`

**Interfaces:**
- Consumes: official Exp289 CSV results.
- Produces: verified diagnosis and next-step decision with Hard unchanged.

- [ ] Queue and execute Exp289 once through the sequential runner.
- [ ] Independently verify 1,117 coverage, 43 candidates, zero errors, tier totals, and output hashes.
- [ ] Write the report without promoting a TEST-informed rule.
- [ ] Update plan status, run `git diff --check`, and commit the execution record.
