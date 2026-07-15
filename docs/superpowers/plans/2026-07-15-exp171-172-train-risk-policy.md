# Exp171/172 TRAIN-normal Risk Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one Exp169 review-only policy under a simultaneous TRAIN-normal 1% risk contract and evaluate that frozen policy retrospectively as Exp172.

**Architecture:** A small policy core recomputes simultaneous Wilson bounds and deterministically selects one configuration. A phase-based runner separates audit, label-free prediction freezing, and label-reading evaluation; an independent script reaggregates the frozen result.

**Tech Stack:** Python 3 standard library, NumPy, SQLite, existing Exp169/170 artifacts and project helpers.

## Global Constraints

- Normal review-request risk target is exactly `0.01` per virtual normal run.
- Use a one-sided `95%` Wilson bound with Bonferroni correction over exactly 60 configurations.
- Use only Exp169 TRAIN-normal OOB audit values to select policy.
- Never use TEST labels, positions, metrics, family names, dataset names, or TEST length to select policy.
- Keep `ALPHAS = (0.005, 0.01, 0.02, 0.05)` and the existing 3 x 5 pool/source grid unchanged.
- Exp172 is review-only retrospective evaluation; autonomous Hard alert metrics are not evaluated.
- Existing Exp169/170 code and outputs must not be overwritten.

---

### Task 1: Policy Core and Contract Tests

**Files:**
- Create: `experiments/exp171_172/__init__.py`
- Create: `experiments/exp171_172/policy_core.py`
- Create: `experiments/exp171_172/test_exp171_172_policy_core.py`

**Interfaces:**
- Produces: `simultaneous_wilson_upper(successes, trials, comparisons, confidence)`.
- Produces: `select_train_risk_policy(rows, dataset_rows, risk_target=0.01)`.
- Produces: a deterministic policy dictionary or explicit abstention.

- [ ] Write failing tests for the Wilson formula, 60-key completeness, eligibility, deterministic tie-breaking, row-order invariance, and no-eligible abstention.
- [ ] Run `python3 -m unittest experiments/exp171_172/test_exp171_172_policy_core.py` and confirm the missing module/API failure.
- [ ] Implement the minimal pure policy core without filesystem or TEST dependencies.
- [ ] Re-run the test and confirm all cases pass.
- [ ] Commit with `feat: add Exp171 train risk policy core`.

### Task 2: Phase-separated Runner and Structure Tests

**Files:**
- Create: `experiments/exp171_172/run_experiment_171_172_train_risk_policy.py`
- Create: `experiments/exp171_172/test_exp171_172_structure.py`

**Interfaces:**
- Consumes: Exp169 `04_exp169_dataset_audit.csv`, `05_exp169_global_summary.csv`, and MANIFEST.
- Consumes: Exp170 `06_frozen_predictions.jsonl` only during prediction.
- Produces: files `01` through `10` from the design.
- CLI: `--phase audit|predict|evaluate|all`, `--output-dir`, and optional `--dataset-limit` for smoke tests.

- [ ] Write failing AST and behavior tests proving policy selection cannot read Exp170 metrics, prediction cannot read labels, evaluation verifies hashes before labels, and forbidden routing inputs are absent.
- [ ] Run the structure test and confirm failure for the missing runner.
- [ ] Implement `audit`, `predict`, and `evaluate` with strict hash/coverage gates.
- [ ] Ensure prediction JSONL has exactly one selected row per dataset and no evaluation fields.
- [ ] Re-run policy and structure tests.
- [ ] Commit with `feat: add Exp171-172 phase separated runner`.

### Task 3: Independent Reaggregation and Documentation

**Files:**
- Create: `experiments/exp171_172/independent_reaggregate_exp171_172.py`
- Create: `experiments/exp171_172/test_exp171_172_reaggregation.py`
- Create: `docs/EXP171_172_PREREGISTRATION_KO.md`
- Create: `docs/EXP171_172_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: frozen policy, prediction manifest, prediction JSONL, dataset metrics, and global summary.
- Produces: `11_independent_reaggregation.json` with mismatch count and hashes.

- [ ] Write failing tests for policy-hash tampering, prediction-hash tampering, 316-row coverage, index bounds, and independent metric sums.
- [ ] Run the reaggregation test and confirm failure for the missing implementation.
- [ ] Implement the independent checker and deterministic JSON output.
- [ ] Document the fixed risk contract, commands, review-only scope, and retrospective limitation in Korean.
- [ ] Run all Exp171/172 tests.
- [ ] Commit with `test: add Exp171-172 independent verification`.

### Task 4: Dashboard and Queue Registration

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`
- Modify: `scratch/test_rank_queue_safety.py`

**Interfaces:**
- Produces: queue/history labels for Exp171 audit and Exp172 frozen retrospective evaluation.

- [ ] Add failing tests that require both experiments to appear with research-review scope and no autonomous Hard metric.
- [ ] Run the dashboard and queue tests and confirm the missing-registration failure.
- [ ] Add minimal queue/history registrations and progress metadata.
- [ ] Re-run dashboard and queue tests.
- [ ] Commit with `feat: register Exp171-172 policy validation`.

### Task 5: Smoke, Full Run, and Final Verification

**Files:**
- Create through execution: `outputs/exp171_172_train_risk_policy/*`

**Interfaces:**
- Consumes: frozen Exp169/170 artifacts from `main`.
- Produces: complete outputs and MANIFEST with file hashes and commands.

- [ ] Run a 10-dataset smoke audit/predict/evaluate in `outputs/exp171_172_train_risk_policy_smoke` without modifying full outputs.
- [ ] Inspect smoke prediction rows for labels and evaluation metrics.
- [ ] Run full `audit`, `predict`, `evaluate`, and independent reaggregation sequentially.
- [ ] Verify the selected policy was determined before TEST evaluation and was not changed afterward.
- [ ] Run the Exp164-172 regression suite, dashboard tests, queue tests, `py_compile`, and `git diff --check`.
- [ ] Write `08_invariance_report.md`, `09_result_analysis.md`, completion JSON, and MANIFEST.
- [ ] Commit with `results: add Exp171-172 frozen policy evaluation`.

