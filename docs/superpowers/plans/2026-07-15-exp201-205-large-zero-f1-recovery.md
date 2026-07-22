# Exp201-205 Large-Dataset Zero-F1 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, queue, and sequentially run five retrospective shadow experiments that attempt to reduce Exp137 Hard-alert Zero-F1 among datasets with at least 100 TRAIN-normal samples while requiring newly added candidates to reach 80% precision.

**Architecture:** A focused `experiments/exp201_205` package reads hash-verified Exp137, Exp190, Exp162, and Exp164 artifacts. Prediction creates label-free frozen shadow candidates; evaluation verifies the prediction hash before loading labels; Exp205 independently reaggregates all stage results. The existing rank queue and dashboard expose five separate sequential stages.

**Tech Stack:** Python 3, NumPy, SQLite, standard-library CSV/JSON/hash utilities, `unittest`, existing rank queue/dashboard infrastructure.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-15-exp201-205-large-zero-f1-recovery-design.md` exactly.
- Preserve Exp137 autonomous Hard indices byte-for-byte.
- Use exactly 251 datasets with `train_normal_count >= 100`; report `100-199` and `200+` separately.
- Freeze `alpha=0.01`; TEST metrics may not select alpha, sources, gates, or routing.
- Prediction must not read TEST labels, anomaly positions, Zero-F1 membership, family names, TEST length as policy input, or past family TEST performance.
- Added candidates are `shadow_hard_diagnostic`, never operational Hard alerts.
- Run one queued stage at a time with seven CPU workers where dataset work is independent.
- Do not modify or overwrite Exp137, Exp162, Exp164, or Exp186-200 outputs.

---

### Task 1: Policy algebra and frozen artifact boundary

**Files:**
- Create: `experiments/exp201_205/__init__.py`
- Create: `experiments/exp201_205/core.py`
- Create: `experiments/exp201_205/artifacts.py`
- Create: `experiments/exp201_205/test_exp201_205_core.py`
- Create: `experiments/exp201_205/test_exp201_205_structure.py`

**Interfaces:**
- Consumes: Exp137 lane rows, Exp190 and Exp198 prediction rows, Exp162 score-cache manifest/NPZ files, Exp164 TRAIN-risk summary.
- Produces: `SizeStratum`, `classify_size_stratum()`, `empirical_upper_tail_pvalues()`, `three_source_agreement()`, `cross_confirmed_candidates()`, `validate_prediction_payload()`, `FrozenInputs`, `load_frozen_inputs()`, `freeze_predictions()`, and `load_verified_predictions()`.

- [ ] **Step 1: Write failing core tests**

Test `classify_size_stratum(100) == "100-199"`, `classify_size_stratum(200) == "200+"`, exact three-of-three intersection, current/alternative cross-confirmation, Exp137-lane exclusion, out-of-range rejection, and evaluation-field rejection.

- [ ] **Step 2: Run the core tests and confirm RED**

Run: `PYTHONPATH=. python3 -m unittest experiments/exp201_205/test_exp201_205_core.py`

Expected: import failure because the package does not exist.

- [ ] **Step 3: Implement deterministic policy functions**

Use set algebra only. `three_source_agreement()` returns the exact intersection of the derivative-spectral, normal-shape-prototype, and interval-quantile candidate sets. `cross_confirmed_candidates()` intersects that set with the union of frozen current-source candidates. Every result excludes all Exp137 lanes.

- [ ] **Step 4: Write and implement artifact-boundary tests**

Assert 251 unique eligible datasets, complete source coverage, alpha contract `0.01`, input SHA verification, prediction hash tamper rejection, and no `y_test` parameter on prediction interfaces.

- [ ] **Step 5: Run Task 1 tests**

Run: `PYTHONPATH=. python3 -m unittest experiments/exp201_205/test_exp201_205_core.py experiments/exp201_205/test_exp201_205_structure.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add experiments/exp201_205
git commit -m "feat: add exp201-205 frozen policy core"
```

### Task 2: Label-free Exp201-204 prediction and hash-gated evaluation

**Files:**
- Create: `experiments/exp201_205/run_experiment_201_205.py`
- Modify: `experiments/exp201_205/test_exp201_205_core.py`
- Modify: `experiments/exp201_205/test_exp201_205_structure.py`

**Interfaces:**
- Consumes: `FrozenInputs`, fixed `alpha=0.01`, and Task 1 policy functions.
- Produces: `predict_dataset(stage, inputs, dataset) -> list[dict]`, frozen stage JSONL/manifests, dataset result CSVs, and stage summaries.

- [ ] **Step 1: Add failing stage-rule tests**

Cover exact rules:

```python
STAGE_RULES = {
    "201": "frozen_large_size_baseline",
    "202": "alternative_3of3_train_risk_rescue",
    "203": "current_alternative_cross_confirmation",
    "204": "size_stratified_shadow_router",
}
```

Assert Exp202 emits both preregistered variants, Exp203 requires frozen Exp198 support, and Exp204 uses cross-confirmation for `100-199` and Exp190 high+standard plus unique cross-confirmed rescue for `200+`.

- [ ] **Step 2: Run stage tests and confirm RED**

Run: `PYTHONPATH=. python3 -m unittest experiments/exp201_205/test_exp201_205_core.py experiments/exp201_205/test_exp201_205_structure.py`

- [ ] **Step 3: Implement prediction**

Prediction reads only frozen scores and indices. It writes `contains_test_labels=false`, `contains_evaluation_metrics=false`, `autonomous_alert=false`, `final_lane=shadow_hard_diagnostic`, source evidence, size stratum, and candidate indices. Exp201 writes baseline evidence rows without creating a new lane.

- [ ] **Step 4: Implement hash-gated evaluation**

Evaluation verifies the stage prediction SHA256 and only then loads TEST labels. Compute added TP/FP/precision, counterfactual union metrics, workload, Zero-F1 before/after/fixed/new, Wilson intervals, and paired bootstrap intervals. Keep autonomous and shadow scopes in separate columns.

- [ ] **Step 5: Run package tests**

Run: `PYTHONPATH=. python3 -m unittest experiments/exp201_205/test_exp201_205_core.py experiments/exp201_205/test_exp201_205_structure.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add experiments/exp201_205
git commit -m "feat: add exp201-204 frozen shadow stages"
```

### Task 3: Independent Exp205 reaggregation and reporting

**Files:**
- Create: `experiments/exp201_205/independent_reaggregate.py`
- Modify: `experiments/exp201_205/test_exp201_205_structure.py`

**Interfaces:**
- Consumes: hash-verified Exp201-204 result and summary CSVs.
- Produces: Exp205 result CSV, frontier summary CSV, overlap CSV, verification report, manifest, and completion marker.

- [ ] **Step 1: Add failing independent-arithmetic tests**

Test TP+FP=candidate count, before/fixed/new Zero-F1 identities, exact set overlaps, size-stratum totals, Wilson arithmetic, and rejection of any `best`, `optimal`, or `recommended` selection field.

- [ ] **Step 2: Run structure tests and confirm RED**

Run: `PYTHONPATH=. python3 -m unittest experiments/exp201_205/test_exp201_205_structure.py`

- [ ] **Step 3: Implement Exp205**

Read all stage CSVs independently, recompute global and size-stratified metrics, compare with each stage summary, and fail on any mismatch. Emit every frozen variant without choosing one from TEST performance.

- [ ] **Step 4: Run package tests**

Run: `PYTHONPATH=. python3 -m unittest discover -s experiments/exp201_205 -p 'test_*.py'`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add experiments/exp201_205
git commit -m "feat: add exp205 independent reaggregation"
```

### Task 4: Sequential queue and dashboard registration

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_queue_safety.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Consumes: stage IDs, result paths, summary paths, logs, expected dataset counts, and completion markers.
- Produces: `EXP201_205_BATCH`, `queue_exp201_205_batch()`, `queue-exp201-205`, five dashboard catalog entries, and live progress metadata.

- [ ] **Step 1: Add failing queue and dashboard tests**

Assert exact queue order Exp201 through Exp205, one active experiment maximum, worker count 7, labels/scopes, expected dataset count 251, summary/detail/log paths, and planned/queued/running visibility.

- [ ] **Step 2: Run queue/dashboard tests and confirm RED**

Run: `PYTHONPATH=. python3 scratch/test_rank_queue_safety.py && PYTHONPATH=. python3 scratch/test_rank_dashboard_operational.py`

- [ ] **Step 3: Register the five stages**

Add one experiment catalog row per stage. Add `queue-exp201-205` using the existing archive-before-run and state reconciliation behavior. Dashboard scopes must say `retrospective_shadow_hard_diagnostic`; Exp137 autonomous metrics remain unchanged.

- [ ] **Step 4: Run queue/dashboard tests**

Run: `PYTHONPATH=. python3 scratch/test_rank_queue_safety.py && PYTHONPATH=. python3 scratch/test_rank_dashboard_operational.py`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add run_rank_experiments_sequential.py serve_rank_dashboard.py scratch/test_rank_queue_safety.py scratch/test_rank_dashboard_operational.py
git commit -m "feat: register exp201-205 sequential batch"
```

### Task 5: Smoke verification, queue launch, and dashboard check

**Files:**
- Create during execution: `/Users/minho/Documents/Dataset/exp201_205_large_zero_f1_recovery/*`
- Create during execution: `/Users/minho/Documents/Dataset/experiment_201_*_results.csv` through `experiment_205_*_summary.csv`

**Interfaces:**
- Consumes: committed experiment package and frozen source artifacts.
- Produces: verified smoke outputs, five queued stages, detached sequential runner, and a reachable dashboard.

- [ ] **Step 1: Run static and unit verification**

```bash
python3 -m py_compile experiments/exp201_205/*.py run_rank_experiments_sequential.py serve_rank_dashboard.py
PYTHONPATH=. python3 -m unittest discover -s experiments/exp201_205 -p 'test_*.py'
PYTHONPATH=. python3 scratch/test_rank_queue_safety.py
PYTHONPATH=. python3 scratch/test_rank_dashboard_operational.py
git diff --check
```

- [ ] **Step 2: Run a ten-dataset smoke prediction/evaluation**

```bash
python3 experiments/exp201_205/run_experiment_201_205.py --stage 202 --dataset-limit 10 --output-root /tmp/exp201_205_smoke
```

Verify ten unique datasets, zero errors, no evaluation fields in frozen prediction JSONL, hash-gated evaluation, valid indices, and unchanged Exp137 Hard indices.

- [ ] **Step 3: Commit verification repairs if needed**

Only contract-preserving fixes made before full TEST evaluation may be committed. Do not alter alpha, source rules, size strata, or success gates.

- [ ] **Step 4: Register the batch**

```bash
RANK_EXPERIMENT_WORKERS=7 python3 run_rank_experiments_sequential.py queue-exp201-205
python3 run_rank_experiments_sequential.py list
```

Expected: Exp201, Exp202, Exp203, Exp204, and Exp205 are queued in that order.

- [ ] **Step 5: Start the sequential runner detached**

Use the repository's existing detached launcher pattern so the process survives the shell. Confirm only one stage is active and the runner PID is live.

- [ ] **Step 6: Restart and verify the dashboard without touching the experiment PID**

```bash
python3 dashboardctl.py restart
python3 dashboardctl.py status
curl -fsS http://127.0.0.1:8765/api/status
```

Expected: all five stages are visible, the active stage has live progress, and the experiment PID is unchanged by dashboard restart.

- [ ] **Step 7: Stop agent monitoring**

After confirming queue and dashboard visibility, leave the queue runner and dashboard running. Do not start a polling or monitoring agent; the user will observe progress and request analysis after completion.
