# Exp165 Small-Sample Calibration And Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch the Exp165 TRAIN-only small-sample calibration audit with complete, reliable dashboard visibility.

**Architecture:** Exp165 consumes frozen Exp164 OOF caches, applies leave-query-out similarity-pooled calibration to 316 small-sample targets, and writes resumable audit artifacts. A shared atomic progress publisher connects standalone or queued experiments to the dashboard, while stale runtime state is normalized against live PIDs and completed artifacts.

**Tech Stack:** Python 3, NumPy, SQLite metadata audit, `unittest`, existing sequential runner, existing threaded HTTP dashboard.

## Global Constraints

- Do not read TEST values, labels, positions, lengths, or metrics in Exp165.
- Keep alphas `(0.005, 0.01, 0.02, 0.05)` and minimum pool reference `200` fixed.
- Keep `n_train < 5` as complete abstention.
- Do not choose an operating variant from these sensitivity results.
- Run exactly one experiment process with seven workers.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Exp165 Core Contract

**Files:**
- Create: `experiments/exp165/similarity_calibration_core.py`
- Test: `experiments/exp165/test_exp165_small_sample_calibration.py`

**Interfaces:**
- Produces robust score scaling, descriptor construction, deterministic donor selection, pooled thresholds, and agreement flags.

- [ ] Write tests for leave-query-out normalization and deterministic tie-preserving donor selection.
- [ ] Run tests and confirm they fail because the module is absent.
- [ ] Implement the minimum core functions.
- [ ] Run tests and confirm all core tests pass.

### Task 2: Exp165 Runner And Artifacts

**Files:**
- Create: `experiments/exp165/run_experiment_165_small_sample_calibration.py`
- Create: `experiments/exp165/test_exp165_structure.py`
- Create: `experiments/exp165/__init__.py`

**Interfaces:**
- Consumes frozen Exp164 contract, manifest, and cache files.
- Produces checkpointed target caches, dataset/global summaries, transition diagnostics, errors, contract, and manifest.

- [ ] Write structure tests rejecting TEST access and forbidden policy branches.
- [ ] Run tests and confirm the missing runner fails them.
- [ ] Implement input hash validation, target/donor partitioning, checkpoint resume, seven-worker execution, and summaries.
- [ ] Run unit and structure tests.

### Task 3: Shared Dashboard Progress

**Files:**
- Create: `experiment_progress.py`
- Test: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Produces `ExperimentProgress.start/update/finish/fail` writing atomic `rank_dashboard_external_live_run.json` records.

- [ ] Write a failing atomic progress lifecycle test.
- [ ] Implement the progress publisher without importing experiment code.
- [ ] Run dashboard tests.

### Task 4: Runtime Reconciliation And History

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Test: `scratch/test_rank_queue_safety.py`
- Test: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Registers Exp165 with the sequential runner.
- Adds historical entries for Exp153 through Exp165.
- Normalizes dead stale PIDs before queue and dashboard status decisions.

- [ ] Write failing tests for a dead state PID and missing Exp158-165 history.
- [ ] Implement stale-state reconciliation at the runner source.
- [ ] Add Exp165 runner registration and historical adapters.
- [ ] Verify audit-only experiments are excluded from F1 ranking.

### Task 5: Smoke And Full Queue Launch

**Files:**
- Output: `outputs/exp165_small_sample_calibration_smoke/`
- Output: `outputs/exp165_small_sample_calibration/`

- [ ] Compile all new Python files and run focused tests.
- [ ] Run a ten-target smoke with the full donor universe.
- [ ] Verify no errors, no TEST fields, minimum pool reference 200, and deterministic replay.
- [ ] Reconcile stale Exp153-157 state and restart the dashboard.
- [ ] Queue Exp165 and start one detached queue runner with seven workers.
- [ ] Verify `/api/status` reports Exp165, live PID, current dataset, progress, errors, elapsed time, and ETA.
