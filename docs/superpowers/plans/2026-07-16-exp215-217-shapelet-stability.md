# Exp215-217 Shapelet Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and queue TRAIN-only diagnostics that isolate random shapelet dictionaries, test deterministic normal medoids, and measure normal false-alert risk.

**Architecture:** A focused `experiments/exp215_217` package owns deterministic dictionaries, fold scoring, prediction/evaluation separation, and normal-only audit. The existing sequential runner and dashboard receive only experiment metadata and output paths.

**Tech Stack:** Python 3, NumPy, SQLite read-only dataset access, existing robust score helpers, unittest.

## Global Constraints

- Keep `/Users/minho/Documents/Dataset/univariate_ts.db` and DATA_DIR unchanged.
- TRAIN normal only determines representations, scores, thresholds, and risk.
- Do not use TEST labels, positions, family, dataset-name policy branches, or TEST length in prediction.
- Do not change `ALPHAS = (0.005, 0.01, 0.02, 0.05)`.
- Preserve Exp137 and all previous outputs.
- Mark every output retrospective research-only.

---

### Task 1: Deterministic dictionary and fold helpers

**Files:**
- Create: `experiments/exp215_217/__init__.py`
- Create: `experiments/exp215_217/core.py`
- Test: `experiments/exp215_217/test_exp215_217_core.py`

**Interfaces:**
- Produces `fixed_kernel_dictionary() -> list[np.ndarray]`.
- Produces `central_normal_medoid_dictionary(x_fit: np.ndarray) -> list[np.ndarray]`.
- Produces `fit_shape_source(mode, x_fit, x_query, seed) -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write failing tests** for repeatable fixed kernels, deterministic medoids, and fit-fold isolation.
- [ ] **Step 2: Run** `python3 -m unittest experiments.exp215_217.test_exp215_217_core` and confirm failure before implementation.
- [ ] **Step 3: Implement** fixed analytic kernels, deterministic central-window medoids, and train-fitted robust scoring.
- [ ] **Step 4: Run** the same unittest and require PASS.

### Task 2: Fold prediction and TRAIN-normal risk audit

**Files:**
- Create: `experiments/exp215_217/run_experiment_215_217.py`
- Create: `experiments/exp215_217/test_exp215_217_structure.py`

**Interfaces:**
- `--stage 215` writes fixed-dictionary transport audit results.
- `--stage 216` writes medoid transport audit and frozen TEST predictions.
- `--stage 217` writes TRAIN-normal false-alert audit results.

- [ ] **Step 1: Write failing structural tests** that prediction has no labels or metrics and Exp217 receives no TEST rows.
- [ ] **Step 2: Implement** stage-specific output generation with a frozen prediction manifest before evaluation.
- [ ] **Step 3: Run** core and structure unittests until PASS.
- [ ] **Step 4: Run** three-dataset smoke stages 215, 216, and 217 and inspect payload fields.

### Task 3: Queue and dashboard integration

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`

- [ ] **Step 1: Add** `experiment_215_fixed_dictionary_transport_control`, `experiment_216_normal_consensus_medoid_transport`, and `experiment_217_train_normal_shapelet_risk_audit` with expected cohort 251.
- [ ] **Step 2: Add** `queue-exp215-217`, preserving one active stage at a time.
- [ ] **Step 3: Add** dashboard records with retrospective research scope and explicit Exp137-unchanged status.
- [ ] **Step 4: Run** `python3 -m py_compile` for changed modules and `git diff --check`.

### Task 4: Full verification and sequential launch

**Files:**
- Create: `outputs/exp215_217_shapelet_stability/README.md`

- [ ] **Step 1: Run** all Exp211-214 and Exp215-217 unit tests.
- [ ] **Step 2: Run** full stages 215, 216, and 217 sequentially only after tests and smoke pass.
- [ ] **Step 3: Verify** output coverage, no errors, frozen prediction hashes, and queue/dashboard status.
- [ ] **Step 4: Commit** only implementation, documentation, tests, and this batch's small manifests; leave large CSV results in Dataset.

