# Exp424 MPS Score Cache Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Exp423's repeated OOF NumPy scoring and MPS model materialization while preserving frozen decisions and keeping the run review-only.

**Architecture:** Add an immutable `HorizonScoreCalibration` that validates one fixed OOF horizon once and scores queries with the same formulas. Add a batched residual predictor that materializes each trained model once per batch call, then wire both into the review-only Exp424 runner. The runner records stage/query progress and retains explicit CPU-contract/MPS-review metadata.

**Tech Stack:** Python 3.14, NumPy, PyTorch 2.13.0, Apple MPS, unittest, existing Exp417 checkpoint contracts.

## Global Constraints

- TRAIN-only model fitting and OOF calibration; TEST labels remain unopened during prediction.
- Exp417 official CPU artifacts and operational lanes remain unchanged.
- Exp424 is `review_only=true`, `official_prediction=false`, and uses one MPS-heavy process.
- Keep `np.percentile(..., method="linear")`, float64 score arrays, frozen seeds, horizons, thresholds, and routing rules unchanged.
- Adopt an optimization only after finite-output, candidate-coordinate, score-tolerance, and wall-time checks pass.
- Preserve the stopped Exp423 checkpoint and log; do not silently reuse them under changed provenance.

---

### Task 1: Implement immutable horizon calibration cache

**Files:**
- Modify: `experiments/exp417_causal_multihorizon_shadow/scoring.py`
- Test: `experiments/exp417_causal_multihorizon_shadow/tests/test_scoring.py`

**Interfaces:**
- Produces `HorizonScoreCalibration` with `from_oof(oof_residuals, horizon)` and `score(query_residual) -> HorizonScore`.
- Keeps `score_horizon()` as the reference implementation for parity tests.

- [ ] **Step 1: Write the failing tests**

Add tests that construct two finite OOF vectors and assert that
`HorizonScoreCalibration.from_oof(...).score(query)` matches `score_horizon(...)` for
score, p-value, persistence, and horizon. Add a test that rejects zero MAD, nonfinite,
negative, empty, and misaligned query residuals.

- [ ] **Step 2: Run the focused test to verify it fails**

Run:
`PATH=/usr/bin:/bin /Users/mac/Documents/Time\ Series/.venv/bin/python -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_scoring.HorizonScoringTests -v`

Expected: FAIL because `HorizonScoreCalibration` is not defined.

- [ ] **Step 3: Implement the minimal cache**

Add a frozen dataclass storing only the validated horizon, OOF count, center, scale,
threshold, and immutable float64 OOF scores. Build those fields with the exact operations
already used by `score_horizon()`. In `score()`, validate one query, normalize it, call the
same q95 and persistence helpers, and compute the plus-one p-value from cached OOF scores.

- [ ] **Step 4: Run the focused test to verify it passes**

Run the focused unittest again and expect PASS, then run the complete `test_scoring.py`.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow/scoring.py \
  experiments/exp417_causal_multihorizon_shadow/tests/test_scoring.py
git commit -m "perf(exp424): cache frozen horizon score calibration"
```

### Task 2: Attach calibration evidence to model evidence and runner scoring

**Files:**
- Modify: `experiments/exp417_causal_multihorizon_shadow/runner.py`
- Modify: `experiments/exp417_causal_multihorizon_shadow/tests/test_runner.py`

**Interfaces:**
- `TrainModelEvidence.score_calibrations` is an immutable mapping keyed by length, seed,
  and horizon, built once from `oof_bundles`.
- `_residual_decisions(..., calibrations=None)` accepts the cache while retaining a reference
  fallback for synthetic tests.

- [ ] **Step 1: Write the failing integration test**

Build the existing synthetic `TrainModelEvidence`, assert that every usable bundle has one
calibration per frozen seed and horizon, and compare `_residual_decisions` with and without
the cache for identical decisions.

- [ ] **Step 2: Run the focused runner tests and verify the new assertion fails**

Run:
`PATH=/usr/bin:/bin /Users/mac/Documents/Time\ Series/.venv/bin/python -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_runner -v`

Expected: FAIL because `score_calibrations` is absent.

- [ ] **Step 3: Build and use the cache once per evidence object**

Compute calibration mappings in `TrainModelEvidence.__post_init__` after bundle validation.
Use `score_horizon_cached` in normal OOF qualification, synthetic qualification, and TEST
value scoring. Leave the label-free SQL and contract order unchanged.

- [ ] **Step 4: Run focused and regression tests**

Run `test_runner.py` and `test_scoring.py`; expect PASS and unchanged decision fixtures.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow/runner.py \
  experiments/exp417_causal_multihorizon_shadow/tests/test_runner.py
git commit -m "perf(exp424): reuse OOF score calibrations in runner"
```

### Task 3: Add one-materialization batched residual prediction

**Files:**
- Modify: `experiments/exp417_causal_multihorizon_shadow/model.py`
- Test: `experiments/exp417_causal_multihorizon_shadow/tests/test_model.py`

**Interfaces:**
- Produces `predict_residuals_batch(model, channels_by_index, batch_size=32)` returning
  immutable per-index, per-horizon residuals.
- Existing `predict_residuals()` remains a one-query compatibility wrapper.

- [ ] **Step 1: Write the failing parity test**

Use a synthetic CPU `TrainedSeedModel` and two validated channel arrays. Compare every
residual from `predict_residuals_batch` with two calls to `predict_residuals` using exact
array equality; assert input order and keys are preserved.

- [ ] **Step 2: Run the focused model test and verify it fails**

Run:
`PATH=/usr/bin:/bin /Users/mac/Documents/Time\ Series/.venv/bin/python -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_model.ModelBatchPredictionTests -v`

Expected: FAIL because the batch function and test class do not exist.

- [ ] **Step 3: Implement the minimal batch path**

Validate sorted unique indices and channel shapes once, materialize the network once, run
chunks of at most 32 rows, transfer/synchronize only at the existing device boundaries,
and freeze each output residual. Keep the existing single-query path as the reference.

- [ ] **Step 4: Run CPU parity and existing model tests**

Run the focused test, then the non-training architecture/evidence tests. If MPS is available,
run a small MPS-vs-single-query smoke and require finite outputs and candidate parity.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow/model.py \
  experiments/exp417_causal_multihorizon_shadow/tests/test_model.py
git commit -m "perf(exp424): batch residual inference with one model materialization"
```

### Task 4: Wire batch inference and stage progress into the review runner

**Files:**
- Modify: `experiments/exp417_causal_multihorizon_shadow/runner.py`
- Modify: `scratch/run_exp423_mps_full_test_eval.py` (rename or replace with Exp424 runner)
- Create: `scratch/run_exp424_mps_score_cache.py`
- Test: `experiments/exp417_causal_multihorizon_shadow/tests/test_runner.py`

**Interfaces:**
- Runner groups synthetic and TEST query channels by exact length and calls the batch
  predictor once per final seed, preserving instance order.
- Exp424 status records `current_dataset`, `stage`, `completed_queries`, and
  `expected_queries` without reading labels.

- [ ] **Step 1: Write the failing progress and batch integration tests**

Assert that a synthetic score call uses the same decisions as the reference path and that
the progress payload includes the current dataset and stage fields while labels remain false.

- [ ] **Step 2: Run focused tests and verify failure**

Run the new runner tests and expect failure for the missing stage fields/batch wiring.

- [ ] **Step 3: Implement grouped batch scoring**

For OOF validation, synthetic qualification, and TEST value scoring, create deterministic
ordered channel maps, call `predict_residuals_batch`, and feed each result into the cached
calibration. Write progress after stage boundaries and after each deterministic query chunk.

- [ ] **Step 4: Run regression tests and representative benchmark**

Run runner/model/scoring tests. On one TRAIN-only representative, compare reference and
optimized residuals, scores, candidate coordinates, wall time, and MPS synchronization time.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow/runner.py \
  scratch/run_exp424_mps_score_cache.py \
  experiments/exp417_causal_multihorizon_shadow/tests/test_runner.py
git commit -m "perf(exp424): wire cached scoring and staged batch progress"
```

### Task 5: Freeze Exp424 documentation and run gate

**Files:**
- Create: `docs/EXP424_MPS_SCORE_CACHE_KO.md`
- Create: `docs/EXP424_MPS_SCORE_CACHE_RUNBOOK_KO.md`
- Modify: `scratch/run_exp424_mps_score_cache.py`

**Interfaces:**
- Manifest records optimization version, parity results, benchmark timings including
  transfer/synchronization, input hashes, code commit, and explicit review-only flags.

- [ ] **Step 1: Add a label-free preregistration/runbook test fixture**

Assert manifest fields include `contains_test_labels=false`, `test_labels_opened=false`,
`official_prediction=false`, `optimization_adopted`, and the parity/speed evidence.

- [ ] **Step 2: Run the fixture and verify the new fields fail**

Run the focused runner test and expect failure because Exp424 manifest/docs are absent.

- [ ] **Step 3: Add the documents and fail-closed gates**

Record cache formula identity, batch size, MPS/CPU parity tolerance, fallback reason, and
the rule that failed optimization gates retain the reference path. Refuse a full run unless
MPS is available, Git is clean, inputs match frozen hashes, and no other MPS-heavy process is
running.

- [ ] **Step 4: Run full relevant tests and compile checks**

Run the Exp417 model, scoring, runner, and checkpoint tests that do not launch the full
population; run `py_compile`, `git diff --check`, and the representative smoke benchmark.

- [ ] **Step 5: Commit and prepare the full run**

```bash
git add docs/EXP424_MPS_SCORE_CACHE_KO.md \
  docs/EXP424_MPS_SCORE_CACHE_RUNBOOK_KO.md scratch/run_exp424_mps_score_cache.py
git commit -m "docs(exp424): freeze score-cache MPS run gates"
```

Only after the manifest gate passes should the single Exp424 MPS process be started. Its
TEST-label evaluator remains a separate post-hoc command after prediction hash verification.
