# Exp255 Change-point Score Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and queue an independently evaluated TRAIN-normal change-point score source using robust CUSUM/MOSUM and PELT L1/L2/RBF features.

**Architecture:** A new `experiments.exp255_change_point_features` package mirrors the proven Exp254 cache, prediction, separated evaluation, reporting, and dashboard contracts. A dedicated `features.py` fits point-wise robust normalization on TRAIN only and emits four fixed-width blocks; existing TRAIN OOF KNN local-gap scoring and empirical calibration are reused through experiment-local wrappers.

**Tech Stack:** Python 3.14, NumPy, SciPy, scikit-learn, ruptures 1.1.9, SQLite-backed project loaders, unittest, seven CPU workers.

## Global Constraints

- Preserve Exp254 processes, files, and outputs.
- Use only normal TRAIN values for feature fitting, score calibration, and thresholds.
- Do not pass TEST labels, actual anomaly positions, family policy, dataset-name policy, or TEST-length budget into prediction.
- Use `ALPHAS = (0.005, 0.01, 0.02, 0.05)` and report every value without selecting one from TEST performance.
- Use working length 32 through 256, MOSUM fractions `(1/32, 1/16, 1/8, 1/4)`, PELT `min_size=max(4,L//32)`, `jump=max(1,L//64)`, and `penalty=3*log(L)`.
- Use seed `20260724`, five OOF folds, KNN 3 local-gap, and seven CPU workers.
- Treat all results as retrospective score-source qualification; do not modify autonomous Hard policy.
- Use MLX only after exact parity and at least 1.1x speedup; Exp255 remains CPU because no such benchmark exists.

---

### Task 1: Dependency, configuration, and change-point feature blocks

**Files:**
- Create: `experiments/exp255_change_point_features/__init__.py`
- Create: `experiments/exp255_change_point_features/config.py`
- Create: `experiments/exp255_change_point_features/features.py`
- Create: `experiments/exp255_change_point_features/tests/__init__.py`
- Create: `experiments/exp255_change_point_features/tests/test_features.py`

**Interfaces:**
- Produces: `RobustReference.fit(train_rows)`, `RobustReference.transform(rows)`, `ChangePointFeatureExtractor.fit_transform_train(rows)`, `ChangePointFeatureExtractor.transform(rows)`, `FEATURE_DIMENSIONS`.
- Consumes: aligned two-dimensional float arrays and the fixed Exp255 configuration.

- [ ] **Step 1: Install the pinned detector dependency**

Run:

```bash
/Users/minho/Documents/Dataset/venvs/tsai-sktime-py314/bin/python -m pip install 'ruptures==1.1.9'
```

Expected: `ruptures-1.1.9` is installed without changing the DB or DATA_DIR.

- [ ] **Step 2: Write failing feature tests**

The tests must assert:

```python
def test_reference_uses_train_only():
    reference = RobustReference.fit(TRAIN)
    np.testing.assert_allclose(reference.transform(TEST), reference.transform(TEST.copy()))

def test_step_change_exceeds_constant_change_strength():
    extractor = ChangePointFeatureExtractor(working_length=64)
    train = np.vstack([np.zeros(64), np.full(64, 0.01), np.full(64, -0.01),
                       np.linspace(-0.01, 0.01, 64), np.linspace(0.01, -0.01, 64)])
    extractor.fit_transform_train(train)
    blocks = extractor.transform(np.vstack([np.zeros(64), np.r_[np.zeros(32), np.ones(32)]]))
    assert blocks["cusum_mosum_18"][1].max() > blocks["cusum_mosum_18"][0].max()

def test_feature_dimensions_are_fixed_and_finite():
    assert FEATURE_DIMENSIONS == {
        "cusum_mosum_18": 18,
        "pelt_l1_l2_10": 10,
        "pelt_rbf_5": 5,
        "change_point_combined_33": 33,
    }
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
/Users/minho/Documents/Dataset/venvs/tsai-sktime-py314/bin/python -m unittest experiments.exp255_change_point_features.tests.test_features -v
```

Expected: FAIL because the package and interfaces do not exist.

- [ ] **Step 4: Implement minimal fixed feature extraction**

Implement:

```python
FEATURE_DIMENSIONS = {
    "cusum_mosum_18": 18,
    "pelt_l1_l2_10": 10,
    "pelt_rbf_5": 5,
    "change_point_combined_33": 33,
}
```

`RobustReference.fit()` must calculate point-wise TRAIN median and MAD scale with a finite positive fallback. `_cusum_mosum()` must emit two global CUSUM strengths and four statistics at each of four window fractions. `_pelt_summary()` must remove the terminal breakpoint `L`, calculate five segment summaries, and return zeros when there is no internal breakpoint. Use `rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(row).predict(pen=3.0*np.log(L))`.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Task 1 test command. Expected: all feature tests PASS with no RuntimeWarning.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp255_change_point_features
git commit -m "feat: add Exp255 change-point feature blocks"
```

### Task 2: TRAIN-only cache and KNN score generation

**Files:**
- Create: `experiments/exp255_change_point_features/scoring.py`
- Create: `experiments/exp255_change_point_features/cache_pipeline.py`
- Create: `experiments/exp255_change_point_features/tests/test_scoring.py`

**Interfaces:**
- Consumes: `FEATURE_DIMENSIONS`, `ChangePointFeatureExtractor`, `load_bundle(dataset)`.
- Produces: `build_cache_unit(root_text, dataset, reset=False)`, `cache_all(...)`, NPZ arrays named `<variant>__train_oof` and `<variant>__test_scores`.

- [ ] **Step 1: Write failing cache and score tests**

Tests must verify deterministic finite scores, empty arrays for `n_train < 5`, config flags `contains_test_labels=False`, `uses_family_name=False`, `uses_test_length_budget=False`, and cache hash rejection on tampering.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/Users/minho/Documents/Dataset/venvs/tsai-sktime-py314/bin/python -m unittest experiments.exp255_change_point_features.tests.test_scoring -v
```

Expected: FAIL because scoring and cache modules do not exist.

- [ ] **Step 3: Implement scoring and atomic cache**

Reuse the project contracts `fit_train_sanitizer`, five-fold OOF KNN local-gap scoring, `stable_json_hash`, `sha256_file`, and atomic NPZ/JSON replacement. `source_config()` must record ruptures version, all detector constants, seed, dimensions, and forbidden-input flags.

- [ ] **Step 4: Run Task 1 and Task 2 tests**

Expected: all tests PASS; all generated score arrays contain finite floats.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp255_change_point_features
git commit -m "feat: add Exp255 TRAIN-only score cache"
```

### Task 3: Frozen prediction and separated evaluation

**Files:**
- Create: `experiments/exp255_change_point_features/prediction.py`
- Create: `experiments/exp255_change_point_features/evaluation.py`
- Create: `experiments/exp255_change_point_features/independent_reaggregate.py`
- Create: `experiments/exp255_change_point_features/reporting.py`
- Create: `experiments/exp255_change_point_features/tests/test_evaluation.py`

**Interfaces:**
- Consumes: cache NPZ/metadata, Exp236/Exp253 frozen lane maps, `empirical_candidates`.
- Produces: `frozen_predictions.jsonl`, prediction SHA256, `results.csv`, `summary.csv`, independent consistency report, Markdown report, manifest.

- [ ] **Step 1: Write failing separation tests**

Tests must assert prediction rows omit `y_test`, TP, FP, F1, and combined F1; lane views are intersections only; tampered prediction hashes fail evaluation; and independent reaggregation exactly matches summary totals.

- [ ] **Step 2: Run tests and verify RED**

Run the new evaluation test module. Expected: FAIL because prediction/evaluation modules do not exist.

- [ ] **Step 3: Implement prediction freeze**

For every dataset, four variants, and four alphas, write sorted JSONL rows containing candidate indices and intersections with frozen Exp236/Exp253 lanes. Set `hard_policy_changed=False`, `comparison_eligible=False`, `contains_test_labels=False`, and `contains_evaluation_metrics=False`.

- [ ] **Step 4: Implement evaluation after hash verification**

Only after verifying prediction SHA256, call `load_test_labels_for_evaluation(dataset)`. Report candidate, frozen Hard intersection, Standard intersection, Priority intersection, outside-lane, Exp253 Hard intersection, and Exp253 remaining Standard intersection independently.

- [ ] **Step 5: Run all Exp255 tests**

Expected: all tests PASS and hash tamper test rejects modified JSONL.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp255_change_point_features
git commit -m "feat: add Exp255 separated evaluation"
```

### Task 4: Pipeline, dashboard status, and sequential runner

**Files:**
- Create: `experiments/exp255_change_point_features/run_pipeline.py`
- Create: `experiments/exp255_change_point_features/tests/test_pipeline.py`
- Create: `scratch/run_exp255_change_point_features.py`
- Modify: `serve_rank_dashboard.py`

**Interfaces:**
- Produces: CLI stages `audit/cache/predict/evaluate/all`, atomic status heartbeat, sequential state registration, dashboard history/result discovery.
- Consumes: experiment modules from Tasks 1-3 and `/Users/minho/Documents/Dataset/rank_experiments_sequential_state.json`.

- [ ] **Step 1: Write failing pipeline/dashboard tests**

Tests must verify CLI stages, a zero-progress initial status, label-free cache/predict status, completed evaluation status, queue registration, and dashboard discovery of Exp255 output metadata.

- [ ] **Step 2: Run tests and verify RED**

Expected: FAIL because the runner and dashboard registration do not exist.

- [ ] **Step 3: Implement pipeline and atomic status**

Use Exp255 labels and output root `/Users/minho/Documents/Dataset/exp255_change_point_features`. Status must contain `done`, `expected`, `current_dataset`, `errors`, `progress_percent`, `updated_at`, and `stdout_log`.

- [ ] **Step 4: Implement sequential runner**

The runner must add Exp255 to the queue, wait while any experiment is active, then execute `run_pipeline --stage all --workers 7 --reset-cache`. On failure it must return Exp255 to the queue and preserve the log.

- [ ] **Step 5: Run all Exp255 tests and dashboard tests**

Expected: all PASS without modifying the active Exp254 state.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp255_change_point_features scratch/run_exp255_change_point_features.py serve_rank_dashboard.py
git commit -m "feat: add Exp255 pipeline and dashboard queue"
```

### Task 5: Smoke validation and queue registration

**Files:**
- Create during execution: `/Users/minho/Documents/Dataset/exp255_change_point_features_smoke/`
- Modify through atomic state API: `/Users/minho/Documents/Dataset/rank_experiments_sequential_state.json`

**Interfaces:**
- Consumes: completed Exp255 package and pinned environment.
- Produces: a ten-dataset smoke result and one queued sequential full experiment.

- [ ] **Step 1: Run compile and complete unit suite**

```bash
/Users/minho/Documents/Dataset/venvs/tsai-sktime-py314/bin/python -m py_compile experiments/exp255_change_point_features/*.py scratch/run_exp255_change_point_features.py
/Users/minho/Documents/Dataset/venvs/tsai-sktime-py314/bin/python -m unittest discover -s experiments/exp255_change_point_features/tests -v
```

Expected: no compile errors and all tests PASS.

- [ ] **Step 2: Run ten-dataset smoke**

```bash
/Users/minho/Documents/Dataset/venvs/tsai-sktime-py314/bin/python -m experiments.exp255_change_point_features.run_pipeline --stage all --workers 2 --dataset-limit 10 --reset-cache --output-root /Users/minho/Documents/Dataset/exp255_change_point_features_smoke
```

Expected: coverage 10/10, cache errors 0, prediction rows `10 * 4 * 4 = 160`, prediction fields contain no labels or metrics, and independent reaggregation passes.

- [ ] **Step 3: Verify current Exp254 remains active or completed normally**

Inspect process and sequential state. Do not kill, restart, or modify Exp254.

- [ ] **Step 4: Queue Exp255 without starting a competing run**

Start the sequential runner detached. It must display Exp255 as pending while Exp254 is active and begin automatically only after `running` becomes null.

- [ ] **Step 5: Verify queue and dashboard**

Confirm Exp255 appears in Experiment Queue, Exp254 remains the sole active experiment, and Exp255 has not allocated worker processes early.

- [ ] **Step 6: Final verification commit if smoke exposes documentation-only corrections**

Run `git diff --check` and commit only scoped Exp255 corrections. Do not commit generated DB/output artifacts.
