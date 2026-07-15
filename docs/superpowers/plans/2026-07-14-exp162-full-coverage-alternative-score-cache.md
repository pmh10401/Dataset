# Exp162 Full-Coverage Alternative Score Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic TRAIN-normal-only score caches for three alternative source types over the frozen 1,117-dataset population without selecting candidates or reading TEST labels.

**Architecture:** A values-only DB bridge supplies aligned TRAIN/TEST matrices. Three pure feature extractors feed one shared TRAIN-fitted robust scorer. A checkpointed runner writes one atomic NPZ per dataset and a hash-rich manifest; evaluation is deliberately absent.

**Tech Stack:** Python 3, NumPy, scikit-learn `RobustScaler`, SQLite, `concurrent.futures`, `unittest`.

## Global Constraints

- Database remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Frozen population is the 1,117 unique dataset names in Exp160 prediction SHA256 `ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d`.
- Prediction/cache code must not read TEST labels or anomaly positions.
- No family/dataset-name policy branch, TEST-length budget, threshold, candidate, lane, TP, FP, FN, or F1 is allowed.
- Workers default to 7; every worker limits numeric-library inner threads to 1.
- Results are retrospective infrastructure artifacts, not prospective validation.

---

### Task 1: Pure Source And Score Contracts

**Files:**
- Create: `experiments/exp162/__init__.py`
- Create: `experiments/exp162/full_coverage_score_core.py`
- Create: `experiments/exp162/test_exp162_full_coverage_score_cache.py`

**Interfaces:**
- Produces: `spectral_features(X)`, `shapelet_features(dataset_name, X_train, X_test, target_len)`, `interval_features(X)`, `robust_score_pair(train_features, test_features)`, and `source_config_hash()`.

- [ ] Write failing tests for feature dimensions, deterministic shapelets, finite score vectors, TRAIN-only scaler behavior, and fixed configuration hash.
- [ ] Run `python3 -m unittest experiments/exp162/test_exp162_full_coverage_score_cache.py` and confirm missing imports/functions fail.
- [ ] Implement the three frozen source configurations exactly as the design document specifies.
- [ ] Re-run the unit test and confirm all core tests pass.

### Task 2: Values-Only Bridge And Atomic Cache

**Files:**
- Create: `experiments/exp162/exp162_bridge.py`
- Create: `experiments/exp162/cache_io.py`
- Modify: `experiments/exp162/test_exp162_full_coverage_score_cache.py`

**Interfaces:**
- Consumes: pure source functions from Task 1.
- Produces: `load_dataset_series_only(name)`, `build_dataset_cache(name)`, `write_cache_atomic(path, payload, metadata)`, and `validate_cache(path, expected_config_hash)`.

- [ ] Write failing structural tests proving the DB query selects `values_blob` and never selects `label`.
- [ ] Write failing cache tests for required arrays, forbidden metadata keys, hash validation, and tamper rejection.
- [ ] Reuse the Exp151/152 TRAIN-derived alignment contract without importing its label loader.
- [ ] Implement atomic NPZ writes and validation, then run all Exp162 tests.

### Task 3: Checkpointed Runner And Audit Artifacts

**Files:**
- Create: `experiments/exp162/run_experiment_162_full_coverage_alternative_score_cache.py`
- Create: `experiments/exp162/test_exp162_structure.py`
- Create: `docs/EXP162_PREREGISTRATION_KO.md`
- Create: `docs/EXP162_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: `build_dataset_cache` and `validate_cache` from Task 2.
- Produces: per-dataset NPZ files, `01_source_contract.json`, `02_cache_manifest.csv`, `03_errors.csv`, `04_coverage_summary.csv`, `CHECKPOINT.jsonl`, `MANIFEST.csv`, and `MANIFEST.md`.

- [ ] Write failing CLI/structure tests for `--dataset-limit`, `--workers`, `--reset-checkpoint`, `--output-dir`, direct execution, and exact 1,117-name population validation.
- [ ] Implement seven-worker dataset scheduling, atomic checkpoint updates, resume validation, visible errors, and nonzero exit when full coverage is incomplete.
- [ ] Document exact smoke/full commands and interpretation limits.
- [ ] Run `py_compile` and both Exp162 test modules.

### Task 4: Smoke Verification

**Files:**
- Generate: `outputs/exp162_full_coverage_alternative_score_cache_smoke/`

**Interfaces:**
- Consumes: frozen source code and first 10 sorted datasets.
- Produces: 10 validated NPZ caches and complete audit artifacts.

- [ ] Run the 10-dataset smoke with 2 workers and a reset checkpoint.
- [ ] Independently inspect all NPZ files for six finite arrays, matching row counts, no forbidden metadata, unique datasets, and matching config hashes.
- [ ] Run the same smoke without reset and verify checkpoint reuse does not change cache hashes.
- [ ] Stop before full execution if any smoke error, coverage gap, or hash drift remains.

### Task 5: Full 1,117-Dataset Cache

**Files:**
- Generate: `outputs/exp162_full_coverage_alternative_score_cache/`
- Create: `outputs/exp162_full_coverage_alternative_score_cache/VERIFICATION_REPORT.md`

**Interfaces:**
- Consumes: smoke-verified Exp162 implementation.
- Produces: a frozen full-coverage cache suitable as input to a separately preregistered Exp163.

- [ ] Run the full experiment with 7 workers and checkpoint reset only on the first attempt.
- [ ] Verify 1,117 unique cache rows, 1,117 valid NPZ files, zero errors, and complete coverage for all three sources.
- [ ] Independently reaggregate array finiteness, dimensions, row counts, source/config hashes, and output hashes without reading TEST labels.
- [ ] Write the verification report and regenerate the artifact manifest.
- [ ] Do not evaluate anomaly performance or start Exp163 in this task.
