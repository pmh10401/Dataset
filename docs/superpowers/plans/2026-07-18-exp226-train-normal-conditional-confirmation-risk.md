# Exp226 TRAIN-normal Conditional Confirmation Risk Audit Implementation Plan

> Superseded before implementation by the connected Exp226-235 pipeline design
> in `docs/superpowers/specs/2026-07-19-exp226-235-hard-alert-quality-pipeline-design.md`.
> No Exp226 production code was created from this standalone plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay the frozen Exp225 review rule on aligned TRAIN-normal pseudo-runs and report dual/single confirmation risk without reading TEST data or selecting a policy.

**Architecture:** A focused core module restores Exp215 fold-concatenated p-values to original TRAIN row indices, derives Exp224 fold-local p-values, and partitions candidates into three exclusive lanes. A runner verifies all frozen input hashes, writes label-free prediction artifacts, and produces TRAIN-only risk summaries. Smoke and full executions use the same code path and differ only by dataset limit.

**Tech Stack:** Python 3, NumPy, CSV/JSON/JSONL, `unittest`, existing Exp206/Exp219/Exp224 artifact helpers.

## Global Constraints

- Population is the 251 frozen Exp219 primary datasets.
- Shape source is `fixed_dictionary_control_conservative_max__0.005`.
- Confirmation alpha is the frozen Exp224 selected alpha `0.01`.
- TEST series, TEST labels, TEST positions, family names, and past TEST performance are forbidden inputs.
- Dataset name is a cache lookup key only.
- Exp224 `risk_target=0.005` is a reference line, not an Exp226 policy-selection rule.
- No automatic alert is generated and no review lane is promoted.
- Large artifacts are written under `/Users/minho/Documents/Dataset/exp226_train_normal_conditional_risk/`.

---

### Task 1: Aligned TRAIN-normal p-values and lane partition

**Files:**
- Create: `experiments/exp226/__init__.py`
- Create: `experiments/exp226/core.py`
- Create: `experiments/exp226/test_exp226_train_normal_risk.py`

**Interfaces:**
- Produces: `restore_fold_concatenated(values, plan, n_rows) -> np.ndarray`
- Produces: `fold_local_pvalues(scores, plan) -> np.ndarray`
- Produces: `confirmation_lanes(shape_p, temporal_p, canonical_p, shape_alpha, confirmation_alpha) -> dict[str, np.ndarray]`
- Produces: `build_train_risk_row(dataset, ...) -> dict[str, object]`

- [ ] **Step 1: Write failing alignment and p-value tests**

```python
def test_restore_fold_concatenated_returns_original_row_order():
    plan = [(np.array([1, 3]), np.array([0, 2])), (np.array([0, 2]), np.array([1, 3]))]
    actual = restore_fold_concatenated(np.array([0.1, 0.3, 0.2, 0.4]), plan, 4)
    np.testing.assert_allclose(actual, [0.1, 0.2, 0.3, 0.4])

def test_fold_local_pvalues_excludes_query_rows_from_reference():
    plan = [(np.array([0, 1]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1]))]
    actual = fold_local_pvalues(np.array([1.0, 2.0, 100.0, 200.0]), plan)
    np.testing.assert_allclose(actual, [1 / 3, 1 / 3, 1 / 3, 1 / 3])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest experiments/exp226/test_exp226_train_normal_risk.py`

Expected: import failure because `experiments.exp226.core` does not exist.

- [ ] **Step 3: Implement strict alignment validation and fold-local empirical p-values**

`restore_fold_concatenated` must reject duplicate/missing query indices and incorrect value counts. `fold_local_pvalues` must call the existing empirical upper-tail p-value helper with only each fold's fit rows.

- [ ] **Step 4: Write failing lane partition and forbidden-field tests**

```python
def test_confirmation_lanes_partition_shape_candidates():
    lanes = confirmation_lanes(
        shape_p=np.array([0.001, 0.001, 0.001, 0.5]),
        temporal_p=np.array([0.001, 0.001, 0.5, 0.001]),
        canonical_p=np.array([0.001, 0.5, 0.5, 0.001]),
        shape_alpha=0.005,
        confirmation_alpha=0.01,
    )
    assert lanes["dual"].tolist() == [0]
    assert lanes["single"].tolist() == [1]
    assert lanes["unconfirmed"].tolist() == [2]

def test_train_risk_row_contains_no_test_or_evaluation_fields():
    row = build_train_risk_row("demo", np.ones(5), np.ones(5), np.ones(5))
    forbidden = {"y_test", "tp", "fp", "f1", "precision", "test_labels"}
    assert forbidden.isdisjoint(row)
```

- [ ] **Step 5: Run tests and verify RED, implement the minimal lane and row functions, then verify GREEN**

Run: `python3 -m unittest experiments/exp226/test_exp226_train_normal_risk.py`

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add experiments/exp226
git commit -m "feat: add Exp226 train-normal lane audit core"
```

### Task 2: Frozen artifact verification and label-free runner

**Files:**
- Create: `experiments/exp226/run_exp226.py`
- Modify: `experiments/exp226/test_exp226_train_normal_risk.py`

**Interfaces:**
- Consumes: Task 1 core functions.
- Produces: `load_verified_inputs(dataset_limit=None) -> list[InputRecord]`
- Produces: `run_audit(output_root, dataset_limit=None) -> tuple[Path, Path]`
- Produces: CLI options `--output-dir`, `--dataset-limit`, and `--reset`.

- [ ] **Step 1: Write failing tests for hash, coverage, and length mismatch rejection**

Tests create temporary manifests/caches and assert that altered SHA-256, missing primary datasets, and unequal Exp215/Exp224 TRAIN lengths raise `ValueError` with the dataset or file name.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python3 -m unittest experiments.exp226.test_exp226_train_normal_risk.Exp226InputContractTests`

Expected: failure because runner verification functions are absent.

- [ ] **Step 3: Implement immutable input verification**

Reuse `load_verified_exp219_primary`, `load_verified_exp224_policy`, and `sha256_file`. Verify Exp219 prediction/manifest, Exp224 policy/replay, every consumed Exp224 artifact SHA, and every consumed Exp215 cache SHA recorded into the new manifest. Verify the source loaders both use `ORDER BY instance_index` as a documented code provenance check.

- [ ] **Step 4: Write failing deterministic output test**

Run the audit twice over synthetic temporary caches and assert sorted JSONL bytes and SHA-256 are equal. Assert every row has `contains_test_labels=false`, `contains_evaluation_metrics=false`, `autonomous_alert=false`, and lane partition equality.

- [ ] **Step 5: Implement deterministic JSONL, manifest, CSV, and summary output**

Write sorted dataset rows atomically. Include experiment ID, constants, input hashes, code hashes, row count, prediction SHA-256, retrospective scope, and known limitations. Summary reports counts, rates per 100 runs, one-sided Wilson upper bounds, dataset count distributions, and the Exp224 risk target as a reference only.

- [ ] **Step 6: Run all Exp226 tests and verify GREEN**

Run: `python3 -m unittest experiments/exp226/test_exp226_train_normal_risk.py`

Expected: all tests pass without warnings.

- [ ] **Step 7: Commit Task 2**

```bash
git add experiments/exp226
git commit -m "feat: add Exp226 frozen TRAIN-risk runner"
```

### Task 3: Smoke and full audit execution

**Files:**
- Create: `outputs/exp226_validation/SMOKE_REPORT.md`
- Create: `outputs/exp226_validation/RESULT_ANALYSIS.md`
- Copy generated summaries into: `outputs/exp226_validation/`

**Interfaces:**
- Consumes: Task 2 CLI.
- Produces: immutable full result artifacts and human-readable audit reports.

- [ ] **Step 1: Run three-dataset smoke audit**

```bash
python3 experiments/exp226/run_exp226.py \
  --dataset-limit 3 \
  --reset \
  --output-dir /Users/minho/Documents/Dataset/exp226_train_normal_conditional_risk_smoke
```

Expected: 3 datasets, no errors, no forbidden fields, exact lane partition, deterministic manifest hash.

- [ ] **Step 2: Independently validate smoke artifacts**

Use a read-only Python command to rehash the JSONL, recount lanes, verify index ranges, and scan keys for TEST/evaluation fields. Record commands and exact results in `SMOKE_REPORT.md`.

- [ ] **Step 3: Run full 251-dataset audit**

```bash
python3 experiments/exp226/run_exp226.py \
  --reset \
  --output-dir /Users/minho/Documents/Dataset/exp226_train_normal_conditional_risk
```

Expected: 251 datasets, 0 errors, all consumed hashes recorded.

- [ ] **Step 4: Independently reaggregate the full JSONL**

Recalculate total pseudo-runs, shape candidates, dual/single/unconfirmed counts, per-100 rates, one-sided Wilson upper bounds, dataset distribution quantiles, and lane partitions without importing the runner summary code. Compare every value with the generated summary.

- [ ] **Step 5: Write result interpretation without policy selection**

`RESULT_ANALYSIS.md` must state whether the frozen Exp225 review rule has low or high observed TRAIN-normal workload, while explicitly avoiding an operating threshold recommendation, autonomous alert promotion, TEST performance claim, or prospective claim.

- [ ] **Step 6: Commit Task 3**

```bash
git add outputs/exp226_validation
git commit -m "results: record Exp226 train-normal risk audit"
```

### Task 4: Regression verification and publication readiness

**Files:**
- Modify only if required by verification: `outputs/exp226_validation/RESULT_ANALYSIS.md`

**Interfaces:**
- Consumes: all Exp226 code and outputs.
- Produces: a clean, reproducible branch ready to push.

- [ ] **Step 1: Run focused and neighboring regression tests**

```bash
python3 -m unittest \
  experiments/exp215_217/test_exp215_217_core.py \
  experiments/exp215_217/test_exp215_217_structure.py \
  experiments/exp224/test_core.py \
  experiments/exp224/test_artifacts.py \
  experiments/exp224/test_policy_replay.py \
  experiments/exp225/test_exp225_fixed_dictionary_confirmation.py \
  experiments/exp226/test_exp226_train_normal_risk.py
```

Expected: all tests pass.

- [ ] **Step 2: Verify repository integrity**

Run: `git diff --check && git status --short --branch`

Expected: no whitespace errors; only intended Exp226 work is present.

- [ ] **Step 3: Verify artifact hashes once more**

Rehash the full prediction JSONL, result CSV, summary JSON/CSV, and source manifests. Confirm hashes equal the Exp226 manifest.

- [ ] **Step 4: Push the current experiment branch**

```bash
git push origin experiment/exp177-185-batch
```

Expected: remote branch advances to the verified Exp226 result commit.
