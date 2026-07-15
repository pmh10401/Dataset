# Exp167/168 Score-Calibration Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fold-matched, label-free score-calibration parity audit (Exp167) and a frozen retrospective research-review evaluation (Exp168) for the 316 datasets with `5 <= n_train < 20`.

**Architecture:** A small pure core computes fold-paired empirical p-values and fixed source-agreement candidates. A runner reuses Exp164 fold scoring to score held-out TRAIN normal traces and TEST traces with the same fold model, freezes label-free predictions, and only then evaluates with TEST labels. Existing Exp165/166 files remain immutable.

**Tech Stack:** Python 3, NumPy, SQLite, `concurrent.futures`, `unittest`, existing Exp162/164/165/166 helpers, existing dashboard and sequential queue.

## Global Constraints

- Database path remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Data directory remains `/Users/minho/Documents/Dataset`.
- Population is exactly the Exp165/166 target set with `5 <= n_train < 20`; expected dataset count is 316.
- `n_train < 5` remains complete abstention.
- `ALPHAS = (0.005, 0.01, 0.02, 0.05)` is immutable.
- Sources are `spectral`, `shapelet`, and `interval` with no feature changes.
- Prediction may use TEST values but must not read TEST labels or actual anomaly positions.
- No TEST-length top-N budget, family branch, dataset-name policy branch, tail replacement, or past TEST performance rule.
- All source pairs, `2of3`, and `3of3` are reported; TEST metrics do not select one.
- Outputs are retrospective research-review sensitivities, not autonomous Hard alerts or prospective validation.
- Existing Exp165/166 code and results are never overwritten.

---

### Task 1: Pure Fold-Paired Parity Core

**Files:**
- Create: `experiments/exp167_168/__init__.py`
- Create: `experiments/exp167_168/parity_core.py`
- Create: `experiments/exp167_168/test_exp167_168_parity_core.py`

**Interfaces:**
- Consumes: per-source held-out arrays shaped `(n_folds,)` and fold TEST matrices shaped `(n_folds, n_test)`.
- Produces: `fold_paired_p_values(heldout_scores, test_scores) -> np.ndarray`, `candidate_rows(...) -> list[dict]`, `assert_label_free_prediction(row) -> None`, and `evaluate_indices(indices, labels) -> dict`.

- [ ] **Step 1: Write failing finite-sample and fold-order tests**

```python
def test_fold_paired_p_values_use_matching_fold_and_plus_one(self):
    heldout = np.array([1.0, 5.0, 3.0])
    test = np.array([[2.0, 2.0], [4.0, 6.0], [3.0, 2.0]])
    actual = fold_paired_p_values(heldout, test)
    np.testing.assert_allclose(actual, np.array([3 / 4, 2 / 4]))

def test_fold_order_does_not_change_p_values(self):
    order = np.array([2, 0, 1])
    np.testing.assert_allclose(
        fold_paired_p_values(heldout, test),
        fold_paired_p_values(heldout[order], test[order]),
    )
```

- [ ] **Step 2: Run the core tests and verify RED**

Run:

```bash
python3 -m unittest experiments/exp167_168/test_exp167_168_parity_core.py
```

Expected: import failure because `experiments.exp167_168.parity_core` does not exist.

- [ ] **Step 3: Implement finite-sample fold-paired p-values**

```python
def fold_paired_p_values(heldout_scores, test_scores):
    heldout = np.asarray(heldout_scores, dtype=np.float64)
    test = np.asarray(test_scores, dtype=np.float64)
    if heldout.ndim != 1 or test.ndim != 2 or test.shape[0] != heldout.size:
        raise ValueError("fold score shape mismatch")
    if heldout.size == 0 or not np.isfinite(heldout).all() or not np.isfinite(test).all():
        raise ValueError("fold scores must be finite and non-empty")
    exceedance = (heldout[:, None] >= test).sum(axis=0)
    return (1.0 + exceedance) / (heldout.size + 1.0)
```

- [ ] **Step 4: Add failing candidate-grid and label-free tests**

```python
def test_candidate_rows_emit_all_fixed_source_configs_and_alphas(self):
    rows = candidate_rows(dataset="demo", n_train=9, n_test=3, p_values=p_values, config_hash="abc")
    self.assertEqual(len(rows), 4 * 5)
    self.assertEqual({row["source_config"] for row in rows}, {
        "spectral+shapelet", "spectral+interval", "shapelet+interval", "2of3", "3of3"
    })

def test_prediction_rows_reject_evaluation_fields(self):
    row = valid_prediction_row()
    row["tp"] = 1
    with self.assertRaises(ValueError):
        assert_label_free_prediction(row)
```

- [ ] **Step 5: Implement fixed candidate rows and evaluation helper**

Use exact constants:

```python
ALPHAS = (0.005, 0.01, 0.02, 0.05)
SOURCES = ("spectral", "shapelet", "interval")
SOURCE_CONFIGS = {
    "spectral+shapelet": ("spectral", "shapelet"),
    "spectral+interval": ("spectral", "interval"),
    "shapelet+interval": ("shapelet", "interval"),
}
AGREEMENTS = ("2of3", "3of3")
```

Pair rows require both listed source flags. `2of3` requires at least two source flags and `3of3` requires all three. Every row declares `lane="research_review_candidate"`, `contains_test_labels=False`, and `contains_evaluation_metrics=False`.

- [ ] **Step 6: Run tests and commit the core**

```bash
python3 -m unittest experiments/exp167_168/test_exp167_168_parity_core.py
git add experiments/exp167_168
git commit -m "feat: add fold-paired parity core"
```

Expected: all Task 1 tests pass.

---

### Task 2: Fold-Matched Scoring and Label-Free Runner Contract

**Files:**
- Create: `experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py`
- Create: `experiments/exp167_168/test_exp167_168_structure.py`
- Modify: `experiments/exp167_168/test_exp167_168_parity_core.py`

**Interfaces:**
- Consumes: `experiments.exp164.train_normal_risk_core.fold_plan`, `experiments.exp164.crossfit_sources.score_fold`, Exp162 cache manifest, and Exp165 target filters.
- Produces: `score_dataset_folds(dataset, train_raw, test_raw) -> dict`, `predict_dataset(row, frozen_config) -> dict`, `run_audit`, `run_prediction`, and `run_evaluation`.

- [ ] **Step 1: Write failing structure tests**

Tests must assert:

```python
parameters = inspect.signature(runner.predict_dataset).parameters
self.assertNotIn("y_test", parameters)
self.assertNotIn("labels", parameters)

source = inspect.getsource(runner.predict_dataset)
self.assertNotIn("load_test_labels", source)
self.assertNotIn("y_test", source)
```

Also assert fixed DB/DATA paths, alpha tuple, expected dataset count 316, no TEST-length budget symbol, and no imports from Exp166 frozen final index maps.

- [ ] **Step 2: Run structure tests and verify RED**

```bash
python3 -m unittest experiments/exp167_168/test_exp167_168_structure.py
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement fold-matched score collection**

For each leave-one-out fold:

```python
fold_scores, target_len = score_fold(dataset, fit_raw, heldout_raw + test_raw)
heldout_scores[source][fold_idx] = fold_scores[source][1][0]
test_scores[source][fold_idx, :] = fold_scores[source][1][1:]
fit_score_summary[source].append(summarize(fold_scores[source][0]))
```

Validate that every fold contains one held-out TRAIN trace, all three sources are finite, every TEST matrix has shape `(n_train, n_test)`, and target length comes only from fold TRAIN data.

- [ ] **Step 4: Implement label-free prediction and checkpointing**

`predict_dataset` computes source p-values with `fold_paired_p_values`, then calls `candidate_rows`. It records `minimum_p_value=1/(n_train+1)`, source fold variability, config hash, and no evaluation fields.

Checkpoint rows are keyed by dataset. A cached row is reusable only when dataset, config hash, source config hash, expected row count `20`, and prediction contract all match.

- [ ] **Step 5: Implement separate evaluation**

`run_evaluation` must:

1. read `05_prediction_manifest.json`;
2. verify SHA-256 of `04_frozen_predictions.jsonl`;
3. validate every prediction row as label-free;
4. only then call `load_test_labels(dataset)`;
5. write dataset metrics, global summaries, and Exp166-local transitions.

- [ ] **Step 6: Run structure and core tests and commit**

```bash
python3 -m unittest \
  experiments/exp167_168/test_exp167_168_parity_core.py \
  experiments/exp167_168/test_exp167_168_structure.py
git add experiments/exp167_168
git commit -m "feat: add Exp167-168 parity runner"
```

Expected: all Exp167/168 tests pass.

---

### Task 3: Invariance, Coverage, and Tamper Tests

**Files:**
- Modify: `experiments/exp167_168/test_exp167_168_parity_core.py`
- Modify: `experiments/exp167_168/test_exp167_168_structure.py`

**Interfaces:**
- Consumes: Task 1 core and Task 2 runner.
- Produces: executable evidence for fold pairing, label separation, full fixed grid, deterministic hashes, and frozen prediction tamper detection.

- [ ] **Step 1: Add failing invariance tests**

Add tests for:

- source p-values unchanged by fold permutation;
- changing a synthetic label vector does not alter candidate rows;
- duplicate or out-of-range candidate indices fail;
- one dataset emits exactly `4 alphas x 5 source configs = 20` rows;
- a modified prediction file fails manifest hash verification;
- repeated identical inputs produce identical sorted JSON hash;
- minimum p-value equals `1 / (n_train + 1)`;
- alpha below minimum p-value is recorded as resolution-blocked.

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m unittest experiments/exp167_168/test_exp167_168_parity_core.py experiments/exp167_168/test_exp167_168_structure.py
```

Expected: failures for missing hash and resolution helpers.

- [ ] **Step 3: Implement minimal validation helpers**

Add `sha256_file`, deterministic JSON serialization, `verify_prediction_manifest`, and `resolution_blocked(n_train, alpha)` without adding new policy choices.

- [ ] **Step 4: Run current and regression tests**

```bash
python3 -m unittest \
  experiments/exp164/test_exp164_train_normal_risk_audit.py \
  experiments/exp165/test_exp165_small_sample_calibration.py \
  experiments/exp165/test_exp165_structure.py \
  experiments/exp166/test_exp166_core.py \
  experiments/exp166/test_exp166_structure.py \
  experiments/exp167_168/test_exp167_168_parity_core.py \
  experiments/exp167_168/test_exp167_168_structure.py
git add experiments/exp167_168
git commit -m "test: enforce Exp167-168 parity invariants"
```

Expected: all tests pass.

---

### Task 4: Smoke Run and Full Exp167/168 Execution

**Files:**
- Generate: `outputs/exp167_168_score_calibration_parity_smoke/`
- Generate: `outputs/exp167_168_score_calibration_parity/`

**Interfaces:**
- Consumes: Task 2 runner and local database.
- Produces: frozen prediction and evaluation artifacts defined by the design.

- [ ] **Step 1: Run 10-dataset label-free smoke audit and prediction**

```bash
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase audit --dataset-limit 10 \
  --output-dir outputs/exp167_168_score_calibration_parity_smoke
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase predict --dataset-limit 10 --workers 2 --reset-checkpoint \
  --output-dir outputs/exp167_168_score_calibration_parity_smoke
```

Expected: 10 audit rows, 200 prediction rows, zero errors, no evaluation fields.

- [ ] **Step 2: Inspect smoke predictions before evaluation**

Run a read-only validation that checks row count, dataset count, fixed alpha/source grid, unique in-range indices, `contains_test_labels=false`, and absence of TP/FP/F1 fields.

- [ ] **Step 3: Run smoke evaluation**

```bash
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase evaluate --dataset-limit 10 \
  --output-dir outputs/exp167_168_score_calibration_parity_smoke
```

Expected: prediction SHA verified before labels are loaded and 200 dataset metric rows written.

- [ ] **Step 4: Run full audit and prediction with seven workers**

```bash
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase audit \
  --output-dir outputs/exp167_168_score_calibration_parity
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase predict --workers 7 --reset-checkpoint \
  --output-dir outputs/exp167_168_score_calibration_parity
```

Expected: 316 datasets, 6,320 prediction rows, zero errors, deterministic prediction SHA-256.

- [ ] **Step 5: Freeze and evaluate**

```bash
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase evaluate \
  --output-dir outputs/exp167_168_score_calibration_parity
```

Do not modify the prediction JSONL between prediction and evaluation.

- [ ] **Step 6: Independently verify and commit compact artifacts**

Verify dataset count, 20 rows per dataset, prediction hash, candidate union, index ranges, no evaluation fields in predictions, and global aggregation from dataset metrics. Exclude cache and checkpoint files from Git.

```bash
git add experiments/exp167_168 outputs/exp167_168_score_calibration_parity
git commit -m "results: add Exp167-168 parity evaluation"
```

---

### Task 5: Result Analysis and Dashboard/Queue Registration

**Files:**
- Create: `docs/EXP167_168_PREREGISTRATION_KO.md`
- Create: `docs/EXP167_168_RUNBOOK_KO.md`
- Create: `outputs/exp167_168_score_calibration_parity/09_invariance_report.md`
- Create: `outputs/exp167_168_score_calibration_parity/10_result_analysis.md`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`
- Modify: `scratch/test_rank_queue_safety.py`

**Interfaces:**
- Consumes: Task 4 output summaries and existing dashboard experiment catalog.
- Produces: visible Exp167/168 history, queue status, artifact links, and a non-selective result interpretation.

- [ ] **Step 1: Write failing dashboard and queue tests**

Assert that both IDs exist:

```text
experiment_167_score_calibration_parity_audit
experiment_168_fold_matched_parity_evaluation
```

Exp167 is `completed_research`, comparison-ineligible, label-free audit. Exp168 is `completed_research`, comparison-ineligible, retrospective human-assisted review sensitivity with `selected_alpha=None` and `autonomous_metric_scope="not_evaluated"`.

- [ ] **Step 2: Run dashboard tests and verify RED**

```bash
PYTHONPATH=. python3 scratch/test_rank_dashboard_operational.py
PYTHONPATH=. python3 scratch/test_rank_queue_safety.py
```

Expected: missing Exp167/168 catalog entries.

- [ ] **Step 3: Register experiments and artifact paths**

Add queue/catalog metadata using `07_exp168_global_summary.csv` as summary, `06_exp168_dataset_metrics.csv` as detail, expected dataset count 316, and the full-run progress log. Do not mark either experiment as operationally comparable.

- [ ] **Step 4: Write Korean preregistration, runbook, invariance report, and analysis**

The analysis must separate:

- parity mismatch evidence;
- p-value resolution limits;
- TP/FP and Mean F1 retrospective results;
- review workload;
- no selected alpha/source config;
- no autonomous or prospective claim.

- [ ] **Step 5: Run final verification and commit**

```bash
python3 -m py_compile experiments/exp167_168/*.py run_rank_experiments_sequential.py serve_rank_dashboard.py
python3 -m unittest \
  experiments/exp167_168/test_exp167_168_parity_core.py \
  experiments/exp167_168/test_exp167_168_structure.py
PYTHONPATH=. python3 scratch/test_rank_dashboard_operational.py
PYTHONPATH=. python3 scratch/test_rank_queue_safety.py
git diff --check
git add docs/EXP167_168_PREREGISTRATION_KO.md docs/EXP167_168_RUNBOOK_KO.md \
  outputs/exp167_168_score_calibration_parity/09_invariance_report.md \
  outputs/exp167_168_score_calibration_parity/10_result_analysis.md \
  run_rank_experiments_sequential.py serve_rank_dashboard.py \
  scratch/test_rank_dashboard_operational.py scratch/test_rank_queue_safety.py
git commit -m "feat: expose Exp167-168 parity results"
```

Expected: all tests and `git diff --check` pass.

---

### Task 6: Branch Completion

**Files:**
- Verify all committed files from Tasks 1-5.

**Interfaces:**
- Consumes: complete implementation branch.
- Produces: a reviewed branch ready for publication without modifying the original dirty worktree.

- [ ] **Step 1: Run the complete regression set**

```bash
python3 -m unittest \
  experiments/exp164/test_exp164_train_normal_risk_audit.py \
  experiments/exp165/test_exp165_small_sample_calibration.py \
  experiments/exp165/test_exp165_structure.py \
  experiments/exp166/test_exp166_core.py \
  experiments/exp166/test_exp166_structure.py \
  experiments/exp167_168/test_exp167_168_parity_core.py \
  experiments/exp167_168/test_exp167_168_structure.py
PYTHONPATH=. python3 scratch/test_rank_dashboard_operational.py
PYTHONPATH=. python3 scratch/test_rank_queue_safety.py
git status --short --branch
git log --oneline main..HEAD
```

- [ ] **Step 2: Review scope and provenance**

Confirm no DB, virtual environment, score cache, checkpoint, smoke output, or unrelated original-worktree file is tracked. Verify prediction and MANIFEST hashes one final time.

- [ ] **Step 3: Complete with the finishing-development-branch workflow**

Present commit list, test results, experiment coverage, result limitations, and integration options. Do not merge or publish without the user's explicit instruction.
