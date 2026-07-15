# Exp169/170 OOB Pooled Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic OOB score audit and target-excluded TRAIN-only pooled calibration evaluation for the 316 datasets with `5 <= n_train < 20`.

**Architecture:** Two pure modules define balanced OOB aggregation and donor-pool construction. A phase-separated runner builds immutable per-dataset OOB caches, performs label-free Exp169 normal-risk audit and Exp170 prediction, freezes the prediction SHA256, and only then evaluates TEST labels. All three pool policies, four alphas, and five source configurations remain unselected research sensitivities.

**Tech Stack:** Python 3, NumPy, SQLite, `concurrent.futures`, `unittest`, existing Exp164 source scorers, Exp167/168 target and reporting helpers, existing dashboard and sequential queue.

## Global Constraints

- Database path remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Data directory remains `/Users/minho/Documents/Dataset`.
- Target set is exactly 316 datasets with `5 <= n_train < 20`.
- Sources remain `spectral`, `shapelet`, and `interval`.
- `ALPHAS = (0.005, 0.01, 0.02, 0.05)` is immutable.
- OOB model count is 32, holdout fraction is 0.35, seed is 20260715.
- Minimum reference is 200 and donor contribution cap is 20.
- Pool policies are `similarity_pool`, `coarse_strata_pool`, and `global_balanced_pool`.
- Prediction may read TEST values but never TEST labels or anomaly positions.
- No family branch, dataset-name policy branch, TEST-length budget, tail replacement, or past TEST-performance rule.
- All 60 variants per dataset are reported; no alpha, source configuration, or pool policy is selected.
- Outputs are retrospective research-review sensitivities, not autonomous Hard alerts or prospective validation.
- Existing Exp165-168 code and outputs are immutable.

---

### Task 1: Deterministic OOB Aggregation Core

**Files:**
- Create: `experiments/exp169_170/__init__.py`
- Create: `experiments/exp169_170/oob_core.py`
- Create: `experiments/exp169_170/test_exp169_170_oob_core.py`

**Interfaces:**
- Consumes: `n_train`, component fit scores, OOB query scores, and TEST query scores.
- Produces: `balanced_oob_plan(n_train, models=32, holdout_fraction=0.35, seed=20260715) -> list[OOBSplit]`, `robust_model_scale(fit_scores) -> tuple[float, float]`, and `aggregate_oob_scores(...) -> tuple[np.ndarray, np.ndarray, int]`.

- [ ] **Step 1: Write failing split-plan tests**

```python
def test_balanced_oob_plan_is_deterministic_and_covers_every_train_row():
    first = balanced_oob_plan(7)
    second = balanced_oob_plan(7)
    self.assertEqual(first, second)
    self.assertEqual(len(first), 32)
    counts = np.zeros(7, dtype=int)
    for split in first:
        counts[list(split.holdout_indices)] += 1
        self.assertEqual(len(split.holdout_indices), 3)
        self.assertEqual(len(split.fit_indices), 4)
        self.assertFalse(set(split.fit_indices) & set(split.holdout_indices))
    self.assertGreaterEqual(counts.min(), 1)
```

- [ ] **Step 2: Run the test and verify RED**

```bash
python3 -m unittest experiments/exp169_170/test_exp169_170_oob_core.py
```

Expected: import failure because `oob_core.py` does not exist.

- [ ] **Step 3: Implement the split contract**

```python
@dataclass(frozen=True)
class OOBSplit:
    model_index: int
    fit_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]

def balanced_oob_plan(n_train, models=32, holdout_fraction=0.35, seed=20260715):
    holdout_size = max(2, int(math.ceil(holdout_fraction * n_train)))
    if n_train < 5 or holdout_size >= n_train:
        raise ValueError("OOB plan requires at least five TRAIN rows")
    order = np.random.default_rng(seed + n_train).permutation(n_train).tolist()
    splits = []
    for model_index in range(models):
        start = model_index % n_train
        held = tuple(sorted(order[(start + offset) % n_train] for offset in range(holdout_size)))
        fit = tuple(index for index in range(n_train) if index not in set(held))
        splits.append(OOBSplit(model_index, fit, held))
    return splits
```

- [ ] **Step 4: Add failing scale and equal-component aggregation tests**

Tests must prove that scale is the maximum of MAD/IQR/std/epsilon, every TRAIN row uses the first `q_d` OOB components, every TEST row uses the first `q_d` model components, and non-finite inputs fail.

```python
def test_train_and_test_aggregate_the_same_component_count():
    plan = balanced_oob_plan(5)
    train_components = np.arange(32 * 5, dtype=float).reshape(32, 5)
    test_components = np.arange(32 * 3, dtype=float).reshape(32, 3)
    train, test, q_d = aggregate_oob_scores(plan, train_components, test_components)
    self.assertEqual(q_d, min(sum(i in s.holdout_indices for s in plan) for i in range(5)))
    self.assertEqual(train.shape, (5,))
    self.assertEqual(test.shape, (3,))
```

- [ ] **Step 5: Implement normalization and aggregation**

`robust_model_scale` computes median, scaled MAD, IQR scale, population standard deviation, and the specified epsilon. `aggregate_oob_scores` validates shapes and finiteness, computes `q_d`, uses model-index order, and returns median aggregates.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest experiments/exp169_170/test_exp169_170_oob_core.py
git add experiments/exp169_170
git commit -m "feat: add deterministic OOB score core"
```

---

### Task 2: Target-excluded Donor Pool Core

**Files:**
- Create: `experiments/exp169_170/pool_core.py`
- Create: `experiments/exp169_170/test_exp169_170_pool_core.py`

**Interfaces:**
- Consumes: TRAIN-only per-dataset descriptors and OOB aggregate source scores.
- Produces: `build_descriptor(...) -> np.ndarray`, `build_pool_manifest(...) -> list[dict]`, `pooled_p_values(query, reference) -> np.ndarray`, and `fixed_candidate_rows(...) -> list[dict]`.

- [ ] **Step 1: Write failing target-exclusion and minimum-reference tests**

```python
def test_every_pool_excludes_target_and_meets_reference_contract():
    pools = build_pool_manifest(dataset="A", packs=synthetic_packs(), minimum_reference=4, donor_cap=2)
    for policy, pool in pools.items():
        self.assertNotIn("A", pool.donors)
        for source in SOURCES:
            self.assertGreaterEqual(pool.references[source].size, 4)
            self.assertLessEqual(max(pool.per_donor_counts[source].values()), 2)
```

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m unittest experiments/exp169_170/test_exp169_170_pool_core.py
```

- [ ] **Step 3: Implement descriptors and the three frozen policies**

Use a 23-value descriptor: log TRAIN count, log TRAIN target length, and seven summary values for each of three sources. Robust-scale descriptor columns across all 316 targets. Similarity selection includes distance ties. Coarse strata uses the fixed count bins and TRAIN-length quartiles. Global-balanced uses every non-target donor. Apply the donor cap before concatenation and require 200 scores for each source.

- [ ] **Step 4: Add failing p-value and candidate-grid tests**

```python
def test_pooled_p_values_use_plus_one_finite_sample_correction():
    reference = np.array([1.0, 2.0, 3.0])
    query = np.array([0.0, 2.5, 4.0])
    np.testing.assert_allclose(pooled_p_values(query, reference), [1.0, 0.5, 0.25])

def test_fixed_grid_has_three_pools_four_alphas_five_configs():
    rows = fixed_candidate_rows("demo", n_test=3, p_values=synthetic_p_values(), config_hash="x")
    self.assertEqual(len(rows), 60)
```

- [ ] **Step 5: Implement p-values and candidate grid**

The output row records pool policy, alpha, source configuration, reference counts, minimum p-values, candidate indices, `lane="research_review_candidate"`, and all usage flags set to zero. Pair, 2-of-3, and 3-of-3 behavior must match Exp167/168.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest \
  experiments/exp169_170/test_exp169_170_oob_core.py \
  experiments/exp169_170/test_exp169_170_pool_core.py
git add experiments/exp169_170
git commit -m "feat: add TRAIN-only pooled calibration core"
```

---

### Task 3: OOB Cache Builder and Exp169 Audit

**Files:**
- Create: `experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py`
- Create: `experiments/exp169_170/test_exp169_170_structure.py`

**Interfaces:**
- Consumes: raw TRAIN/TEST values, `experiments.exp164.crossfit_sources.score_fold`, Task 1 OOB core, and Task 2 pool core.
- Produces: `build_dataset_cache(dataset, train_raw, test_raw) -> dict`, `run_cache(...)`, and `run_audit(...)`.

- [ ] **Step 1: Write failing structure tests**

Assert exact DB path, dataset count 316, fixed constants, phase choices, absence of family and TEST-budget symbols, and that prediction functions have no `y_test` or `labels` argument.

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m unittest experiments/exp169_170/test_exp169_170_structure.py
```

- [ ] **Step 3: Implement cache generation**

For every OOB split, call the existing source scorer on fit TRAIN, held-out TRAIN, and TEST in one scoring invocation. Normalize held-out and TEST scores with the fit score distribution from that same model. Store source-level TRAIN and TEST component matrices, aggregate arrays, split metadata, target length derived from TRAIN only, config hash, and no labels.

The cache manifest must contain 316 successful rows. Any dataset error writes an error CSV and stops the phase with `SystemExit(2)`.

- [ ] **Step 4: Implement donor pack and Exp169 audit**

Load all 316 TRAIN OOB aggregate arrays, build descriptors and all three target-excluded pools, then score each target's TRAIN-normal aggregates against its pool. Write:

- `01_contract.json`
- `02_oob_cache_manifest.csv`
- `03_pool_manifest.csv`
- `04_exp169_dataset_audit.csv`
- `05_exp169_global_summary.csv`

No file may include TEST labels or evaluation metrics.

- [ ] **Step 5: Add tests for cache and audit contracts**

Tests cover cache hash validation, target exclusion, 316-row coverage validation, no TEST-label query in cache/audit functions, and failure on partial pool coverage.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest experiments/exp169_170/test_exp169_170_*.py
git add experiments/exp169_170
git commit -m "feat: add Exp169 OOB calibration audit"
```

---

### Task 4: Frozen Exp170 Prediction and Separate Evaluation

**Files:**
- Modify: `experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py`
- Modify: `experiments/exp169_170/test_exp169_170_structure.py`
- Create: `experiments/exp169_170/independent_reaggregate_exp169_170.py`

**Interfaces:**
- Consumes: validated OOB caches and pool manifests from Task 3.
- Produces: `run_prediction(...)`, `verify_prediction_manifest(...)`, `run_evaluation(...)`, and independent aggregate verification.

- [ ] **Step 1: Write failing prediction/evaluation separation tests**

Tests prove that prediction rows reject TP/FP/F1 and TEST-label fields, row-grid validation requires exactly 18,960 rows, tampered prediction hash fails, and label loading occurs only after manifest verification.

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m unittest experiments/exp169_170/test_exp169_170_structure.py
```

- [ ] **Step 3: Implement label-free prediction**

For every dataset, compute source p-values against each frozen pool and emit 60 rows. Sort by dataset, pool policy, alpha, and source configuration. Write:

- `06_frozen_predictions.jsonl`
- `07_prediction_manifest.json`

The manifest records 316 datasets, 18,960 rows, config hash, prediction SHA256, cache-manifest hash, pool-manifest hash, workers, and research-only scopes.

- [ ] **Step 4: Implement evaluation and comparisons**

After SHA verification, load TEST labels and write:

- `08_exp170_dataset_metrics.csv`
- `09_exp170_global_summary.csv`
- `10_candidate_transition.csv`
- `11_completion.json`

Compare against Exp166 local/pool and Exp168 without selecting a policy. Keep autonomous metrics unset.

- [ ] **Step 5: Implement independent reaggregation**

The read-only script recomputes all 60 summary rows from dataset metrics and verifies prediction counts, index ranges, candidate counts, TP+FP equality, source-grid coverage, and global summaries. It writes `12_independent_reaggregation.json` and exits nonzero on any mismatch.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest experiments/exp169_170/test_exp169_170_*.py
git add experiments/exp169_170
git commit -m "feat: add frozen Exp170 review evaluation"
```

---

### Task 5: Smoke and Full Sequential Execution

**Files:**
- Generate: `outputs/exp169_170_oob_pooled_calibration_smoke/`
- Generate: `outputs/exp169_170_oob_pooled_calibration/`

**Interfaces:**
- Consumes: Task 4 runner.
- Produces: complete frozen audit, prediction, evaluation, and independent verification artifacts.

- [ ] **Step 1: Run 10-dataset smoke cache and audit**

```bash
SMOKE=outputs/exp169_170_oob_pooled_calibration_smoke
python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py \
  --phase cache --dataset-limit 10 --workers 2 --reset-checkpoint --output-dir "$SMOKE"
python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py \
  --phase audit --dataset-limit 10 --minimum-reference 20 --output-dir "$SMOKE"
```

The smoke-only minimum reference override is recorded in a distinct config hash and cannot be reused by full prediction.

- [ ] **Step 2: Run smoke prediction and evaluation**

```bash
python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py \
  --phase predict --dataset-limit 10 --minimum-reference 20 --workers 2 --output-dir "$SMOKE"
python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py \
  --phase evaluate --dataset-limit 10 --output-dir "$SMOKE"
```

Verify 600 prediction rows, no evaluation fields, no index errors, and SHA verification before labels.

- [ ] **Step 3: Run full cache with seven workers**

```bash
OUT=outputs/exp169_170_oob_pooled_calibration
python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py \
  --phase cache --workers 7 --reset-checkpoint --output-dir "$OUT"
```

Expected: 316 caches, zero errors, complete OOB coverage.

- [ ] **Step 4: Run full audit, prediction, and evaluation sequentially**

```bash
python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py --phase audit --output-dir "$OUT"
python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py --phase predict --workers 7 --output-dir "$OUT"
python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py --phase evaluate --output-dir "$OUT"
python3 experiments/exp169_170/independent_reaggregate_exp169_170.py --output-dir "$OUT"
```

- [ ] **Step 5: Commit compact artifacts**

Track contracts, manifests, frozen prediction, metrics, summaries, transitions, and independent verification. Exclude `cache/`, checkpoints, smoke output, and temporary logs.

```bash
git add -f outputs/exp169_170_oob_pooled_calibration/*.csv
git add outputs/exp169_170_oob_pooled_calibration/*.json outputs/exp169_170_oob_pooled_calibration/*.jsonl
git commit -m "results: add Exp169-170 OOB pooled evaluation"
```

---

### Task 6: Analysis, Dashboard Registration, and Final Verification

**Files:**
- Create: `docs/EXP169_170_PREREGISTRATION_KO.md`
- Create: `docs/EXP169_170_RUNBOOK_KO.md`
- Create: `outputs/exp169_170_oob_pooled_calibration/13_invariance_report.md`
- Create: `outputs/exp169_170_oob_pooled_calibration/14_result_analysis.md`
- Create: `outputs/exp169_170_oob_pooled_calibration/MANIFEST.csv`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`
- Modify: `scratch/test_rank_queue_safety.py`

**Interfaces:**
- Consumes: Task 5 outputs.
- Produces: research-history dashboard entries, artifact links, Korean operating notes, and final provenance.

- [ ] **Step 1: Write failing dashboard and queue tests**

Assert Exp169 and Exp170 are `completed_research`, comparison-ineligible, expected datasets 316, `selected_alpha=None`, `selected_pool_policy=None`, and `autonomous_metric_scope="not_evaluated"` for Exp170.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=. python3 scratch/test_rank_dashboard_operational.py
PYTHONPATH=. python3 scratch/test_rank_queue_safety.py
```

- [ ] **Step 3: Register both experiments and review variants**

Use `04_exp169_dataset_audit.csv` and `05_exp169_global_summary.csv` for Exp169, and `08_exp170_dataset_metrics.csv` and `09_exp170_global_summary.csv` for Exp170. Dashboard variant rows must show pool policy, alpha, source configuration, review candidates, review TP/FP, precision, and review mean F1 without labeling them Hard metrics.

- [ ] **Step 4: Write reports and manifest**

The analysis separates normal-risk audit, TEST review sensitivity, candidate resolution, workload, and retrospective limitations. It explicitly states that no variant was selected. MANIFEST records hashes, execution commands, code commit, DB hash, config hash, and scopes.

- [ ] **Step 5: Run final regression and provenance checks**

```bash
python3 -m py_compile experiments/exp169_170/*.py run_rank_experiments_sequential.py serve_rank_dashboard.py
python3 -m unittest \
  experiments/exp164/test_exp164_train_normal_risk_audit.py \
  experiments/exp165/test_exp165_small_sample_calibration.py \
  experiments/exp166/test_exp166_core.py \
  experiments/exp167_168/test_exp167_168_parity_core.py \
  experiments/exp167_168/test_exp167_168_structure.py \
  experiments/exp169_170/test_exp169_170_oob_core.py \
  experiments/exp169_170/test_exp169_170_pool_core.py \
  experiments/exp169_170/test_exp169_170_structure.py
PYTHONPATH=. python3 scratch/test_rank_dashboard_operational.py
PYTHONPATH=. python3 scratch/test_rank_queue_safety.py
git diff --check
```

Verify prediction and MANIFEST hashes, clean branch status, and absence of tracked cache/checkpoint/smoke files.

- [ ] **Step 6: Commit final integration**

```bash
git add docs/EXP169_170_PREREGISTRATION_KO.md docs/EXP169_170_RUNBOOK_KO.md \
  outputs/exp169_170_oob_pooled_calibration/13_invariance_report.md \
  outputs/exp169_170_oob_pooled_calibration/14_result_analysis.md \
  run_rank_experiments_sequential.py serve_rank_dashboard.py \
  scratch/test_rank_dashboard_operational.py scratch/test_rank_queue_safety.py
git add -f outputs/exp169_170_oob_pooled_calibration/MANIFEST.csv
git commit -m "feat: expose Exp169-170 OOB pooled results"
```

---

### Task 7: Branch Completion

**Files:**
- Verify all files created or modified in Tasks 1-6.

**Interfaces:**
- Consumes: complete implementation branch.
- Produces: a verified stacked branch ready for PR publication after Exp167/168.

- [ ] **Step 1: Re-run final checks from Task 6**

Expected: zero failures and a clean worktree.

- [ ] **Step 2: Review branch scope**

```bash
git status --short --branch
git log --oneline experiment/exp167-168-score-calibration-parity..HEAD
git diff --stat experiment/exp167-168-score-calibration-parity...HEAD
```

Confirm only Exp169/170 code, results, docs, tests, and dashboard integration are present.

- [ ] **Step 3: Complete with finishing-development-branch workflow**

Keep the branch stacked on `experiment/exp167-168-score-calibration-parity` until PR #4 is merged. Do not merge or push without explicit user instruction.
