# Exp206-210 Residual Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and sequentially run a five-stage, label-isolated experiment batch that reconstructs alternative-source OOF scores and tests continuous residual-coverage rules over the 251 datasets with at least 100 TRAIN-normal instances.

**Architecture:** A focused `experiments/exp206_210` package separates pure score mathematics, database/artifact loading, cross-fitted feature scoring, label-free prediction, delayed evaluation, and independent reaggregation. Exp206 must complete a hash-verified OOF cache before Exp207 can freeze risk-eligible rules; Exp208/209 consume only those frozen rules, and Exp210 independently reconciles all results. The existing sequential queue and dashboard expose each stage independently.

**Tech Stack:** Python 3, NumPy, scikit-learn, SQLite, `unittest`, JSONL/CSV/NPZ artifacts, existing queue/dashboard runtime.

## Global Constraints

- Exp137 autonomous Hard, Standard-review, and Priority-review indices remain unchanged.
- Candidate index means TEST instance, not time point; no neighboring-index tolerance or event clustering.
- Prediction never receives or reads TEST labels or anomaly positions.
- Evaluation verifies prediction SHA256 before loading TEST labels.
- Dataset/family names are lookup keys only and never policy inputs.
- TEST length cannot determine threshold, risk, budget, source, or routing.
- Fixed alpha grid is `(0.005, 0.01, 0.02, 0.05)` and every variant is reported.
- Shadow risk contract is one-sided Wilson upper bound `<= 0.001`; review risk contract is `<= 0.005`.
- No TEST metric selects a source, fusion method, alpha, margin, or risk target.
- Results remain retrospective diagnostics; Exp204 remains a separate prospective-shadow candidate.
- One stage runs at a time and dataset work uses seven CPU workers where valid.

---

## File Structure

- Create `experiments/exp206_210/__init__.py`: package marker.
- Create `experiments/exp206_210/core.py`: pure folds, p-values, fusions, Wilson risk, lane classification, and payload validation.
- Create `experiments/exp206_210/artifacts.py`: frozen input loaders, hashes, DB value-only loading, delayed label loading, and cache IO.
- Create `experiments/exp206_210/alternative_oof.py`: fold-safe derivative spectral, shape-prototype, and interval-distribution scoring.
- Create `experiments/exp206_210/run_experiment_206_210.py`: stage dispatcher, progress/checkpoints, prediction/evaluation, and summaries.
- Create `experiments/exp206_210/independent_reaggregate.py`: Exp210 arithmetic and overlap verification.
- Create `experiments/exp206_210/test_exp206_210_core.py`: pure behavior tests.
- Create `experiments/exp206_210/test_exp206_210_structure.py`: leakage, source, queue, and direct-execution tests.
- Create `docs/EXP206_210_PREREGISTRATION_KO.md`: Korean scientific contract.
- Create `docs/EXP206_210_RUNBOOK_KO.md`: smoke, queue, resume, and audit commands.
- Modify `run_rank_experiments_sequential.py`: stage metadata, batch list, queue command.
- Modify `serve_rank_dashboard.py`: static discovery fallback for Exp206-210 logs/results.

### Task 1: Pure residual-fusion mathematics

**Files:**
- Create: `experiments/exp206_210/__init__.py`
- Create: `experiments/exp206_210/core.py`
- Create: `experiments/exp206_210/test_exp206_210_core.py`

**Interfaces:**
- Produces: `fold_plan(n, folds, seed)`, `empirical_upper_pvalues(reference, query)`, `leave_one_out_pvalues(scores)`, `robust_z_pair(train, query)`, `top2_mean(matrix)`, `inverse_spearman_weights(matrix)`, `fisher_stat(pvalues)`, `wilson_upper(successes, trials)`, `classify_risk(rate_upper)`, `validate_prediction_payload(row)`.

- [ ] **Step 1: Write failing tests for deterministic folds and leakage-free p-values**

```python
def test_fold_plan_covers_every_index_once():
    folds = fold_plan(101, 5, 20260716)
    heldout = np.concatenate([query for _, query in folds])
    self.assertEqual(sorted(heldout.tolist()), list(range(101)))
    self.assertTrue(all(not set(fit) & set(query) for fit, query in folds))

def test_leave_one_out_excludes_query_score():
    p = leave_one_out_pvalues(np.asarray([1.0, 2.0, 100.0]))
    self.assertAlmostEqual(p[-1], 1.0 / 3.0)
```

- [ ] **Step 2: Run the tests and confirm missing imports/functions fail**

Run: `python3 -m unittest experiments/exp206_210/test_exp206_210_core.py`
Expected: FAIL because `experiments.exp206_210.core` does not yet provide the interfaces.

- [ ] **Step 3: Implement deterministic math with finite-value checks**

```python
ALPHAS = (0.005, 0.01, 0.02, 0.05)
SHADOW_RISK = 0.001
REVIEW_RISK = 0.005
SEED = 20260716

def classify_risk(upper: float) -> str:
    if upper <= SHADOW_RISK:
        return "shadow_hard_diagnostic"
    if upper <= REVIEW_RISK:
        return "standard_review_diagnostic"
    return "research_sensitivity_only"
```

Implement Fisher combination as `-2 * sum(log(max(p, tiny)))`; calibrate the resulting statistic later against TRAIN OOF meta-scores rather than a chi-square assumption.

- [ ] **Step 4: Add tests for robust scaling, weighted top-two fusion, risk boundaries, and forbidden payload fields**

```python
def test_risk_boundaries_are_frozen():
    self.assertEqual(classify_risk(0.001), "shadow_hard_diagnostic")
    self.assertEqual(classify_risk(0.005), "standard_review_diagnostic")
    self.assertEqual(classify_risk(0.0051), "research_sensitivity_only")
```

- [ ] **Step 5: Run and commit**

Run: `python3 -m unittest experiments/exp206_210/test_exp206_210_core.py`
Expected: PASS.

Commit: `git add experiments/exp206_210 && git commit -m "feat: add exp206-210 residual fusion core"`

### Task 2: Fold-safe alternative OOF scoring

**Files:**
- Create: `experiments/exp206_210/alternative_oof.py`
- Extend: `experiments/exp206_210/test_exp206_210_core.py`

**Interfaces:**
- Consumes: `fold_plan` from Task 1 and existing deterministic feature functions in `experiments.exp162.full_coverage_score_core`.
- Produces: `crossfit_alternative_sources(x_train, x_test, *, folds=5, seed=SEED) -> dict[str, np.ndarray]` and `fit_score_source(source, x_fit, x_query, *, seed) -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write a failing test proving query rows cannot affect fit scores**

```python
def test_query_values_do_not_change_fit_scores():
    fit_a, query_a = fit_score_source("spectral", train, test, seed=7)
    changed_query = test.copy(); changed_query[0] *= 100
    fit_b, query_b = fit_score_source("spectral", train, changed_query, seed=7)
    np.testing.assert_allclose(fit_a, fit_b)
    self.assertFalse(np.array_equal(query_a, query_b))
```

- [ ] **Step 2: Verify the test fails because the crossfit scorer is absent**

Run: `python3 -m unittest experiments.exp206_210.test_exp206_210_core.Exp206210AlternativeOOFTests`
Expected: FAIL.

- [ ] **Step 3: Implement source-specific fit/query scoring**

For shape prototypes, call `choose_shapelets(x_fit, 32, lengths, seed)` and never pass query rows to prototype selection. Use `seed + fold_number`; do not derive seed from dataset/family name. Fit `RobustScaler`, center, and MAD-like spread on fit features only, then score fit/query features.

- [ ] **Step 4: Add deterministic and complete-coverage tests**

Assert every source returns arrays of lengths `n_train` and `n_test`, all OOF entries are finite, and repeated calls have byte-identical arrays.

- [ ] **Step 5: Run and commit**

Run: `python3 -m unittest experiments/exp206_210/test_exp206_210_core.py`
Expected: PASS.

Commit: `git add experiments/exp206_210 && git commit -m "feat: add fold-safe alternative source scoring"`

### Task 3: Frozen artifact boundary and Exp206 cache runner

**Files:**
- Create: `experiments/exp206_210/artifacts.py`
- Create: `experiments/exp206_210/run_experiment_206_210.py`
- Create: `experiments/exp206_210/test_exp206_210_structure.py`

**Interfaces:**
- Produces: `load_population_values_only(limit=None)`, `load_current_oof_cache(dataset)`, `load_exp206_cache(dataset)`, `load_test_labels_for_evaluation(dataset)`, `freeze_predictions(...)`, and runner `--stage 206`.

- [ ] **Step 1: Write failing structure tests**

```python
def test_prediction_loaders_do_not_accept_y_test(self):
    self.assertNotIn("y_test", inspect.signature(artifacts.load_population_values_only).parameters)

def test_stage_ids_are_frozen(self):
    self.assertEqual(list(runner.STAGE_META), ["206", "207", "208", "209", "210"])
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments/exp206_210/test_exp206_210_structure.py`
Expected: FAIL because files/interfaces are missing.

- [ ] **Step 3: Implement read-only loaders and Exp206 checkpointed processing**

Use `/Users/minho/Documents/Dataset/univariate_ts.db`, the frozen Exp137 CSV, and the primary-workspace Exp153 cache. Select exactly 251 sorted datasets with `train_normal_count >= 100`. Save Exp206 NPZ files under `/Users/minho/Documents/Dataset/exp206_210_residual_coverage/cache/` and a manifest containing dataset, n_train, n_test, config hash, cache SHA256, fold seed, and status.

- [ ] **Step 4: Implement cache validation and direct-script startup test**

The runner must support `--stage 206 --dataset-limit 2 --workers 1 --output-root <tmp>` and run from a directory outside the repository.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m unittest experiments/exp206_210/test_exp206_210_core.py experiments/exp206_210/test_exp206_210_structure.py`
Expected: PASS.

Commit: `git add experiments/exp206_210 && git commit -m "feat: build exp206 alternative OOF cache"`

### Task 4: Exp207 TRAIN-OOF conditional novelty audit

**Files:**
- Modify: `experiments/exp206_210/run_experiment_206_210.py`
- Extend: `experiments/exp206_210/test_exp206_210_core.py`

**Interfaces:**
- Consumes: verified Exp206 alternative OOF and Exp153-157 current four-source OOF arrays.
- Produces: `exp207_frozen_rules.json`, dataset/source audit CSV, risk summary CSV, completion marker.

- [ ] **Step 1: Write failing tests for conditional novelty arithmetic and frozen rule classification**

Use synthetic boolean source matrices to assert tail counts, conditional counts, phi/Jaccard, Wilson upper, and lane classification.

- [ ] **Step 2: Verify failure, then implement Exp207 without opening TEST labels**

The stage reports every alpha and fusion method. It writes no prediction candidate index and no evaluation metric.

- [ ] **Step 3: Verify source coverage and risk contract**

Exp207 fails unless all 251 datasets have seven OOF vectors with lengths equal to their frozen `n_train`.

- [ ] **Step 4: Run tests and commit**

Run: `python3 -m unittest experiments/exp206_210/test_exp206_210_core.py experiments/exp206_210/test_exp206_210_structure.py`
Expected: PASS.

Commit: `git add experiments/exp206_210 && git commit -m "feat: audit exp207 conditional source novelty"`

### Task 5: Exp208 continuous fusion prediction and delayed evaluation

**Files:**
- Modify: `experiments/exp206_210/run_experiment_206_210.py`
- Extend: `experiments/exp206_210/test_exp206_210_structure.py`

**Interfaces:**
- Consumes: frozen Exp207 rules and complete Exp137 lane indices.
- Produces: three methods x four alphas x 251 label-free prediction rows, manifest/hash, result/summary CSV after delayed evaluation.

- [ ] **Step 1: Write failing tests for exact variant count, lane exclusion, and hash-before-label order**

Expect `3 * 4 * 251 = 3012` full prediction rows. Monkeypatch the label loader and assert prediction completes without calling it; tamper with JSONL and assert evaluation fails before label access.

- [ ] **Step 2: Implement label-free prediction**

Fit robust-z, inverse-correlation weights, and empirical meta references from alternative TRAIN OOF only. Exclude all Exp137 lane indices after candidate generation and record excluded counts separately.

- [ ] **Step 3: Implement delayed evaluation and stratified summaries**

Report `100-199`, `200+`, and `ALL` without choosing a winning method or alpha.

- [ ] **Step 4: Run tests and commit**

Run: `python3 -m unittest experiments/exp206_210/test_exp206_210_core.py experiments/exp206_210/test_exp206_210_structure.py`
Expected: PASS.

Commit: `git add experiments/exp206_210 && git commit -m "feat: evaluate exp208 continuous residual fusion"`

### Task 6: Exp209 seven-source joint normal novelty

**Files:**
- Modify: `experiments/exp206_210/core.py`
- Modify: `experiments/exp206_210/run_experiment_206_210.py`
- Extend: `experiments/exp206_210/test_exp206_210_core.py`

**Interfaces:**
- Produces: `rank_gaussian_diagonal_score(train, query)` and `shrinkage_covariance_score(train, query)` plus two methods x four alphas x 251 prediction rows.

- [ ] **Step 1: Write failing tests for query isolation and covariance abstention**

Assert TRAIN transformation parameters are unchanged when query values change. Assert covariance returns an explicit abstention when finite TRAIN rows are insufficient or the fit is non-finite.

- [ ] **Step 2: Implement both joint scores and OOF calibration**

Use empirical column ranks transformed to bounded Gaussian-like quantiles for the diagonal score. Use fixed shrinkage toward the diagonal for covariance; do not tune shrinkage from TEST metrics.

- [ ] **Step 3: Implement Exp209 prediction/evaluation with the same hash boundary as Exp208**

Expect `2 * 4 * 251 = 2008` full prediction rows, including explicit abstention metadata where needed.

- [ ] **Step 4: Run tests and commit**

Run: `python3 -m unittest experiments/exp206_210/test_exp206_210_core.py experiments/exp206_210/test_exp206_210_structure.py`
Expected: PASS.

Commit: `git add experiments/exp206_210 && git commit -m "feat: add exp209 seven-source normal novelty"`

### Task 7: Exp210 independent reaggregation and documentation

**Files:**
- Create: `experiments/exp206_210/independent_reaggregate.py`
- Create: `docs/EXP206_210_PREREGISTRATION_KO.md`
- Create: `docs/EXP206_210_RUNBOOK_KO.md`
- Modify: `experiments/exp206_210/run_experiment_206_210.py`

**Interfaces:**
- Consumes: hash-verified Exp206-209 artifacts and frozen Exp137/190/198/201/204 summaries.
- Produces: comparison CSV, overlap CSV, bootstrap CSV, report Markdown, manifest, completion marker.

- [ ] **Step 1: Write failing arithmetic tests**

Assert candidate count equals TP+FP, candidate indices are in range, Exp137 lane overlap is zero, all fixed variants exist, and source summary totals equal independent totals.

- [ ] **Step 2: Implement independent reaggregation**

Use fixed seed `20260716` for paired dataset bootstrap intervals. Keep autonomous Exp137, shadow, review, and research-only scopes in separate rows.

- [ ] **Step 3: Write Korean preregistration and runbook**

Document the data grain, fixed variants, risk contracts, smoke/full commands, resume behavior, dashboard visibility, retrospective limitation, and Exp204 prospective gate.

- [ ] **Step 4: Run tests and commit**

Run: `python3 -m unittest experiments/exp206_210/test_exp206_210_core.py experiments/exp206_210/test_exp206_210_structure.py`
Expected: PASS.

Commit: `git add experiments/exp206_210 docs/EXP206_210_PREREGISTRATION_KO.md docs/EXP206_210_RUNBOOK_KO.md && git commit -m "feat: independently reaggregate exp206-210"`

### Task 8: Queue, dashboard, smoke, Agy review, and launch

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Extend: `experiments/exp206_210/test_exp206_210_structure.py`
- Create: `outputs/exp206_210_validation/agy_implementation_review.md`

**Interfaces:**
- Produces: `queue-exp206-210` command and five visible sequential dashboard rows.

- [ ] **Step 1: Write failing queue tests**

Assert exact queue order Exp206 through Exp210, expected datasets 251, stage script/args, per-stage log paths, and dashboard discovery metadata.

- [ ] **Step 2: Implement queue and dashboard registration**

Use one active stage at a time. Stage 206 receives `--workers 7`; dependent stages receive only `--stage`. Preserve unrelated queue entries and do not modify running experiments.

- [ ] **Step 3: Run complete unit and structural tests**

Run: `python3 -m unittest experiments/exp206_210/test_exp206_210_core.py experiments/exp206_210/test_exp206_210_structure.py scratch/test_rank_queue_safety.py scratch/test_rank_dashboard_operational.py`
Expected: PASS.

- [ ] **Step 4: Run a two-dataset smoke batch**

```bash
python3 experiments/exp206_210/run_experiment_206_210.py --stage 206 --dataset-limit 2 --workers 1 --output-root /tmp/exp206_210_smoke
python3 experiments/exp206_210/run_experiment_206_210.py --stage 207 --dataset-limit 2 --output-root /tmp/exp206_210_smoke
python3 experiments/exp206_210/run_experiment_206_210.py --stage 208 --dataset-limit 2 --output-root /tmp/exp206_210_smoke
python3 experiments/exp206_210/run_experiment_206_210.py --stage 209 --dataset-limit 2 --output-root /tmp/exp206_210_smoke
python3 experiments/exp206_210/run_experiment_206_210.py --stage 210 --dataset-limit 2 --output-root /tmp/exp206_210_smoke
```

Expected: all stages complete, no missing rows/errors, prediction payloads contain no evaluation fields, and evaluation manifests verify hashes.

- [ ] **Step 5: Ask Agy for an implementation-contract review and repair important findings**

Provide the design, plan, full branch diff, unit-test log, and smoke artifacts. Require Agy to check OOF leakage, instance-vs-time grain, source coverage, seed/name independence, prediction/evaluation isolation, and risk classification. Codex independently verifies every suggested correction before applying it.

- [ ] **Step 6: Commit reviewed integration**

Commit: `git add run_rank_experiments_sequential.py serve_rank_dashboard.py experiments/exp206_210/test_exp206_210_structure.py outputs/exp206_210_validation/agy_implementation_review.md && git commit -m "feat: queue exp206-210 residual coverage batch"`

- [ ] **Step 7: Register and start sequential execution**

```bash
python3 run_rank_experiments_sequential.py queue-exp206-210
python3 dashboardctl.py start
nohup python3 run_rank_experiments_sequential.py run \
  > /Users/minho/Documents/Dataset/exp206_210_residual_coverage/queue_runner.log 2>&1 &
```

Expected: dashboard shows Exp206 running and Exp207-210 pending; only one experiment process is active; queue runner exits after Exp210 or a visible blocker.
