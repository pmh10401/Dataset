# Exp267-273 Zero-Support FN Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three independent TRAIN-normal score families, recover previously zero-support false negatives into review lanes, and audit a narrow future Hard-alert promotion candidate without changing the current autonomous policy.

**Architecture:** A new `experiments.exp267_273_zero_support_fn_recovery` package reuses the frozen DB loader, preprocessing, calibration contracts, and Exp266 lane artifacts. Exp267 performs read-only diagnostics; Exp268-270 generate source-isolated OOF and TEST scores; Exp271 qualifies sources; Exp272 constructs review-only recovery lanes; Exp273 performs a hypothetical promotion audit. Prediction artifacts are frozen before any TEST labels are read.

**Tech Stack:** Python 3.14, NumPy, SciPy, scikit-learn, PyTorch, Kymatio `Scattering1D`, iisignature, SQLite, unittest, CSV/JSONL/SHA-256 artifacts.

## Global Constraints

- Database path remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Large outputs go to `/Users/minho/Documents/Dataset/exp267_273_zero_support_fn_recovery/`.
- Repository evidence goes to `outputs/exp267_273_zero_support_fn_recovery/`.
- Population accounting is exactly 1,117 unique datasets; no silent omission is allowed.
- Primary alpha is `0.005`; `0.01`, `0.02`, and `0.05` are sensitivity outputs only.
- `n_train < 5` is calibration abstention, `n_train >= 20` enables lightweight forecasting, and `n_train >= 100` enables Causal TCN forecasting. RFF forecasting additionally abstains before any scoring when its TRAIN-only residual-work estimate exceeds `RFF_MAX_TRAIN_RESIDUAL_VALUES = 10_000_000`. Causal TCN additionally abstains before any TEST-value access or training when `n_train_normal * TRAIN target_length > TCN_MAX_TRAIN_ELEMENTS = 5_000_000`. Both are preregistered TRAIN-only resource-safety boundaries, not TEST-performance policies.
- Pre-freeze DB metadata audit for the TCN boundary: among the 253 DB metadata rows with `n_train_normal >= 100`, 252 are within 5,000,000 TRAIN elements and one is over it (`18,378 * 4,000 = 73,512,000`). The frozen 1,117-dataset Exp266 population is a separate snapshot with 251 `n_train_normal >= 100` rows, all within the guard. This is rationale for an explicit resource abstention, not a dataset-name routing rule.
- TEST labels, anomaly positions, family names, dataset names, and TEST length never select features, thresholds, eligibility, candidates, budgets, epochs, or routes.
- Existing Exp253 Hard and Exp266 review-lane indices are immutable through Exp272.
- Tail replacement and TEST-length top-N budgets are forbidden.
- Exp272 additions are human-assisted review only. Exp273 is a hypothetical audit, not a policy activation.
- All results are retrospective counterfactual evidence, not prospective or deployment validation.
- One full experiment runs at a time; CPU stages use at most seven workers and MPS stages are exclusive.

---

## File Structure

Create one focused package:

```text
experiments/exp267_273_zero_support_fn_recovery/
├── __init__.py                 package marker
├── config.py                   immutable paths, seeds, alpha grid, eligibility, model configs
├── contracts.py                source, cache, prediction, lane, and audit dataclasses
├── dependencies.py             Kymatio/iisignature/PyTorch capability audit
├── population.py               frozen 1,117 population and Exp266 input hashes
├── diagnostics.py              Exp267 zero-support diagnostic aggregation
├── scattering_source.py        Exp268 Kymatio scattering features
├── signature_source.py         Exp269 iisignature path features
├── forecast_rff_source.py      Exp270-A RFF ridge forecasting
├── forecast_tcn_source.py      Exp270-B causal TCN forecasting
├── source_registry.py          source names, evidence-family identity, factory
├── source_pipeline.py          resumable cache, prediction freeze, status heartbeat
├── train_risk.py               Exp270-C query-excluded TRAIN-normal risk audit
├── qualification.py            Exp271 independent-source qualification
├── recovery_routing.py         Exp272 immutable-lane review routing
├── promotion_audit.py          Exp273 hypothetical autonomous gates
├── evaluation.py               post-hash TEST-label evaluation
├── independent_reaggregate.py  read-only result reconstruction
├── reporting.py                CSV, Markdown, manifest, and claim reports
└── run_pipeline.py             stage-oriented CLI
```

Tests live under `experiments/exp267_273_zero_support_fn_recovery/tests/`. Add a small queue wrapper at `scratch/run_exp267_273_zero_support_fn_recovery.py`. Modify only the experiment registry sections of `run_rank_experiments_sequential.py` and `serve_rank_dashboard.py`.

---

### Task 1: Freeze Configuration, Dependencies, And Contracts

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/__init__.py`
- Create: `experiments/exp267_273_zero_support_fn_recovery/config.py`
- Create: `experiments/exp267_273_zero_support_fn_recovery/contracts.py`
- Create: `experiments/exp267_273_zero_support_fn_recovery/dependencies.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_contracts.py`

**Interfaces:**
- Produces: `SourceSpec`, `SourceScore`, `FrozenSourcePrediction`, `RecoveryLanes`, `PromotionAudit`; constants `ALPHAS`, `PRIMARY_ALPHA`, `SEED`, `SECOND_SEED`, `RFF_MIN_TRAIN`, `TCN_MIN_TRAIN`.
- Consumes: no experiment-local modules.

- [ ] **Step 1: Write failing configuration and contract tests**

```python
def test_frozen_policy_constants(self):
    self.assertEqual(ALPHAS, (0.005, 0.01, 0.02, 0.05))
    self.assertEqual(PRIMARY_ALPHA, 0.005)
    self.assertEqual(RFF_MIN_TRAIN, 20)
    self.assertEqual(TCN_MIN_TRAIN, 100)

def test_prediction_contract_has_no_evaluation_fields(self):
    fields = {field.name for field in dataclasses.fields(FrozenSourcePrediction)}
    self.assertFalse(fields & {"y_test", "tp", "fp", "fn", "f1", "precision"})
```

- [ ] **Step 2: Run the tests and confirm the package is missing**

Run:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_contracts -v
```

Expected: import failure for the new package.

- [ ] **Step 3: Implement immutable configuration and dataclasses**

Use these exact policy constants:

```python
SEED = 20260721
SECOND_SEED = 20261721
ALPHAS = (0.005, 0.01, 0.02, 0.05)
PRIMARY_ALPHA = 0.005
RFF_MIN_TRAIN = 20
TCN_MIN_TRAIN = 100
REVIEW_RISK_LIMIT = 0.01
HARD_RISK_LIMIT = 0.005
MIN_RETROSPECTIVE_PRECISION = 0.80
```

Define `FrozenSourcePrediction` with only dataset, source, variant, alpha, candidate indices, score hash, eligibility, abstention reason, config hash, and `contains_test_labels=False`. Define `RecoveryLanes` with immutable existing lanes plus new Recovery Shadow and Recovery Priority indices.

- [ ] **Step 4: Implement a dependency audit without importing labels or DB rows**

```python
def dependency_report() -> dict[str, object]:
    return {
        "kymatio": importlib.metadata.version("kymatio"),
        "iisignature": importlib.metadata.version("iisignature"),
        "torch": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
```

Install missing libraries once, then freeze the installed versions in the audit:

```bash
python3 -m pip install --upgrade kymatio iisignature
```

If either import fails on Python 3.14, record `BLOCKER_REPORT.md`; do not replace the algorithm after viewing TEST results.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_contracts -v
git add experiments/exp267_273_zero_support_fn_recovery
git commit -m "feat: freeze Exp267-273 source contracts"
```

Expected: all contract tests pass.

---

### Task 2: Freeze Population And Build Exp267 Diagnostic Harness

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/population.py`
- Create: `experiments/exp267_273_zero_support_fn_recovery/diagnostics.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_diagnostics.py`

**Interfaces:**
- Consumes: Exp266 `03_frozen_predictions.jsonl`, `05_dataset_results.csv`, and `06_global_summary.csv`.
- Produces: `load_frozen_population() -> tuple[str, ...]`, `build_zero_support_diagnostics(...) -> list[dict[str, object]]`.

- [ ] **Step 1: Write failing population and support-count tests**

```python
def test_population_requires_1117_unique_datasets(self):
    with self.assertRaises(PopulationContractError):
        validate_population(("a", "a"), expected=1117)

def test_zero_support_is_computed_from_frozen_candidates(self):
    row = diagnostic_row(true_indices={1, 2}, combined_indices={1}, support={2: ()})
    self.assertEqual(row["remaining_fn_indices"], [2])
    self.assertEqual(row["zero_support_fn_indices"], [2])
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_diagnostics -v
```

Expected: missing `population` and `diagnostics` modules.

- [ ] **Step 3: Implement hash-checked population loading**

Read Exp266 prediction rows first and validate 1,117 unique datasets. Keep label reading in a separate diagnostic evaluation function. Persist input paths and SHA-256 values before computing any label-backed diagnostic.

- [ ] **Step 4: Implement Exp267 outputs**

Write:

```text
00_dependency_audit.json
01_population_contract.json
02_zero_support_fn_dataset.csv
03_zero_support_fn_summary.csv
04_zero_support_fn_train_tiers.csv
05_zero_support_fn_test_size_diagnostic.csv
06_exp267_report.md
```

The summary must independently reproduce 1,553 combined FN, 1,405 zero-support FN, and 148 single-source FN or stop with a contract error.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_diagnostics -v
git add experiments/exp267_273_zero_support_fn_recovery
git commit -m "feat: add Exp267 zero-support diagnostic harness"
```

---

### Task 3: Implement Exp268 Multiscale Scattering Source

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/scattering_source.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_scattering_source.py`

**Interfaces:**
- Consumes: `DatasetBundle` and `feature_distance_result` from Exp239-248 shared infrastructure.
- Produces: `MultiscaleScatteringDeviation.score(bundle, seed, artifact_dir) -> SourceScore`.

- [ ] **Step 1: Write failing shape, determinism, and eligibility tests**

```python
def test_scattering_features_are_fixed_width_and_finite(self):
    rows = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
    features = scattering_features(rows)
    self.assertEqual(features.shape[0], 8)
    self.assertTrue(np.isfinite(features).all())

def test_scattering_is_deterministic(self):
    np.testing.assert_array_equal(scattering_features(self.rows), scattering_features(self.rows))
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_scattering_source -v
```

- [ ] **Step 3: Implement frozen Kymatio features**

Use `kymatio.numpy.Scattering1D` for the primary CPU path with `max_order=2`, `Q=(8, 1)`, and `J=max(2, min(6, floor(log2(target_length)) - 2))`. Aggregate log-absolute coefficients by mean, standard deviation, p90, and maximum over the residual time axis. The config and resulting dimension must be stored in provenance.

- [ ] **Step 4: Use robust OOF distance scoring**

Call `feature_distance_result(..., minimum_train=5)` so every TRAIN query is scored against a model fitted without that query fold. Reject non-finite feature or score arrays.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_scattering_source -v
git add experiments/exp267_273_zero_support_fn_recovery
git commit -m "feat: add Exp268 scattering deviation source"
```

---

### Task 4: Implement Exp269 Ordered Path-Signature Source

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/signature_source.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_signature_source.py`

**Interfaces:**
- Produces: `OrderedPathSignatureDeviation.score(...) -> SourceScore` and `signature_features(rows) -> np.ndarray`.

- [ ] **Step 1: Write failing signature-dimension and order-sensitivity tests**

```python
def test_signature_dimension_matches_library(self):
    features = signature_features(np.ones((4, 64), dtype=np.float32))
    expected_per_window = iisignature.siglength(3, 3)
    self.assertEqual(features.shape, (4, expected_per_window * 3))

def test_reversing_trace_changes_ordered_signature(self):
    forward = signature_features(self.rows)
    backward = signature_features(self.rows[:, ::-1])
    self.assertFalse(np.allclose(forward, backward))
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_signature_source -v
```

- [ ] **Step 3: Implement the fixed three-channel path**

For every aligned trace create channels `(value, first_difference, normalized_time)`. Compute depth-3 signatures for the full trace, first half, and second half using batched `iisignature.sig(path, 3)`. Concatenate the three signatures and reject non-finite output.

- [ ] **Step 4: Add OOF robust scoring and provenance**

Use `feature_distance_result(..., minimum_train=5)` and persist depth, channel count, window definitions, expected signature length, and iisignature version.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_signature_source -v
git add experiments/exp267_273_zero_support_fn_recovery
git commit -m "feat: add Exp269 path signature source"
```

---

### Task 5: Implement Exp270-A RFF Ridge Forecasting

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/forecast_rff_source.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_forecast_rff_source.py`

**Interfaces:**
- Produces: `RFFNormalForecastResidual.score(...) -> SourceScore`.

- [ ] **Step 1: Write failing eligibility and query-exclusion tests**

```python
def test_rff_abstains_below_20_train_rows(self):
    result = RFFNormalForecastResidual().score(bundle(n_train=19), SEED, self.tmp)
    self.assertTrue(result.abstained)
    self.assertEqual(result.abstention_reason, "n_train_below_20")

def test_oof_query_never_appears_in_fit_indices(self):
    result = RFFNormalForecastResidual().score(bundle(n_train=20), SEED, self.tmp)
    self.assertEqual(result.provenance["self_reference_violations"], 0)
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_forecast_rff_source -v
```

- [ ] **Step 3: Implement frozen forecast windows and RFF model**

Use context lengths `(8, 16, 32)` clipped to TRAIN-derived target length and a forecast horizon of `max(1, min(8, target_length // 16))`. Use 256 `RBFSampler` components with seed `20260721`. Select ridge alpha by TRAIN-only cross-fitting over `(0.01, 0.1, 1.0, 10.0)` and freeze the lowest TRAIN validation loss with numeric-alpha tie breaking. Before any RFF transform, estimate the full-model residual spool from only `n_train`, TRAIN target length, the frozen contexts, and horizon. If the estimate exceeds `RFF_MAX_TRAIN_RESIDUAL_VALUES = 10_000_000`, abstain with `train_residual_work_exceeds_10000000`; do not inspect TEST values, count, labels, or metrics. This resource boundary leaves the TCN source available for larger eligible datasets.

- [ ] **Step 4: Implement trace-level residual score**

Aggregate absolute forecast residual by median, p95, maximum, and longest run above the TRAIN residual p95. Compute the p95 through an exact float64 disk spool with in-place linear-quantile partitioning, then delete the temporary spool. For OOF, robustly standardize the four aggregates using trace features from the selected outer model on its outer-fit TRAIN rows; for TEST, use trace features from the full model on all TRAIN rows. Sum the four standardized aggregates with equal weights.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_forecast_rff_source -v
git add experiments/exp267_273_zero_support_fn_recovery
git commit -m "feat: add Exp270 lightweight normal forecasting"
```

---

### Task 6: Implement Exp270-B Causal TCN Forecasting

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/forecast_tcn_source.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_forecast_tcn_source.py`

**Interfaces:**
- Produces: `CausalTCNNormalForecast.score(...) -> SourceScore`, `choose_device(requested, benchmark_record) -> str`, and `benchmark_device(...) -> dict[str, object]`.

- [ ] **Step 1: Write failing causality, eligibility, and label-boundary tests**

```python
def test_tcn_abstains_below_100_train_rows(self):
    result = CausalTCNNormalForecast().score(bundle(n_train=99), SEED, self.tmp)
    self.assertEqual(result.abstention_reason, "n_train_below_100")

def test_prediction_prefix_is_not_affected_by_future_mutation(self):
    model = tiny_frozen_tcn()
    original = model(self.x).detach().numpy()
    changed = self.x.clone(); changed[..., -8:] += 100
    mutated = model(changed).detach().numpy()
    np.testing.assert_allclose(original[..., :-8], mutated[..., :-8], atol=1e-6)
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_forecast_tcn_source -v
```

- [ ] **Step 3: Implement the fixed causal network**

Use four residual `Conv1d` blocks with 32 channels, kernel size 3, dilations `(1, 2, 4, 8)`, causal left padding, GELU, and dropout 0.1. Predict an eight-point future patch from the last causal hidden state. Use AdamW with learning rate `1e-3`, weight decay `1e-4`, batch size 32, maximum 20 epochs, patience 3 on fit TRAIN loss only, Huber loss, and seeds `20260721` and `20261721`. Before TEST-value access, abstain when the TRAIN-only workload `n_train_normal * TRAIN target_length` exceeds the frozen `TCN_MAX_TRAIN_ELEMENTS = 5_000_000`. The boundary is a pre-evaluation resource-safety contract; it is not selected from TEST metrics.

- [ ] **Step 4: Implement TRAIN-only cross-fitting and score variants**

Build deterministic five-fold OOF scores for TRAIN rows. Fit the final model on all TRAIN-normal rows for TEST scoring. Persist per-seed OOF and TEST scores. Report `seed_median` and `seed_intersection` candidate variants; count both as one forecasting evidence family.

- [ ] **Step 5: Add CPU/MPS smoke benchmark**

Benchmark one fixed 100-row synthetic batch for three warm iterations and five timed iterations. Select MPS only when it is available, produces finite scores within `rtol=1e-4, atol=1e-5` of CPU, and median runtime is lower. Freeze the complete benchmark record and canonical record SHA-256 before full prediction; its config hash must exactly equal the current benchmark payload and all required fields must validate. A direct, stale, incomplete, or tampered MPS record must remain CPU. Record a deterministic TRAIN-only estimated-batch/runtime report from this benchmark.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_forecast_tcn_source -v
git add experiments/exp267_273_zero_support_fn_recovery
git commit -m "feat: add Exp270 causal TCN forecasting"
```

---

### Task 7: Build Shared Source Cache And Frozen Prediction Pipeline

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/source_registry.py`
- Create: `experiments/exp267_273_zero_support_fn_recovery/source_pipeline.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_source_pipeline.py`

**Interfaces:**
- Consumes: three source classes and Exp239 `load_bundle`, `crossfit_plan`, and calibration helpers.
- Produces: `cache_source(...)`, `freeze_source_predictions(...)`, `verify_prediction_hash(...)`.

- [ ] **Step 1: Write failing source-family and resume tests**

```python
def test_tcn_variants_count_as_one_evidence_family(self):
    self.assertEqual(registration("tcn_seed_median").family, "normal_forecast")
    self.assertEqual(registration("tcn_seed_intersection").family, "normal_forecast")

def test_valid_cache_is_reused_only_when_config_hash_matches(self):
    first = build_cache_unit(self.root, "scattering", "d1", reset=False)
    second = build_cache_unit(self.root, "scattering", "d1", reset=False)
    self.assertEqual(first["cache_sha256"], second["cache_sha256"])
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_source_pipeline -v
```

- [ ] **Step 3: Implement atomic per-dataset cache and manifest**

Persist `train_oof`, `test_scores`, optional per-seed arrays, metadata JSON, config hash, DB hash, and cache hash. CPU sources use `ProcessPoolExecutor(max_workers=7)`; TCN enforces one worker and exclusive device use. Any error writes `cache_errors.csv` and makes the source incomplete.

- [ ] **Step 4: Implement label-free prediction freeze**

For every source, variant, dataset, and alpha, calculate empirical p-values from TRAIN OOF scores and write candidate indices to `frozen_source_predictions.jsonl`. The JSONL must contain no label or metric field. Canonically sort rows by source, variant, dataset, alpha and write SHA-256 to `source_prediction_manifest.json`.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_source_pipeline -v
git add experiments/exp267_273_zero_support_fn_recovery
git commit -m "feat: add resumable Exp268-270 source pipeline"
```

---

### Task 8: Implement Exp270-C TRAIN-Risk And Exp271 Qualification

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/train_risk.py`
- Create: `experiments/exp267_273_zero_support_fn_recovery/qualification.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_qualification.py`

**Interfaces:**
- Produces: `audit_train_risk(...)`, `qualify_sources(...)`, and `collapse_correlated_families(...)`.

- [ ] **Step 1: Write failing Wilson and evidence-collapse tests**

```python
def test_wilson_upper_bound_matches_known_case(self):
    self.assertAlmostEqual(wilson_upper(successes=0, trials=100, confidence=0.95), 0.0263427, places=6)

def test_variants_of_one_family_cannot_double_vote(self):
    votes = collapse_votes({"tcn_seed_median": {3}, "tcn_seed_intersection": {3}})
    self.assertEqual(votes[3], {"normal_forecast"})
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_qualification -v
```

- [ ] **Step 3: Implement query-excluded TRAIN pseudo-run risk**

For each OOF query, remove its score from the calibration reference, compute the p-value, and count a normal candidate at each alpha. Report rate and one-sided 95% Wilson upper bound by source, variant, TRAIN tier, and dataset.

- [ ] **Step 4: Implement post-freeze qualification**

After verifying the source prediction hash, read TEST labels and calculate standalone, Hard-confirmation, frozen-review-priority, outside-lane, and zero-support-recovery views. Compute Spearman correlation only from aligned TRAIN OOF scores. The frozen correlation contract is absolute Spearman `>= 0.90` with at least `20` aligned finite TRAIN OOF scores. Variants sharing a registered family or evidence identity collapse regardless of observed correlation; distinct qualifying correlations collapse through connected components. Fewer than 20 aligned rows or a non-finite correlation means independence is unverified, so the source cannot add an independent vote or membership. This is TRAIN-only diagnostic information and cannot be retuned after TEST evaluation.

Before reading TEST labels, write `exp272_source_membership.json`. Include a
source only when its required coverage is complete, its TRAIN-normal review-risk
Wilson upper bound is at most 0.01, its output is reproducible, and its
registered algorithm family is distinct. TEST TP, FP, precision, F1, and
recovered-FN counts cannot change this manifest.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_qualification -v
git add experiments/exp267_273_zero_support_fn_recovery
git commit -m "feat: add Exp270 risk and Exp271 qualification"
```

---

### Task 9: Implement Exp272 Recovery Review Routing

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/recovery_routing.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_recovery_routing.py`

**Interfaces:**
- Consumes: hash-verified Exp266 lane rows, Exp266 certified source support, the label-free `exp272_source_membership.json`, and new source predictions.
- Produces: `route_recovery_review(...) -> RecoveryLanes`.

- [ ] **Step 1: Write failing immutability and routing tests**

```python
def test_existing_lanes_are_immutable(self):
    result = route_recovery_review(existing=self.existing, source_candidates=self.sources)
    self.assertEqual(result.hard_indices, self.existing.hard_indices)
    self.assertEqual(result.standard_indices, self.existing.standard_indices)

def test_three_and_two_independent_families_route_separately(self):
    result = route_recovery_review(existing=empty_lanes(), source_candidates=self.sources)
    self.assertEqual(result.recovery_shadow_indices, (4,))
    self.assertEqual(result.recovery_priority_indices, (7,))
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_recovery_routing -v
```

- [ ] **Step 3: Implement ordered routing**

Combine the five existing Exp266 evidence families with the new families frozen
in `exp272_source_membership.json`. Preserve existing lanes first, then Data
Quality Review, then three-or-more-family Recovery Shadow, then
exactly-two-family Recovery Priority. Keep one-source candidates as No Alert and
write them to a separate sensitivity table. Ensure all lanes are pairwise
disjoint and every index is within TEST instance bounds.

- [ ] **Step 4: Freeze Exp272 predictions**

Write one label-free row per dataset with existing lanes, new recovery lanes, support families, source eligibility, config hash, and input hashes. Verify all 1,117 datasets and write the canonical prediction SHA-256.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_recovery_routing -v
git add experiments/exp267_273_zero_support_fn_recovery
git commit -m "feat: add Exp272 recovery review routing"
```

---

### Task 10: Implement Exp273 Promotion Audit And Separated Evaluation

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/promotion_audit.py`
- Create: `experiments/exp267_273_zero_support_fn_recovery/evaluation.py`
- Create: `experiments/exp267_273_zero_support_fn_recovery/independent_reaggregate.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_evaluation.py`

**Interfaces:**
- Produces: `evaluate_frozen_predictions(...)`, `audit_promotion(...) -> PromotionAudit`, and `reaggregate(...)`.

- [ ] **Step 1: Write failing hash, metric-separation, and promotion-gate tests**

```python
def test_evaluation_rejects_tampered_prediction(self):
    self.prediction.write_text(self.prediction.read_text() + " ")
    with self.assertRaises(PredictionHashError):
        evaluate_frozen_predictions(self.prediction, self.manifest)

def test_promotion_requires_all_frozen_gates(self):
    audit = audit_promotion(independent_groups=2, train_u95=0.004, precision=0.81,
                            overall_precision=0.82, invariant=True)
    self.assertTrue(audit.eligible_for_future_experiment)
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_evaluation -v
```

- [ ] **Step 3: Implement post-hash TEST evaluation**

Evaluation opens the SQLite label column only after verifying source and Exp272 prediction hashes. Report autonomous Hard metrics unchanged from Exp253, review metrics separately, and combined F1 only as human-assisted diagnostic. Calculate full-population and preregistered TRAIN/TEST-size diagnostic slices.

- [ ] **Step 4: Implement the hypothetical promotion audit**

Evaluate the frozen at-least-two-independent-groups subset against `HARD_RISK_LIMIT=0.005`, incremental retrospective precision `>=0.80`, overall precision `>=0.80`, and invariance completion. Never write promoted indices into the autonomous Hard field.

- [ ] **Step 5: Implement independent reaggregation**

Read result CSVs only, reconstruct every total and lane overlap count, and compare against saved summaries with exact integer equality and floating tolerance `1e-12`. Fail if the 1,117-dataset population, total anomalies, or Exp253 Hard parity differs.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_evaluation -v
git add experiments/exp267_273_zero_support_fn_recovery
git commit -m "feat: add Exp273 promotion audit and evaluation"
```

---

### Task 11: Add Reporting, CLI, Queue, And Dashboard Visibility

**Files:**
- Create: `experiments/exp267_273_zero_support_fn_recovery/reporting.py`
- Create: `experiments/exp267_273_zero_support_fn_recovery/run_pipeline.py`
- Create: `scratch/run_exp267_273_zero_support_fn_recovery.py`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_pipeline.py`
- Test: `experiments/exp267_273_zero_support_fn_recovery/tests/test_dashboard.py`

**Interfaces:**
- Produces: CLI stages `audit`, `267`, `268`, `269`, `270a`, `270b`, `270c`, `271`, `272`, `273`, `evaluate`, and `all`.

- [ ] **Step 1: Write failing CLI and status tests**

```python
def test_status_has_dashboard_contract_fields(self):
    row = build_status("experiment_268_scattering", "running", 3, 1117, "dataset-a")
    required = {"id", "status", "phase", "done", "expected", "progress_percent",
                "current_dataset", "errors", "updated_at", "stdout_log"}
    self.assertTrue(required <= set(row))

def test_queue_order_is_sequential(self):
    self.assertEqual(EXP267_273_BATCH, EXPECTED_EXP267_273_BATCH)
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_pipeline experiments.exp267_273_zero_support_fn_recovery.tests.test_dashboard -v
```

- [ ] **Step 3: Implement the stage CLI and atomic heartbeat**

Every stage writes `status.json`, `rank_dashboard_external_live_run.json`, and `rank_experiments_dashboard_heartbeat.json` atomically. Status includes recent throughput and ETA when at least two dataset timestamps exist. A dashboard write failure is logged but cannot delete cache or prediction artifacts.

- [ ] **Step 4: Register the sequential queue**

Register these IDs in order:

```python
EXP267_273_BATCH = (
    "experiment_267_zero_support_fn_diagnostic",
    "experiment_268_multiscale_scattering_deviation",
    "experiment_269_ordered_path_signature_deviation",
    "experiment_270a_rff_normal_forecast",
    "experiment_270b_causal_tcn_normal_forecast",
    "experiment_270c_forecast_risk_audit",
    "experiment_271_new_source_qualification",
    "experiment_272_recovery_review_selector",
    "experiment_273_autonomous_promotion_audit",
)
```

The runner must launch one ID at a time and preserve failed status rather than marking it completed.

- [ ] **Step 5: Add dashboard labels and artifact readers**

Display Hard and review metrics in separate columns. Show learner eligibility, abstention, current source, dataset progress, recent throughput, ETA, error count, and prediction hash. Do not label Exp273 as an activated Hard policy.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_pipeline experiments.exp267_273_zero_support_fn_recovery.tests.test_dashboard -v
git add experiments/exp267_273_zero_support_fn_recovery scratch/run_exp267_273_zero_support_fn_recovery.py run_rank_experiments_sequential.py serve_rank_dashboard.py
git commit -m "feat: queue and monitor Exp267-273"
```

---

### Task 12: Smoke, Full Sequential Run, Verification, And Final Evidence

**Files:**
- Create during execution: `outputs/exp267_273_zero_support_fn_recovery/`
- Create during execution: `/Users/minho/Documents/Dataset/exp267_273_zero_support_fn_recovery/`
- Modify after verified run: `docs/task.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: complete frozen artifacts, reports, manifest, and dashboard history.

- [ ] **Step 1: Run syntax and full unit tests**

```bash
python3 -m py_compile experiments/exp267_273_zero_support_fn_recovery/*.py scratch/run_exp267_273_zero_support_fn_recovery.py
python3 -m unittest discover -s experiments/exp267_273_zero_support_fn_recovery/tests -v
```

Expected: all tests pass before any smoke or full run.

- [ ] **Step 2: Run a 10-dataset CPU smoke**

```bash
python3 -m experiments.exp267_273_zero_support_fn_recovery.run_pipeline \
  --stage all --dataset-limit 10 --workers 2 --device cpu \
  --output-root /Users/minho/Documents/Dataset/exp267_273_zero_support_fn_recovery_smoke
```

Expected: 10 datasets, zero silent errors, no labels or metrics in prediction JSONL, disjoint lanes, and matching independent reaggregation.

- [ ] **Step 3: Run the frozen CPU/MPS benchmark**

```bash
python3 -m experiments.exp267_273_zero_support_fn_recovery.run_pipeline \
  --stage device-benchmark \
  --output-root /Users/minho/Documents/Dataset/exp267_273_zero_support_fn_recovery
```

Expected: a hash-recorded CPU or MPS decision before Exp270-B full execution.

- [ ] **Step 4: Queue the full sequential batch**

```bash
python3 run_rank_experiments_sequential.py queue-exp267-273
python3 run_rank_experiments_sequential.py run-queue
```

Expected: one active process; remaining experiments remain queued and visible on the dashboard.

- [ ] **Step 5: Verify every frozen artifact**

Check:

```text
datasets = 1117
missing datasets = 0
lane overlap = 0
out-of-range indices = 0
prediction label fields = 0
prediction metric fields = 0
Exp253 Hard parity mismatches = 0 through Exp272
independent reaggregation = PASS
```

- [ ] **Step 6: Write final reports and manifest**

Required report sections: Exp267 blind spot, each source's TRAIN risk and coverage, source independence, zero-support FN recovery, review workload, hypothetical Exp273 gates, Hard-versus-review metric boundary, retrospective limitation, unresolved historical provenance, and prospective validation requirement.

Report the preregistered usefulness levels without using them to alter policy:
at least 70 zero-support FN recovered is the 5% minimum signal and at least 141
is the 10% strong signal. Report pass or fail exactly as observed.

- [ ] **Step 7: Update task history only after verification**

Add a concise Exp267-273 completion entry to `docs/task.md`. Do not describe a source, alpha, or learner as selected or optimal from TEST results.

- [ ] **Step 8: Run final verification and commit evidence**

```bash
git diff --check
python3 -m unittest discover -s experiments/exp267_273_zero_support_fn_recovery/tests -v
git add outputs/exp267_273_zero_support_fn_recovery docs/task.md
git commit -m "results: record Exp267-273 FN recovery evidence"
```

Expected: clean diff checks and passing tests. Large cache, DB, raw JSONL, and logs remain outside Git.

---

## Execution Decision Boundary

Do not choose a different alpha, epoch count, TCN architecture, RFF dimension,
evidence threshold, or eligibility boundary after observing TEST TP/FP/F1.
Failed gates remain reported failures. Any improvement suggested after evaluation
requires a new experiment number and a new preregistration.
