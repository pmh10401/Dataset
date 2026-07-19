# Exp239-248 Diverse Score Source Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and sequentially run ten TRAIN-normal anomaly score sources that qualify confirmation, review-priority, and outside-lane coverage without changing frozen Exp236 alert lanes.

**Architecture:** A shared foundation loads and aligns the 1,117 datasets, creates deterministic OOF folds, and exposes one `ScoreSource` contract. Ten focused source modules write hashed, label-free score caches; prediction converts scores to frozen empirical p-value candidates, and evaluation reads TEST labels only after prediction hashes are verified. Lightweight sources use seven CPU workers, while neural sources run one at a time and may use MPS after a parity smoke test.

**Tech Stack:** Python 3.13, NumPy 2.3, SciPy 1.17, scikit-learn 1.8, PyWavelets 1.8, PyTorch 2.13 with optional MPS, SQLite, JSONL, NPZ, CSV, unittest.

## Global Constraints

- Database remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Population is exactly the frozen 1,117 Exp236-238 datasets.
- TRAIN contains normal traces only; TEST labels are evaluation-only.
- Frozen alpha grid is `(0.005, 0.01, 0.02, 0.05)`.
- Dataset and family names are lookup keys only and cannot change feature, score, threshold, source, or lane policy.
- TEST length cannot choose a threshold, source, candidate count, or budget.
- `n_train < 5` is explicit calibration abstention; model-specific minimum TRAIN sizes remain fixed at 10, 20, or 50 as specified.
- Prediction files contain no labels, TP, FP, F1, actual anomaly positions, or TEST-derived policy flags.
- Exp236 Hard, Standard review, and Priority review indices remain byte-for-byte unchanged.
- No source is promoted into Hard during Exp239-248.
- TEST results cannot select alpha, model seed, transform, source, or hyperparameter.
- Existing Exp236-238 code and results are never overwritten.

---

## File Structure

Create the package `experiments/exp239_248_diverse_score_sources/` with these responsibilities:

- `config.py`: experiment IDs, source names, seeds, frozen hyperparameters, paths.
- `contracts.py`: `DatasetBundle`, `ScoreResult`, `SourceConfig`, and `ScoreSource` protocol.
- `foundation.py`: DB loading, TRAIN-derived alignment, sanitization, deterministic folds, hashing.
- `calibration.py`: OOF empirical p-values, attainable-resolution flags, candidate extraction.
- `robust_distance.py`: reusable robust scaling and shrinkage-distance backend.
- `wavelet_energy.py`: Exp239.
- `ordinal_mmd.py`: Exp240.
- `symbolic_motif.py`: Exp241.
- `complexity_signature.py`: Exp242.
- `recurrence_dynamics.py`: Exp243.
- `state_space.py`: Exp244.
- `fourier_residual.py`: Exp245.
- `transformation_consistency.py`: Exp246.
- `masked_patch.py`: Exp247 and reusable frozen embeddings.
- `normal_density.py`: Exp248 using Exp247 embeddings.
- `registry.py`: source-name to implementation mapping and dependency ordering.
- `cache_pipeline.py`: resumable per-source/per-dataset cache and manifest.
- `prediction.py`: label-free prediction JSONL and hash manifest.
- `evaluation.py`: post-freeze evaluation against TEST labels.
- `reporting.py`: qualification tables, correlations, summaries, manifests.
- `independent_reaggregate.py`: independent result verification.
- `run_pipeline.py`: CLI stages and dashboard heartbeat.
- `tests/`: focused contract, source, structure, invariance, and reporting tests.

Modify `serve_rank_dashboard.py` only to register Exp239-248 labels and result paths. Add a queue launcher under `scratch/run_exp239_248_sequential.py` rather than expanding the existing oversized sequential runner.

---

### Task 1: Foundation And Information-Boundary Contracts

**Files:**
- Create: `experiments/exp239_248_diverse_score_sources/__init__.py`
- Create: `experiments/exp239_248_diverse_score_sources/config.py`
- Create: `experiments/exp239_248_diverse_score_sources/contracts.py`
- Create: `experiments/exp239_248_diverse_score_sources/foundation.py`
- Create: `experiments/exp239_248_diverse_score_sources/calibration.py`
- Create: `experiments/exp239_248_diverse_score_sources/robust_distance.py`
- Create: `experiments/exp239_248_diverse_score_sources/tests/__init__.py`
- Create: `experiments/exp239_248_diverse_score_sources/tests/test_foundation.py`

**Interfaces:**
- Produces: `load_bundle(dataset: str) -> DatasetBundle`.
- Produces: `deterministic_folds(n_train: int, seed: int, folds: int = 5) -> tuple[np.ndarray, ...]`.
- Produces: `empirical_candidates(train_oof: np.ndarray, test_scores: np.ndarray, alpha: float) -> CalibrationResult`.
- Produces: `robust_distance(train_reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]` returning OOF-compatible reference scores and query scores.

- [ ] **Step 1: Write failing foundation tests**

```python
class FoundationContractTest(unittest.TestCase):
    def test_fold_assignment_ignores_dataset_name(self):
        self.assertEqual(
            tuple(map(tuple, deterministic_folds(23, seed=20260720))),
            tuple(map(tuple, deterministic_folds(23, seed=20260720))),
        )

    def test_empirical_candidates_reports_resolution(self):
        result = empirical_candidates(np.arange(9.0), np.asarray([20.0]), 0.005)
        self.assertEqual(result.attainable_p, 0.1)
        self.assertTrue(result.resolution_blocked)
        self.assertEqual(result.indices, ())

    def test_loader_source_does_not_select_test_labels(self):
        source = inspect.getsource(load_bundle).lower()
        self.assertNotIn("select label", source)
        self.assertNotIn("select i.label", source)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_foundation`

Expected: import failure because the package and contracts do not exist.

- [ ] **Step 3: Implement contracts and deterministic foundation**

```python
@dataclass(frozen=True)
class ScoreResult:
    source: str
    train_oof: np.ndarray
    test_scores: np.ndarray
    abstained: bool
    abstention_reason: str | None
    provenance: Mapping[str, object]

@dataclass(frozen=True)
class CalibrationResult:
    pvalues: np.ndarray
    indices: tuple[int, ...]
    attainable_p: float
    resolution_blocked: bool

class ScoreSource(Protocol):
    name: str
    minimum_train: int
    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> ScoreResult: ...
```

Implement `load_bundle()` by reusing the Exp226-235 values-only loader and alignment logic. Derive target length from TRAIN-normal traces only. Replace NaN and infinity using TRAIN-derived finite medians, then apply per-trace z normalization with a unit fallback for constant traces.

- [ ] **Step 4: Run foundation tests GREEN**

Run: `python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_foundation`

Expected: all tests pass.

- [ ] **Step 5: Commit foundation**

```bash
git add experiments/exp239_248_diverse_score_sources
git commit -m "feat: add Exp239-248 score source foundation"
```

---

### Task 2: Lightweight Sources Exp239-243

**Files:**
- Create: `experiments/exp239_248_diverse_score_sources/wavelet_energy.py`
- Create: `experiments/exp239_248_diverse_score_sources/ordinal_mmd.py`
- Create: `experiments/exp239_248_diverse_score_sources/symbolic_motif.py`
- Create: `experiments/exp239_248_diverse_score_sources/complexity_signature.py`
- Create: `experiments/exp239_248_diverse_score_sources/recurrence_dynamics.py`
- Create: `experiments/exp239_248_diverse_score_sources/tests/test_lightweight_sources.py`

**Interfaces:**
- Consumes: `DatasetBundle`, deterministic folds, and `robust_distance()` from Task 1.
- Produces: five classes implementing `ScoreSource.score()` with source names frozen in the design.

- [ ] **Step 1: Write failing feature and OOF-isolation tests**

```python
class LightweightSourceTest(unittest.TestCase):
    def test_all_sources_return_aligned_finite_scores(self):
        for source in LIGHTWEIGHT_SOURCES:
            result = source.score(self.bundle, seed=20260720, artifact_dir=self.temp)
            self.assertEqual(result.train_oof.shape, (len(self.bundle.train_normal),))
            self.assertEqual(result.test_scores.shape, (len(self.bundle.test_values),))
            self.assertTrue(np.isfinite(result.train_oof).all())
            self.assertTrue(np.isfinite(result.test_scores).all())

    def test_pseudo_test_trace_is_excluded_from_reference(self):
        result = SymbolicMotifRarity().score(self.bundle, 20260720, self.temp)
        self.assertTrue(result.provenance["cross_fitted"])
        self.assertEqual(result.provenance["self_reference_violations"], 0)
```

- [ ] **Step 2: Run lightweight tests RED**

Run: `python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_lightweight_sources`

Expected: source modules are missing.

- [ ] **Step 3: Implement Exp239 wavelet energy**

Use `pywt.wavedec(trace, "db4", level=level)` and emit normalized energy, max local squared coefficient, entropy, and concentration per level. Cross-fit robust covariance on TRAIN features and score TEST features with the frozen model.

- [ ] **Step 4: Implement Exp240 ordinal MMD**

Generate ordinal histograms for `(dimension, delay)` in `((3,1),(3,2),(3,4),(4,1),(4,2),(4,4))` when valid. Compute TRAIN-only median RBF bandwidth and MMD between one trace histogram and fold-reference histograms.

- [ ] **Step 5: Implement Exp241 symbolic motif rarity**

Use TRAIN-only Gaussian SAX breakpoints, alphabet size 5, PAA sizes 16/32/64, and n-grams 2/3. For each fold, estimate Laplace-smoothed motif probabilities without the held-out fold. Return mean negative log probability plus unseen ratio.

- [ ] **Step 6: Implement Exp242 complexity signature**

Emit the exact fixed feature vector from the design and apply shrinkage Mahalanobis scoring. Clamp entropy probabilities with `1e-12`; constant traces remain finite and receive explicit flat-line features.

- [ ] **Step 7: Implement Exp243 recurrence dynamics**

Deterministically resample TRAIN-aligned traces to at most 256 points, build delay embeddings with dimensions 2/3, select delay from TRAIN-only autocorrelation clipped to 1..16, and use a TRAIN distance 10th percentile recurrence threshold. Emit recurrence rate, determinism, laminarity, trapping time, and line entropies before robust distance scoring.

- [ ] **Step 8: Run source tests and compile**

Run:

```bash
python3 -m py_compile experiments/exp239_248_diverse_score_sources/*.py
python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_lightweight_sources
```

Expected: all tests pass; score vectors are finite and aligned.

- [ ] **Step 9: Commit lightweight sources**

```bash
git add experiments/exp239_248_diverse_score_sources
git commit -m "feat: add Exp239-243 lightweight score sources"
```

---

### Task 3: Statistical Model Sources Exp244-245

**Files:**
- Create: `experiments/exp239_248_diverse_score_sources/state_space.py`
- Create: `experiments/exp239_248_diverse_score_sources/fourier_residual.py`
- Create: `experiments/exp239_248_diverse_score_sources/tests/test_statistical_sources.py`

**Interfaces:**
- Consumes: `DatasetBundle`, fold plan, and source result contracts.
- Produces: `RobustStateSpaceInnovation` and `SmoothFourierNormalResidual`.

- [ ] **Step 1: Write failing statistical source tests**

```python
def test_state_space_detects_level_innovation(self):
    clean = np.sin(np.linspace(0, 4 * np.pi, 128))
    shifted = clean.copy(); shifted[64:] += 3.0
    source = RobustStateSpaceInnovation()
    self.assertGreater(source.trace_score(shifted), source.trace_score(clean))

def test_fourier_harmonics_are_train_derived(self):
    result = SmoothFourierNormalResidual().score(bundle, 20260720, temp_dir)
    self.assertEqual(result.provenance["uses_test_frequency_selection"], False)
    self.assertEqual(result.provenance["retained_harmonics"], 8)
```

- [ ] **Step 2: Run statistical tests RED**

Run: `python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_statistical_sources`

Expected: missing modules.

- [ ] **Step 3: Implement a dependency-free robust state-space source**

Implement a local-level and damped-trend Kalman filter directly with NumPy so `statsmodels` is not required. Estimate process and observation scales from TRAIN-normal first differences using MAD. Aggregate standardized innovations with median, p95, maximum, and longest exceedance run. Cross-fit every TRAIN score.

- [ ] **Step 4: Implement smooth Fourier residual**

Rank FFT frequencies by aggregate TRAIN-normal energy, retain exactly eight available harmonics, and select ridge strength from `(1e-4, 1e-3, 1e-2, 1e-1)` using TRAIN-only folds. Standardize residual energy and coefficient distance separately on TRAIN OOF, then average them with fixed 0.5 weights.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_statistical_sources
git add experiments/exp239_248_diverse_score_sources
git commit -m "feat: add Exp244-245 statistical score sources"
```

---

### Task 4: Neural And Density Sources Exp246-248

**Files:**
- Create: `experiments/exp239_248_diverse_score_sources/transformation_consistency.py`
- Create: `experiments/exp239_248_diverse_score_sources/masked_patch.py`
- Create: `experiments/exp239_248_diverse_score_sources/normal_density.py`
- Create: `experiments/exp239_248_diverse_score_sources/tests/test_model_sources.py`

**Interfaces:**
- Produces: `TransformationConsistency`, `MaskedPatchContextError`, and `NonlinearNormalDensity`.
- `MaskedPatchContextError` writes `train_embeddings` and `test_embeddings` into its artifact NPZ.
- `NonlinearNormalDensity.score_from_embeddings(...)` consumes only those frozen embeddings and TRAIN fold assignments.

- [ ] **Step 1: Write failing determinism, MPS parity, and eligibility tests**

```python
def test_model_source_abstains_below_minimum_train(self):
    result = TransformationConsistency().score(bundle_with_19_train, 20260720, temp_dir)
    self.assertTrue(result.abstained)
    self.assertEqual(result.abstention_reason, "n_train_below_20")

def test_masked_patch_seed_is_deterministic(self):
    first = MaskedPatchContextError(epochs=20).score(bundle, 20260720, temp_a)
    second = MaskedPatchContextError(epochs=20).score(bundle, 20260720, temp_b)
    np.testing.assert_allclose(first.test_scores, second.test_scores, rtol=1e-5, atol=1e-6)

def test_density_never_fits_test_embedding(self):
    result = NonlinearNormalDensity().score_from_embeddings(train, test, folds, seed=20260720)
    self.assertFalse(result.provenance["test_used_for_fit"])
```

- [ ] **Step 2: Run model tests RED**

Run: `python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_model_sources`

Expected: missing model sources.

- [ ] **Step 3: Implement the shared compact encoder**

Use three `Conv1d` blocks with channels `(16, 32, 32)`, kernel sizes `(9, 5, 3)`, GELU, adaptive average pooling, and a 32-dimensional embedding. Fix Torch deterministic algorithms where supported and record device, seed, epochs, and model-state hash.

- [ ] **Step 4: Implement Exp246 transformation consistency**

Train only on normal traces using the frozen crop, mask, jitter, and amplitude transforms. Score each trace by the mean normalized embedding dispersion across transforms. Run both frozen seeds and store per-seed scores plus their median; do not select a seed from TEST results.

- [ ] **Step 5: Implement Exp247 masked patch context error**

Use the compact encoder and a small convolutional decoder. Freeze mask ratio at 0.25 and patch size at 16 after padding from TRAIN-derived target length. Score with equal weights on masked reconstruction error and latent nearest-normal distance after TRAIN OOF normalization. Persist embeddings for Exp248.

- [ ] **Step 6: Implement Exp248 compact RealNVP density**

Use four affine coupling layers of width 32 on Exp247's 32-dimensional embeddings, fixed standard-normal base, 40 epochs, early stopping from TRAIN folds only, and two fixed seeds. Report median negative log density and typicality-corrected rank as separate sensitivity variants.

- [ ] **Step 7: Benchmark CPU and MPS smoke parity**

Run one eligible dataset on CPU and MPS with the same seed. Require finite aligned scores, candidate-index equality at all four alphas, and at least 1.1x elapsed-time speedup before enabling MPS in the full run. Otherwise record CPU fallback.

- [ ] **Step 8: Run tests and commit**

```bash
python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_model_sources
git add experiments/exp239_248_diverse_score_sources
git commit -m "feat: add Exp246-248 model score sources"
```

---

### Task 5: Registry, Resumable Cache, And Frozen Prediction

**Files:**
- Create: `experiments/exp239_248_diverse_score_sources/registry.py`
- Create: `experiments/exp239_248_diverse_score_sources/cache_pipeline.py`
- Create: `experiments/exp239_248_diverse_score_sources/prediction.py`
- Create: `experiments/exp239_248_diverse_score_sources/run_pipeline.py`
- Create: `experiments/exp239_248_diverse_score_sources/tests/test_pipeline.py`

**Interfaces:**
- Produces CLI stages `audit`, `cache`, `predict`, `evaluate`, `report`, and `all`.
- Produces one cache manifest and one frozen prediction JSONL per source.
- Produces heartbeat fields `experiment_id`, `source`, `done`, `expected`, `errors`, `phase`, `current_dataset`, and `updated_unix`.

- [ ] **Step 1: Write failing registry and prediction-boundary tests**

```python
def test_registry_has_exact_experiment_order(self):
    self.assertEqual(tuple(SOURCE_REGISTRY), tuple(range(239, 249)))

def test_prediction_rows_have_no_evaluation_fields(self):
    forbidden = {"y_test", "labels", "tp", "fp", "f1", "actual_anomaly_positions"}
    self.assertFalse(forbidden & set(build_prediction_row(fake_result, baseline, 0.005)))

def test_exp248_requires_frozen_exp247_embedding_hash(self):
    with self.assertRaisesRegex(ValueError, "embedding hash mismatch"):
        load_exp247_embeddings(tampered_path, expected_sha256="deadbeef")

def test_prediction_code_cannot_read_labels(self):
    source = inspect.getsource(predict_source).lower()
    self.assertNotIn("load_test_labels", source)
    self.assertNotIn("y_test", inspect.signature(predict_source).parameters)

def test_dataset_rename_does_not_change_scores(self):
    renamed = dataclasses.replace(bundle, dataset="renamed_lookup_key")
    original = ComplexitySignatureDeviation().score(bundle, 20260720, temp_a)
    replay = ComplexitySignatureDeviation().score(renamed, 20260720, temp_b)
    np.testing.assert_allclose(original.test_scores, replay.test_scores)
```

- [ ] **Step 2: Run pipeline tests RED**

Run: `python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_pipeline`

- [ ] **Step 3: Implement registry and dependency ordering**

Map 239..248 to the exact operational names. Enforce Exp247 cache completion and SHA verification before Exp248. Reject unknown source names and duplicate experiment IDs.

- [ ] **Step 4: Implement resumable hashed cache**

Write one NPZ and JSON sidecar per dataset/source atomically. A cache hit is valid only when source config hash, DB hash, input metadata, seed, score lengths, finite values, and file hash agree. Retry only missing or invalid units.

- [ ] **Step 5: Implement frozen prediction**

Read score caches, compute empirical candidates for all frozen alphas, attach frozen Exp236 lane intersections and outside-lane indices, and write canonical sorted JSONL. Hash the complete byte stream before evaluation.

- [ ] **Step 6: Implement CLI and heartbeat**

Support:

```bash
python3 -m experiments.exp239_248_diverse_score_sources.run_pipeline \
  --stage cache --source multiscale_wavelet_energy_deviation \
  --workers 7 --output-root /Users/minho/Documents/Dataset/exp239_248_diverse_score_sources
```

Model sources reject `--workers > 1`. Every status update is atomic and dashboard-visible.

- [ ] **Step 7: Run tests and commit**

```bash
python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_pipeline
git add experiments/exp239_248_diverse_score_sources
git commit -m "feat: add Exp239-248 resumable prediction pipeline"
```

---

### Task 6: Evaluation, Qualification, And Independent Reaggregation

**Files:**
- Create: `experiments/exp239_248_diverse_score_sources/evaluation.py`
- Create: `experiments/exp239_248_diverse_score_sources/reporting.py`
- Create: `experiments/exp239_248_diverse_score_sources/independent_reaggregate.py`
- Create: `experiments/exp239_248_diverse_score_sources/tests/test_reporting.py`

**Interfaces:**
- Consumes only hash-verified frozen predictions and evaluation labels.
- Produces standalone, Hard-confirmation, Standard-priority, outside-lane, TRAIN calibration, correlation, and research-gate tables.

- [ ] **Step 1: Write failing metric-scope and reaggregation tests**

```python
def test_outside_lane_excludes_every_frozen_lane(self):
    outside = outside_lane_indices({1, 2, 3, 4}, hard={1}, standard={2}, priority={3})
    self.assertEqual(outside, frozenset({4}))

def test_human_assisted_metrics_are_not_autonomous(self):
    row = summarize_source(fake_rows)
    self.assertEqual(row["hard_policy_changed"], 0)
    self.assertEqual(row["combined_metric_scope"], "human_assisted_diagnostic_only")

def test_evaluation_rejects_prediction_hash_tampering(self):
    with self.assertRaisesRegex(ValueError, "prediction hash mismatch"):
        evaluate_predictions(tampered_prediction, manifest, label_loader)
```

- [ ] **Step 2: Run reporting tests RED**

Run: `python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_reporting`

- [ ] **Step 3: Implement evaluation after hash verification**

Evaluation must verify prediction SHA256 first, then load TEST labels in a separate function. Calculate exact candidate, TP, FP, precision, recall, and mean dataset F1 for each of the four views without changing prediction rows.

- [ ] **Step 4: Implement TRAIN OOF qualification and correlation**

Report OOF alert rates, attainable alpha, resolution blocking, abstention, seed stability, and Spearman correlation with Exp237-238 source families. Cluster only for diagnosis; do not let TEST metrics define evidence families.

- [ ] **Step 5: Implement preregistered gate reporting**

Emit boolean columns for the confirmation, review-priority, and 80% coverage-expansion gates exactly as written in the design. Label every gate `retrospective_research_gate_only`.

- [ ] **Step 6: Implement independent reaggregation**

Read frozen predictions independently, reload labels, recompute every aggregate, verify disjoint lane views and index range, and compare every integer count against summary CSVs. Exit nonzero on any mismatch.

- [ ] **Step 7: Run tests and commit**

```bash
python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_reporting
git add experiments/exp239_248_diverse_score_sources
git commit -m "feat: add Exp239-248 qualification reporting"
```

---

### Task 7: Dashboard Registration And Sequential Queue

**Files:**
- Modify: `serve_rank_dashboard.py`
- Create: `scratch/run_exp239_248_sequential.py`
- Create: `experiments/exp239_248_diverse_score_sources/tests/test_dashboard_queue.py`

**Interfaces:**
- Queue executes Exp239 through Exp248 one at a time.
- Dashboard history shows operational source names, not only experiment numbers.

- [ ] **Step 1: Write failing dashboard and queue tests**

```python
def test_queue_order_is_239_through_248(self):
    self.assertEqual([item.experiment for item in build_queue()], list(range(239, 249)))

def test_dashboard_catalog_has_operational_names(self):
    for experiment in range(239, 249):
        self.assertIn(experiment, EXPERIMENT_CATALOG)
        self.assertNotRegex(EXPERIMENT_CATALOG[experiment]["label"], r"^Exp\d+$")
```

- [ ] **Step 2: Run tests RED**

Run: `python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_dashboard_queue`

- [ ] **Step 3: Register dashboard entries**

Add Exp239-248 labels, summary paths, and `comparison_eligible=False`. Preserve every existing dashboard entry and UI behavior.

- [ ] **Step 4: Implement sequential queue launcher**

The launcher writes a durable queue JSON, skips only a hash-verified completed source, invokes lightweight sources with seven workers and model sources with one worker, and stops on unrecoverable coverage or integrity errors.

- [ ] **Step 5: Run dashboard tests and commit**

```bash
python3 -m unittest experiments.exp239_248_diverse_score_sources.tests.test_dashboard_queue
git add serve_rank_dashboard.py scratch/run_exp239_248_sequential.py experiments/exp239_248_diverse_score_sources/tests/test_dashboard_queue.py
git commit -m "feat: queue Exp239-248 score qualification"
```

---

### Task 8: Smoke Validation And Full Sequential Execution

**Files:**
- Create: `outputs/exp239_248_diverse_score_sources/SMOKE_REPORT.md`
- Create: `outputs/exp239_248_diverse_score_sources/RESULT_REPORT.md`
- Create: `outputs/exp239_248_diverse_score_sources/MANIFEST.md`

- [ ] **Step 1: Run full unit and structure tests**

```bash
python3 -m py_compile experiments/exp239_248_diverse_score_sources/*.py
python3 -m unittest discover -s experiments/exp239_248_diverse_score_sources/tests -p 'test_*.py'
```

Expected: all tests pass before any smoke or full evaluation.

- [ ] **Step 2: Run 10-dataset smoke for all sources**

Use a separate output root and `--dataset-limit 10`. Verify every prediction file contains no labels or evaluation metrics, lane intersections are disjoint, score lengths match TEST rows, and source errors are zero. Exp248 must verify the Exp247 smoke embedding hash.

- [ ] **Step 3: Run MPS parity benchmark**

Record CPU/MPS elapsed time and candidate-index equality for Exp246-248. Freeze the full-run device decision before reading smoke TEST metrics.

- [ ] **Step 4: Start the full sequential queue**

```bash
screen -dmS exp239_248_queue bash -lc '
  cd /tmp/timeseries-exp239-248-score-sources &&
  python3 scratch/run_exp239_248_sequential.py \
    --output-root /Users/minho/Documents/Dataset/exp239_248_diverse_score_sources \
    2>&1 | tee /Users/minho/Documents/Dataset/exp239_248_diverse_score_sources/full_queue.log
'
```

Expected: one active source at a time, dashboard progress visible, queue continues in numerical order.

- [ ] **Step 5: Verify full prediction coverage before evaluation**

For each source require 1,117 dataset rows per alpha/variant contract, zero silent omissions, explicit abstentions, zero cache errors, and a frozen prediction SHA256. Do not evaluate an incomplete source.

- [ ] **Step 6: Evaluate only frozen predictions**

Run evaluation and reporting stages source by source without modifying prediction files. No result may trigger a config or alpha change.

- [ ] **Step 7: Run independent reaggregation**

Require zero summary mismatches, zero lane overlap, zero out-of-range indices, and exact frozen Exp236 lane preservation.

- [ ] **Step 8: Write result report and manifest**

Report all ten sources, all frozen alphas, confirmation/review/outside-lane views, abstention, resolution, OOF calibration, source correlation, execution device, hashes, and limitations. Explicitly state that no source was selected for deployment.

- [ ] **Step 9: Commit reproducibility artifacts**

```bash
git add experiments/exp239_248_diverse_score_sources serve_rank_dashboard.py scratch/run_exp239_248_sequential.py outputs/exp239_248_diverse_score_sources
git commit -m "results: record Exp239-248 score qualification"
git push origin experiment/exp239-248-score-sources
```

---

## Final Verification Checklist

- [ ] 1,117-dataset population is unchanged.
- [ ] Ten source names and experiment IDs are unique and frozen.
- [ ] Every source has complete rows or explicit abstention reasons.
- [ ] Prediction contains no TEST labels or evaluation metrics.
- [ ] Every prediction hash is verified before evaluation.
- [ ] Exp236 Hard, Standard review, and Priority review are unchanged.
- [ ] Alpha, seed, source, and hyperparameters were not selected from TEST results.
- [ ] TRAIN OOF calibration, resolution, and source correlation are reported.
- [ ] Independent reaggregation has zero mismatches.
- [ ] Dashboard displays current source and queue order.
- [ ] Results are labelled retrospective and not deployment validated.
