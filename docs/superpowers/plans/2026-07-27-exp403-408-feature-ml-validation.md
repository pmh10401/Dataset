# Exp403-408 TRAIN-only Feature ML Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the frozen Exp291 alert lanes while auditing existing feature stability, assigning TRAIN-only ML confidence to current Hard alerts, and evaluating two independent review-only FN recovery learners.

**Architecture:** A values-only bridge produces the existing preprocessing contract and frozen lane universe. Exp403 builds a deterministic TRAIN OOF feature contract; Exp404 scores current Hard alerts with a classical novelty committee; Exp405 evaluates a frozen demotion counterfactual; Exp406 and Exp407 independently create self-supervised and synthetic-anomaly Shadow candidates; Exp408 audits independence, consensus, workload, and retrospective outcomes after prediction hashes are frozen.

**Tech Stack:** Python 3.14, NumPy, SciPy, scikit-learn, PyTorch with optional deterministic MPS, SQLite read-only access, JSONL/CSV/SHA256, `unittest`, existing sequential queue and dashboard.

## Global Constraints

- Database path remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Output root is `/Users/minho/Documents/Dataset/exp403_408_feature_ml_validation`.
- Preserve exactly 1,117 datasets, 2,218 automatic Hard alerts, 806 Standard review candidates, and 31 Priority review candidates.
- `ALPHAS=(0.005, 0.01, 0.02, 0.05)` is frozen; TEST metrics never select one alpha.
- Prediction code never receives or queries TEST labels, anomaly positions, TP, FP, FN, precision, recall, or F1.
- TEST length, family name, dataset name, and historical family TEST performance never select features, models, thresholds, eligibility, candidates, or routes.
- Existing Hard, Standard, and Priority indices remain unchanged through Exp408.
- Exp405 demotion is retrospective counterfactual only; it never writes modified operational lanes.
- Exp406 and Exp407 candidates are review-only Shadow candidates.
- `n_train < 5` abstains from ML; `5 <= n_train < 20` allows robust distance only; `20 <= n_train < 50` allows the classical committee; `n_train >= 50` allows learned methods.
- No tail replacement, TEST-length top-N budget, Priority-to-Hard promotion, or TEST-guided synthetic anomaly selection.
- The preprocessing order is `missing_repair -> train_length_alignment -> normalization`.
- All new results are retrospective and not deployment or prospective validation claims.

---

## File Structure

Create one focused package:

```text
experiments/exp403_408/
├── __init__.py                 experiment IDs and frozen constants
├── contracts.py                dataclasses, schemas, forbidden-field validation
├── foundation.py               DB bridge, preprocessing, folds, frozen lane loading
├── feature_registry.py         adapters and physical feature-group identity
├── feature_stability.py        Exp403 OOF stability, redundancy, summaries
├── classical_committee.py      Exp404 Mahalanobis/IF/OCSVM/LOF scores
├── tri_view_ssl.py             Exp406 time/frequency/residual encoder
├── synthetic_anomaly.py        Exp407 perturbations and light classifiers
├── prediction.py               label-free Exp403/404/406/407 artifacts
├── evaluation.py               hash-gated Exp405/408 TEST evaluation
├── consensus.py                TRAIN-only source independence and Exp408 variants
├── artifact_io.py              atomic JSONL/NPZ/CSV/checkpoint/hash I/O
├── independent_reaggregate.py  read-only result reconstruction
├── reporting.py                Korean reports, manifest, claims
├── pipeline.py                 checkpointed sequential stages
└── tests/
    ├── test_contracts.py
    ├── test_foundation.py
    ├── test_feature_stability.py
    ├── test_classical_committee.py
    ├── test_tri_view_ssl.py
    ├── test_synthetic_anomaly.py
    ├── test_prediction_boundary.py
    ├── test_evaluation.py
    ├── test_consensus.py
    └── test_pipeline.py
```

Additional integration files:

```text
scratch/run_exp403_408_feature_ml_validation.py
scratch/test_exp403_408_queue.py
docs/EXP403_408_RUNBOOK_KO.md
run_rank_experiments_sequential.py
serve_rank_dashboard.py
```

---

### Task 1: Freeze Contracts, Eligibility, and Artifact Schemas

**Files:**
- Create: `experiments/exp403_408/__init__.py`
- Create: `experiments/exp403_408/contracts.py`
- Create: `experiments/exp403_408/tests/__init__.py`
- Create: `experiments/exp403_408/tests/test_contracts.py`

**Interfaces:**
- Produces: `ALPHAS`, `SEED`, `EXPECTED_DATASETS`, and frozen lane totals.
- Produces dataclasses: `TrainEligibility`, `DatasetInput`, `Fold`,
  `FeatureBlock`, `StabilityAudit`, `FeatureGroupSummary`, `CommitteeScores`,
  `FrozenConfig`, `PredictionRow`, `DemotionEvaluation`,
  `LearnedSourceScores`, and `ShadowSourcePrediction`.
- Produces: `validate_prediction_row(row: Mapping[str, object]) -> dict[str, object]`.
- Consumes: no experiment-local code.

- [ ] **Step 1: Write schema and eligibility failure tests**

```python
def test_train_eligibility_boundaries():
    assert train_eligibility(4).level == "abstain"
    assert train_eligibility(5).level == "robust_distance"
    assert train_eligibility(20).level == "classical_committee"
    assert train_eligibility(50).level == "learned_methods"

def test_prediction_rejects_test_metrics():
    row = valid_prediction_row()
    row["hard_fp"] = 1
    with self.assertRaisesRegex(ValueError, "forbidden"):
        validate_prediction_row(row)

def test_prediction_requires_all_four_alphas():
    row = valid_prediction_row()
    del row["scores_by_alpha"]["0.05"]
    with self.assertRaisesRegex(ValueError, "alpha"):
        validate_prediction_row(row)
```

- [ ] **Step 2: Run tests and confirm contract failures**

Run:

```bash
python3 -m unittest experiments.exp403_408.tests.test_contracts -v
```

Expected: import or missing-symbol failures.

- [ ] **Step 3: Implement frozen constants and strict schemas**

Use exact constants:

```python
SEED = 20260727
ALPHAS = (0.005, 0.01, 0.02, 0.05)
EXPECTED_DATASETS = 1117
EXPECTED_HARD = 2218
EXPECTED_STANDARD = 806
EXPECTED_PRIORITY = 31

def train_eligibility(n_train: int) -> TrainEligibility:
    if n_train < 5:
        return TrainEligibility("abstain", False, False, False)
    if n_train < 20:
        return TrainEligibility("robust_distance", True, False, False)
    if n_train < 50:
        return TrainEligibility("classical_committee", True, True, False)
    return TrainEligibility("learned_methods", True, True, True)
```

Recursively reject keys equal to or ending in TEST evaluation fields. Require
exact alpha-key coverage and finite numeric values.

- [ ] **Step 4: Run contract tests**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp403_408
git commit -m "feat: define Exp403-408 TRAIN-only contracts"
```

---

### Task 2: Build the Values-only Foundation and Frozen Lane Bridge

**Files:**
- Create: `experiments/exp403_408/foundation.py`
- Create: `experiments/exp403_408/artifact_io.py`
- Create: `experiments/exp403_408/tests/test_foundation.py`

**Interfaces:**
- Consumes: `load_values_bundle`, `align_bundle`, and preprocessing contract from `experiments.exp226_235_hard_alert_quality`.
- Consumes: frozen Exp291/Exp292 prediction manifests and hashes.
- Produces: `DatasetInput(dataset, train_normal, test_values, target_length, hard_indices, standard_indices, priority_indices)`.
- Produces: `crossfit_plan(n_train: int, seed: int, folds: int = 5)`.
- Produces: atomic JSONL/NPZ writers and SHA256 verification.

- [ ] **Step 1: Write bridge and leakage tests**

```python
def test_dataset_loader_preserves_preprocessing_order():
    bundle = load_dataset_input("toy", loader=fake_values_loader)
    self.assertEqual(bundle.preprocessing_order, (
        "missing_repair", "train_length_alignment", "normalization",
    ))

def test_crossfit_never_scores_query_from_its_fit_fold():
    for fit, query in crossfit_plan(20, seed=20260727):
        self.assertFalse(set(fit) & set(query))

def test_frozen_lane_totals_must_match():
    with self.assertRaisesRegex(ValueError, "Hard"):
        validate_frozen_universe(rows_with_one_hard_removed())
```

- [ ] **Step 2: Run tests to confirm missing implementation**

```bash
python3 -m unittest experiments.exp403_408.tests.test_foundation -v
```

Expected: missing module or symbol failures.

- [ ] **Step 3: Implement read-only loading and deterministic folds**

Open SQLite with URI `mode=ro`. Reuse the established preprocessing functions,
then z-normalize each aligned run. Load only value blobs in prediction paths.
Load frozen lane artifacts only after validating their manifests and SHA256.

Implement:

```python
def load_dataset_input(dataset: str, *, db_path: Path = DB_PATH) -> DatasetInput:
    ...

def list_datasets_small_to_large(db_path: Path = DB_PATH) -> list[str]:
    ...

def crossfit_plan(n_train: int, seed: int = SEED, folds: int = 5) -> tuple[Fold, ...]:
    ...
```

- [ ] **Step 4: Run foundation and existing preprocessing regression tests**

```bash
python3 -m unittest experiments.exp403_408.tests.test_foundation -v
python3 -m unittest experiments.exp393_396.test_preprocessing -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp403_408
git commit -m "feat: add Exp403-408 values-only foundation"
```

---

### Task 3: Register Existing Feature Families

**Files:**
- Create: `experiments/exp403_408/feature_registry.py`
- Create: `experiments/exp403_408/tests/test_feature_stability.py`

**Interfaces:**
- Consumes: existing frozen TRAIN OOF feature/score artifacts from ROCKET, normal-distance, EWMA/change-point, spectral/wavelet, shape/complexity, and process-grammar experiments.
- Produces: `FeatureBlock(name, group, train_oof, test_values, source_hash, available, reason)`.
- Produces: `load_feature_blocks(dataset: str, bundle: DatasetInput) -> tuple[FeatureBlock, ...]`.

- [ ] **Step 1: Write physical-group and source-coverage tests**

```python
def test_rocket_capacities_share_one_physical_group():
    blocks = registry_for_names(["rocket256", "rocket512", "minirocket1024"])
    self.assertEqual({block.group for block in blocks}, {"temporal_random_kernel"})

def test_missing_source_is_explicit_not_empty_candidates():
    block = missing_feature_block("wavelet")
    self.assertFalse(block.available)
    self.assertEqual(block.reason, "source_artifact_missing")
```

- [ ] **Step 2: Run the tests and confirm failure**

```bash
python3 -m unittest experiments.exp403_408.tests.test_feature_stability -v
```

- [ ] **Step 3: Implement adapters with immutable source identity**

Define exactly six groups:

```python
FEATURE_GROUPS = {
    "temporal_random_kernel",
    "normal_space_distance",
    "change_delta",
    "frequency_multiscale",
    "shape_complexity",
    "process_grammar",
}
```

Every adapter must verify source hash, dataset name, TRAIN row count, TEST row
count, seed where present, and absence of label-derived fields. An unavailable
source returns an unavailable `FeatureBlock`; it never returns an empty
candidate set.

- [ ] **Step 4: Run feature registry tests**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp403_408
git commit -m "feat: register Exp403 feature families"
```

---

### Task 4: Implement Exp403 TRAIN OOF Feature Stability

**Files:**
- Create: `experiments/exp403_408/feature_stability.py`
- Modify: `experiments/exp403_408/tests/test_feature_stability.py`

**Interfaces:**
- Consumes: `tuple[FeatureBlock, ...]`.
- Produces: `audit_feature_stability(blocks, folds) -> StabilityAudit`.
- Produces: `build_group_summary(audit) -> FeatureGroupSummary`.

- [ ] **Step 1: Add tests for pruning, correlation collapse, and determinism**

```python
def test_constant_and_nonfinite_features_are_excluded():
    audit = audit_feature_stability(blocks_with_constant_and_nan(), folds())
    self.assertEqual(audit.excluded["constant"], "near_zero_train_variance")
    self.assertEqual(audit.excluded["nan"], "non_finite_train_oof")

def test_correlated_features_collapse_by_train_oof_only():
    audit = audit_feature_stability(highly_correlated_blocks(), folds())
    self.assertEqual(len(audit.correlation_components), 1)
    self.assertEqual(audit.correlation_threshold, 0.90)

def test_row_order_does_not_change_group_summary():
    self.assertEqual(
        stable_hash(build_group_summary(audit_for(rows))),
        stable_hash(build_group_summary(audit_for(rows[::-1]))),
    )
```

- [ ] **Step 2: Run the tests and confirm failures**

Run the Task 3 test command.

- [ ] **Step 3: Implement robust stability calculations**

For each feature compute finite rate, median, MAD, robust variance, fold-center
drift, and bootstrap rank stability. Collapse absolute TRAIN OOF Spearman
components at `>= 0.90`. Select representatives by:

```python
key = (
    non_finite_rate,
    -bootstrap_rank_stability,
    -coverage,
    feature_name,
)
```

Build fixed group summaries from empirical p-values: minimum, median,
robust mean and maximum `-log10(p)`, exceedance count, top-two gap, and
stability.

- [ ] **Step 4: Run feature stability tests**

Expected: all tests pass with no RuntimeWarning.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp403_408
git commit -m "feat: audit Exp403 feature stability"
```

---

### Task 5: Implement Exp404 Classical Normality Committee

**Files:**
- Create: `experiments/exp403_408/classical_committee.py`
- Create: `experiments/exp403_408/tests/test_classical_committee.py`

**Interfaces:**
- Consumes: Exp403 group summaries and deterministic folds.
- Produces: `score_classical_committee(train_features, test_features, eligibility, seed) -> CommitteeScores`.
- Produces: alpha-specific confidence grades without changing lanes.

- [ ] **Step 1: Write model-boundary and cross-fit tests**

```python
def test_small_train_uses_only_robust_distance():
    result = score_classical_committee(train5(), test2(), train_eligibility(5), SEED)
    self.assertEqual(result.available_models, ("shrinkage_mahalanobis",))

def test_lof_scores_only_unseen_queries():
    spy = SpyLOF()
    score_classical_committee(train20(), test2(), train_eligibility(20), SEED, lof_factory=lambda: spy)
    self.assertFalse(spy.scored_fit_rows)

def test_committee_never_changes_hard_indices():
    result = grade_frozen_hard(frozen_hard={1, 4}, committee=normal_votes())
    self.assertEqual(result.hard_indices, {1, 4})
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
python3 -m unittest experiments.exp403_408.tests.test_classical_committee -v
```

- [ ] **Step 3: Implement four fixed models**

Use:

```python
LedoitWolf()
IsolationForest(n_estimators=256, max_samples="auto", contamination="auto",
                random_state=SEED, n_jobs=1)
OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
LocalOutlierFactor(n_neighbors=min(20, n_fit - 1), novelty=True,
                   contamination="auto", n_jobs=1)
```

The `nu=0.01` value is a frozen sensitivity constant, not a TEST-selected
anomaly rate. Score fold queries only from models fitted without those rows.
Convert every model score to an OOF empirical p-value. Grade each frozen Hard
index as high, medium, recheck, or abstain.

- [ ] **Step 4: Run committee tests**

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp403_408
git commit -m "feat: add Exp404 normality committee"
```

---

### Task 6: Freeze Label-free Prediction Artifacts

**Files:**
- Create: `experiments/exp403_408/prediction.py`
- Create: `experiments/exp403_408/tests/test_prediction_boundary.py`
- Modify: `experiments/exp403_408/artifact_io.py`

**Interfaces:**
- Consumes: Exp403 and Exp404 results.
- Produces: `exp403_feature_manifest.csv`, `exp404_hard_confidence.jsonl`, and prediction manifests.
- Produces: `predict_dataset(bundle: DatasetInput, config: FrozenConfig) -> PredictionRow`; no `y_test` parameter.

- [ ] **Step 1: Write forbidden-input and tamper tests**

```python
def test_predict_dataset_signature_has_no_y_test():
    self.assertNotIn("y_test", inspect.signature(predict_dataset).parameters)

def test_prediction_json_has_no_evaluation_fields():
    payload = dataclasses.asdict(predict_dataset(bundle(), config()))
    self.assertFalse(find_forbidden_fields(payload))

def test_modified_prediction_hash_stops_evaluation():
    write_prediction(valid_rows(), output)
    output.joinpath("exp404_hard_confidence.jsonl").write_text("modified")
    with self.assertRaisesRegex(ValueError, "SHA256"):
        verify_prediction(output)
```

- [ ] **Step 2: Run boundary tests and confirm failures**

```bash
python3 -m unittest experiments.exp403_408.tests.test_prediction_boundary -v
```

- [ ] **Step 3: Implement canonical atomic artifacts**

Write sorted rows by dataset and candidate index. Store config, input, feature,
prediction, code, and baseline hashes. Use temporary files plus `os.replace`.
Prediction output contains confidence grades, scores, eligibility, and
abstention reasons but no evaluation fields.

- [ ] **Step 4: Run boundary and contract tests**

```bash
python3 -m unittest \
  experiments.exp403_408.tests.test_contracts \
  experiments.exp403_408.tests.test_prediction_boundary -v
```

- [ ] **Step 5: Commit**

```bash
git add experiments/exp403_408
git commit -m "feat: freeze Exp403-404 predictions"
```

---

### Task 7: Implement Exp405 Demotion Counterfactual Evaluation

**Files:**
- Create: `experiments/exp403_408/evaluation.py`
- Create: `experiments/exp403_408/tests/test_evaluation.py`

**Interfaces:**
- Consumes: verified Exp404 prediction hash and a separate label loader.
- Produces: `evaluate_hard_confidence(prediction, labels) -> DemotionEvaluation`.
- Produces: point estimates, Wilson/bootstrap intervals, dataset/lineage concentration, and immutable lane parity.

- [ ] **Step 1: Write hash-before-label and success-gate tests**

```python
def test_labels_are_not_loaded_before_hash_verification():
    loader = LabelLoaderSpy()
    with self.assertRaises(ValueError):
        evaluate_exp405(tampered_prediction(), loader)
    self.assertEqual(loader.calls, 0)

def test_success_gate_requires_all_three_conditions():
    decision = demotion_gate(fp_fraction=.81, tp_retention=.995, train_stability=.991)
    self.assertTrue(decision.passes_research_gate)
    self.assertFalse(demotion_gate(.79, .995, .991).passes_research_gate)
    self.assertFalse(demotion_gate(.81, .989, .991).passes_research_gate)
    self.assertFalse(demotion_gate(.81, .995, .989).passes_research_gate)
```

- [ ] **Step 2: Run tests and confirm failures**

```bash
python3 -m unittest experiments.exp403_408.tests.test_evaluation -v
```

- [ ] **Step 3: Implement separated retrospective evaluation**

Verify prediction SHA256 first, then load labels. Calculate:

- recheck count, FP and TP
- recheck FP fraction
- baseline TP retention if recheck were demoted
- TRAIN OOF normal stability
- dataset and lineage concentration
- paired dataset bootstrap intervals with seed `20260727`

Always write `operational_policy_changed=false`.

- [ ] **Step 4: Run evaluation tests**

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp403_408
git commit -m "feat: evaluate Exp405 demotion counterfactual"
```

---

### Task 8: Implement Exp406 Tri-view Self-supervised Shadow

**Files:**
- Create: `experiments/exp403_408/tri_view_ssl.py`
- Create: `experiments/exp403_408/tests/test_tri_view_ssl.py`

**Interfaces:**
- Consumes: normalized TRAIN/TEST runs, Exp403 eligibility, frozen seed.
- Produces: `score_tri_view(train, test, config, device) -> LearnedSourceScores`.
- Uses: time, log-spectrum/wavelet, and prototype-residual views.

- [ ] **Step 1: Write shape, leakage, determinism, and device tests**

```python
def test_three_views_have_fixed_width_and_are_finite():
    views = build_tri_views(train8_length64())
    self.assertEqual(set(views), {"time", "frequency", "residual"})
    self.assertTrue(all(np.isfinite(value).all() for value in views.values()))

def test_n_train_49_abstains_from_learned_model():
    result = score_tri_view(train49(), test2(), config(), device="cpu")
    self.assertEqual(result.abstention_reason, "n_train_below_50")

def test_same_seed_produces_same_cpu_prediction_hash():
    self.assertEqual(run_hash(seed=SEED), run_hash(seed=SEED))
```

- [ ] **Step 2: Run tests and confirm failures**

```bash
python3 -m unittest experiments.exp403_408.tests.test_tri_view_ssl -v
```

- [ ] **Step 3: Implement a bounded encoder**

Use a compact encoder with fixed widths:

```python
class TriViewEncoder(torch.nn.Module):
    # each view: Linear(input_dim, 64) -> GELU -> Linear(64, 32)
    # fused: concatenate 3x32 -> Linear(96, 32)
```

Loss is the fixed sum:

```python
loss = masked_reconstruction + cross_view_consistency + 0.1 * prototype_compactness
```

Use TRAIN-only fit/validation/calibration splits, maximum 100 epochs, patience
10, batch size 32, and no TEST-based early stopping. Build OOF TRAIN scores and
empirical p-values. Candidate indices outside frozen Hard become
`self_supervised_shadow`.

- [ ] **Step 4: Verify CPU and optional MPS**

```bash
python3 -m unittest experiments.exp403_408.tests.test_tri_view_ssl -v
```

If MPS is available, run the same smoke twice. Enable MPS in the full run only
when candidate ranks match CPU and score differences stay within `1e-5`.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp403_408
git commit -m "feat: add Exp406 tri-view Shadow source"
```

---

### Task 9: Implement Exp407 Synthetic-anomaly Shadow

**Files:**
- Create: `experiments/exp403_408/synthetic_anomaly.py`
- Create: `experiments/exp403_408/tests/test_synthetic_anomaly.py`

**Interfaces:**
- Consumes: normal TRAIN runs and frozen perturbation config.
- Produces: `generate_synthetic_anomalies(train, config, seed)`.
- Produces: `score_synthetic_models(train, test, config) -> tuple[LearnedSourceScores, ...]`.

- [ ] **Step 1: Write perturbation and TEST-isolation tests**

```python
def test_every_frozen_perturbation_is_generated():
    generated = generate_synthetic_anomalies(train8(), config(), SEED)
    self.assertEqual(set(generated.by_type), {
        "spike", "drift", "level_shift", "flat_line",
        "amplitude_scale", "frequency_warp", "phase_order",
    })

def test_perturbation_scale_depends_only_on_train():
    first = generate_synthetic_anomalies(train8(), config(), SEED)
    second = generate_synthetic_anomalies(train8(), config(), SEED)
    self.assertEqual(first.sha256, second.sha256)
```

- [ ] **Step 2: Run tests and confirm failures**

```bash
python3 -m unittest experiments.exp403_408.tests.test_synthetic_anomaly -v
```

- [ ] **Step 3: Implement fixed perturbations and three classifiers**

Derive magnitudes from TRAIN median, MAD, and fixed length fractions. Generate
the same count per perturbation type. Fit and report all:

```python
LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                               max_leaf_nodes=15, random_state=SEED)
SmallMLP(input_dim, hidden_dim=64, output_dim=1)
```

Use TRAIN-only split and no TEST metric selection. Every classifier produces a
separate sensitivity artifact. Candidates outside frozen Hard become
`synthetic_anomaly_shadow`.

- [ ] **Step 4: Run synthetic source tests**

Expected: all tests pass and repeated hashes match.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp403_408
git commit -m "feat: add Exp407 synthetic anomaly Shadow"
```

---

### Task 10: Implement Exp408 Independence and Consensus

**Files:**
- Create: `experiments/exp403_408/consensus.py`
- Create: `experiments/exp403_408/tests/test_consensus.py`

**Interfaces:**
- Consumes: frozen Exp406/407 predictions, existing feature-group identity, and aligned TRAIN OOF scores.
- Produces: `build_independence_components(sources, threshold=.90, minimum_rows=20)`.
- Produces: five frozen consensus variants from the design.

- [ ] **Step 1: Write identity, correlation, and lane tests**

```python
def test_same_lineage_collapses_even_with_low_observed_correlation():
    components = build_independence_components(two_rocket_capacities())
    self.assertEqual(len(components), 1)

def test_high_train_oof_correlation_collapses_distinct_sources():
    components = build_independence_components(correlated_sources(.95))
    self.assertEqual(len(components), 1)

def test_unverified_independence_does_not_add_vote():
    components = build_independence_components(sources_with_19_rows())
    self.assertEqual(components[1].status, "independence_unverified")

def test_consensus_never_promotes_to_hard():
    result = build_consensus(frozen_lanes(), source_predictions())
    self.assertEqual(result.hard_indices, frozen_lanes().hard_indices)
```

- [ ] **Step 2: Run tests and confirm failures**

```bash
python3 -m unittest experiments.exp403_408.tests.test_consensus -v
```

- [ ] **Step 3: Implement frozen variants**

Create exactly:

```python
VARIANTS = (
    "tri_view_only",
    "synthetic_only",
    "tri_view_and_synthetic",
    "one_new_plus_existing_independent",
    "two_new_plus_existing_independent",
)
```

Collapse same lineage first, then TRAIN OOF correlation components. Record
unverified independence instead of treating it as independent.

- [ ] **Step 4: Run consensus tests**

Expected: all tests pass with Hard/Standard/Priority parity.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp403_408
git commit -m "feat: add Exp408 source consensus"
```

---

### Task 11: Build Checkpointed Pipeline, Queue, and Dashboard Status

**Files:**
- Create: `experiments/exp403_408/pipeline.py`
- Create: `experiments/exp403_408/tests/test_pipeline.py`
- Create: `scratch/run_exp403_408_feature_ml_validation.py`
- Create: `scratch/test_exp403_408_queue.py`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`

**Interfaces:**
- Produces CLI stages: `audit-features`, `classical-predict`, `classical-evaluate`, `ssl-predict`, `synthetic-predict`, `consensus-evaluate`, `report`, `all`.
- Produces atomic `status.json` with stage, dataset, completed, total, percent, ETA, throughput, errors, device, and checkpoint hash.
- Registers one sequential queue batch: `experiment_403_408_feature_ml_validation`.

- [ ] **Step 1: Write pipeline resume and queue tests**

```python
def test_resume_skips_completed_dataset_hashes():
    first = run_smoke(limit=3, stop_after=2)
    second = run_smoke(limit=3, resume=True)
    self.assertEqual(second.recomputed_datasets, 1)

def test_status_has_dashboard_fields():
    status = read_status()
    self.assertTrue({
        "stage", "current_dataset", "completed", "total",
        "progress_percent", "eta_seconds", "throughput_per_minute",
        "errors", "device", "checkpoint_sha256",
    } <= set(status))
```

- [ ] **Step 2: Run tests and confirm failures**

```bash
python3 -m unittest experiments.exp403_408.tests.test_pipeline -v
python3 scratch/test_exp403_408_queue.py
```

- [ ] **Step 3: Implement atomic per-dataset checkpoints**

Process datasets small-to-large, use seven outer workers for CPU-only classical
stages, and one learned-model process at a time for Exp406/407. Never run nested
model parallelism. On per-dataset failure, write the error and explicit
abstention; do not mark a missing result as an empty candidate set.

- [ ] **Step 4: Register queue and dashboard**

Add the batch command and result paths without changing old experiment entries.
Dashboard status must read `status.json` only and must not scan large JSONL
files on every refresh.

- [ ] **Step 5: Run pipeline, queue, and dashboard regression tests**

```bash
python3 -m unittest experiments.exp403_408.tests.test_pipeline -v
python3 scratch/test_exp403_408_queue.py
python3 scratch/test_rank_dashboard.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp403_408 scratch/run_exp403_408_feature_ml_validation.py \
  scratch/test_exp403_408_queue.py run_rank_experiments_sequential.py \
  serve_rank_dashboard.py
git commit -m "feat: queue Exp403-408 feature ML validation"
```

---

### Task 12: Independent Reaggregation, Reports, and Manifest

**Files:**
- Create: `experiments/exp403_408/independent_reaggregate.py`
- Create: `experiments/exp403_408/reporting.py`
- Create: `docs/EXP403_408_RUNBOOK_KO.md`
- Modify: `experiments/exp403_408/tests/test_evaluation.py`
- Modify: `experiments/exp403_408/tests/test_pipeline.py`

**Interfaces:**
- Consumes: frozen prediction/evaluation CSV and manifests only.
- Produces: official and independent metric comparison with mismatch count.
- Produces: Korean reports and hash-rich `MANIFEST.json`/`MANIFEST.md`.

- [ ] **Step 1: Add independent mismatch and claims tests**

```python
def test_independent_reaggregation_matches_official():
    report = independent_reaggregate(fixture_output())
    self.assertEqual(report.mismatch_count, 0)

def test_report_separates_autonomous_and_review_metrics():
    text = build_report(fixture_output())
    self.assertIn("자동 경보는 변경하지 않음", text)
    self.assertIn("human-assisted", text)
    self.assertNotIn("배포 검증 완료", text)
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
python3 -m unittest \
  experiments.exp403_408.tests.test_evaluation \
  experiments.exp403_408.tests.test_pipeline -v
```

- [ ] **Step 3: Implement independent aggregation**

Recalculate:

- Exp404 grade counts and frozen lane parity
- Exp405 recheck FP fraction, TP retention, TRAIN stability
- Exp406/407 candidate counts and overlaps
- Exp408 variant candidate/TP/FP/precision, FN recovery, Zero-F1 diagnostics
- TRAIN-size coverage, dataset/lineage concentration, workload

Fail audit when any integer differs or a float differs by more than `1e-12`.

- [ ] **Step 4: Write reports and manifest**

Include code SHA256, DB/input/prediction hashes, commands, Python/package
versions, seed, retrospective scope, autonomous/human-assisted scope, and known
limitations. Explicitly forbid claims that Exp405 changed production or that
Exp406-408 were prospectively validated.

- [ ] **Step 5: Run report tests**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp403_408 docs/EXP403_408_RUNBOOK_KO.md
git commit -m "docs: report Exp403-408 validation"
```

---

### Task 13: Smoke, Full Sequential Run, and Final Verification

**Files:**
- Generate: `/Users/minho/Documents/Dataset/exp403_408_feature_ml_validation_smoke/*`
- Generate: `/Users/minho/Documents/Dataset/exp403_408_feature_ml_validation/*`

**Interfaces:**
- Consumes: all completed code and frozen inputs.
- Produces: verified smoke/full artifacts without modifying previous outputs.

- [ ] **Step 1: Run complete unit and structure tests**

```bash
python3 -m py_compile experiments/exp403_408/*.py \
  scratch/run_exp403_408_feature_ml_validation.py
python3 -m unittest discover -s experiments/exp403_408/tests -p 'test_*.py' -v
python3 -m unittest discover -s experiments/exp400_402 -p 'test_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 2: Run a representative smoke**

```bash
python3 scratch/run_exp403_408_feature_ml_validation.py \
  --stage all \
  --dataset-limit 20 \
  --workers 2 \
  --device cpu \
  --reset-checkpoint \
  --output-dir \
  /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation_smoke
```

Verify:

- all four TRAIN eligibility levels are represented where available
- prediction evaluation fields = 0
- lane changes = 0
- candidate index errors = 0
- independent mismatch = 0

- [ ] **Step 3: Benchmark optional MPS**

Run Exp406 smoke twice on CPU and twice on MPS. Record runtime, prediction hash,
candidate rank match, and maximum score difference. Select MPS only when rank
matches and maximum score difference is `<= 1e-5`; this is a compute-path
decision, not a TEST metric decision.

- [ ] **Step 4: Run full stages sequentially**

```bash
python3 scratch/run_exp403_408_feature_ml_validation.py \
  --stage all \
  --workers 7 \
  --device auto \
  --reset-checkpoint \
  --output-dir \
  /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation
```

Run only one learned-model stage at a time. Do not start another experiment
batch concurrently.

- [ ] **Step 5: Verify full coverage and invariants**

Require:

- datasets = 1,117
- errors = 0 or explicit per-dataset abstentions with no missing row
- automatic Hard = 2,218 unchanged
- Standard = 806 unchanged
- Priority = 31 unchanged
- forbidden prediction fields = 0
- independent mismatch = 0
- repeat sorted prediction hash exact match

- [ ] **Step 6: Run final regression and diff checks**

```bash
python3 -m unittest discover -s experiments/exp403_408/tests -p 'test_*.py' -v
python3 -m unittest discover -s experiments/exp400_402 -p 'test_*.py' -v
python3 scratch/test_exp403_408_queue.py
git diff --check
git status --short --branch
```

- [ ] **Step 7: Commit final verification metadata**

```bash
git add docs/EXP403_408_RUNBOOK_KO.md
git commit -m "test: verify Exp403-408 full validation"
```

If the runbook has no post-run metadata change, do not create an empty commit.
