# Exp393-396 Normal Process Grammar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a TRAIN-normal-only Normal Process Grammar Score that recovers missed anomalies into the review lane without changing existing autonomous Hard alerts.

**Architecture:** A focused `experiments/exp393_396` package separates deterministic preprocessing and grammar scoring from cross-fit calibration, frozen prediction artifacts, label-based evaluation, and review-only integration. Exp393 writes label-free scores and candidates, Exp394 audits TRAIN pseudo-normal calibration, Exp395 evaluates complementarity only after prediction hash verification, and Exp396 applies the frozen score only to review ranking while asserting that existing Hard indices are unchanged.

**Tech Stack:** Python 3, NumPy, SciPy, scikit-learn `MiniBatchKMeans`, SQLite read-only access, `unittest`, CSV/JSONL/SHA256 artifacts.

## Global Constraints

- Database path remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Output root remains `/Users/minho/Documents/Dataset/exp393_396_normal_process_grammar`.
- Prediction, feature fitting, state fitting, calibration, thresholding, and candidate selection must not read TEST labels or anomaly positions.
- Dataset and family names are lookup keys only and must not change score, threshold, source, or policy.
- TEST length must not set a top-N budget.
- `ALPHAS = (0.005, 0.01, 0.02, 0.05)`.
- `STATE_COUNT = 8`, `RANDOM_SEED = 20260727`, `N_INIT = 10`, `MAX_ITER = 200`.
- `n_train < 5` is calibration abstention.
- Exp393-396 must not modify existing Hard alert indices.
- Priority review remains review-only.
- All TEST-label metrics are retrospective counterfactual diagnostics.
- No TEST result may select alpha, descriptor, state count, threshold, or score weight.

---

## File Map

- `experiments/exp393_396/__init__.py`: package marker and public experiment IDs.
- `experiments/exp393_396/contracts.py`: frozen constants, dataclasses, canonical hashing, and manifest guards.
- `experiments/exp393_396/preprocessing.py`: TRAIN-derived sanitation, alignment, normalization, segmentation, and descriptors.
- `experiments/exp393_396/grammar_core.py`: state fitting, transition/dwell/recovery scoring, cross-fit calibration, and conformal p-values.
- `experiments/exp393_396/artifact_io.py`: read-only DB access, JSONL/CSV writing, SHA256 verification, and baseline lane loading.
- `experiments/exp393_396/reporting.py`: aggregate tables and Markdown reports.
- `experiments/exp393_396/run_experiment_393_396_pipeline.py`: `predict`, `audit`, `evaluate`, `integrate`, and `all` CLI stages.
- `experiments/exp393_396/independent_reaggregate_exp393_396.py`: read-only independent recomputation.
- `experiments/exp393_396/test_preprocessing.py`: preprocessing and descriptor tests.
- `experiments/exp393_396/test_grammar_core.py`: grammar and conformal tests.
- `experiments/exp393_396/test_structure.py`: forbidden-input and source-structure tests.
- `experiments/exp393_396/test_pipeline.py`: artifact hash, lane, and smoke integration tests.
- `docs/EXP393_396_PREREGISTRATION_KO.md`: concise preregistration and interpretation boundary.
- `docs/EXP393_396_RUNBOOK_KO.md`: exact smoke/full commands and recovery procedure.
- `run_rank_experiments_sequential.py`: one queue entry for the sequential pipeline.

---

### Task 1: Frozen Contracts and Deterministic Preprocessing

**Files:**
- Create: `experiments/exp393_396/__init__.py`
- Create: `experiments/exp393_396/contracts.py`
- Create: `experiments/exp393_396/preprocessing.py`
- Create: `experiments/exp393_396/test_preprocessing.py`

**Interfaces:**
- Produces: `GrammarConfig`, `PreparedDataset`, `sanitize_series`, `derive_target_length`, `prepare_dataset`, `segment_descriptors`.
- Consumes: raw TRAIN-normal and TEST trace arrays loaded later by `artifact_io.py`.

- [ ] **Step 1: Write failing preprocessing tests**

```python
class PreprocessingTests(unittest.TestCase):
    def test_train_only_target_length_and_finite_output(self):
        train = [np.array([0.0, np.nan, 2.0]), np.array([0.0, 1.0, 2.0, 3.0])]
        test = [np.array([100.0] * 20)]
        prepared = prepare_dataset(train, test)
        self.assertEqual(prepared.target_length, 4)
        self.assertTrue(np.isfinite(prepared.train).all())
        self.assertEqual(prepared.test.shape[1], 4)

    def test_descriptor_shape_is_deterministic(self):
        trace = np.linspace(0.0, 1.0, 64)
        first = segment_descriptors(trace, target_length=64)
        second = segment_descriptors(trace, target_length=64)
        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape[1], 6)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python3 -m unittest experiments.exp393_396.test_preprocessing -v
```

Expected: import failure because the package and functions do not exist.

- [ ] **Step 3: Implement frozen contracts and preprocessing**

Use these exact public contracts:

```python
@dataclass(frozen=True)
class GrammarConfig:
    alphas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
    state_count: int = 8
    random_seed: int = 20260727
    n_init: int = 10
    max_iter: int = 200
    high_support_rate: float = 0.05
    transition_tail_rate: float = 0.95
    component_tail_rate: float = 0.95

@dataclass(frozen=True)
class PreparedDataset:
    train: np.ndarray
    test: np.ndarray
    target_length: int
    train_center: np.ndarray
    train_scale: np.ndarray
```

`sanitize_series` must interpolate finite internal gaps, extend nearest finite
values to boundary gaps, and return zeros only when no finite value exists.
`derive_target_length` must use TRAIN lengths only. `prepare_dataset` must run
sanitation, alignment, then TRAIN-derived robust normalization in that order.
`segment_descriptors` must emit robust level, slope, curvature energy, MAD,
lag-1 autocorrelation, and spectral entropy.

- [ ] **Step 4: Run preprocessing tests**

Run:

```bash
python3 -m unittest experiments.exp393_396.test_preprocessing -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add experiments/exp393_396
git commit -m "feat: add normal process grammar preprocessing"
```

---

### Task 2: Grammar State Model and Component Scores

**Files:**
- Create: `experiments/exp393_396/grammar_core.py`
- Create: `experiments/exp393_396/test_grammar_core.py`

**Interfaces:**
- Consumes: descriptor matrices from `segment_descriptors`.
- Produces: `fit_discrete_grammar`, `score_state_path`, `fit_grammar_model`,
  `score_trace`, `crossfit_train_scores`, `conformal_p_value`,
  `candidate_indices`.

- [ ] **Step 1: Write failing grammar tests**

```python
class GrammarScoreTests(unittest.TestCase):
    def test_unseen_transition_scores_above_normal_path(self):
        train_paths = [[0, 1, 2, 0], [0, 1, 2, 0], [0, 1, 2, 0]]
        model = fit_discrete_grammar(train_paths, state_count=3)
        normal = score_state_path(model, [0, 1, 2, 0])
        abnormal = score_state_path(model, [0, 2, 1, 0])
        self.assertGreater(abnormal.transition, normal.transition)

    def test_conformal_p_value_uses_finite_sample_correction(self):
        self.assertEqual(conformal_p_value([1.0, 2.0, 3.0], 4.0), 0.25)

    def test_crossfit_excludes_held_out_trace(self):
        calls = []
        crossfit_train_scores(
            traces=[np.arange(32.0) + i for i in range(5)],
            fit_observer=lambda indices: calls.append(tuple(indices)),
        )
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(len(indices) == 4 for indices in calls))
```

- [ ] **Step 2: Run grammar tests and verify failure**

Run:

```bash
python3 -m unittest experiments.exp393_396.test_grammar_core -v
```

Expected: import failure for missing grammar functions.

- [ ] **Step 3: Implement the grammar model**

Use focused dataclasses:

```python
@dataclass(frozen=True)
class GrammarModel:
    clusterer: MiniBatchKMeans
    transition_probability: np.ndarray
    dwell_lengths: dict[int, np.ndarray]
    high_support_states: frozenset[int]
    transition_surprise_q95: float

@dataclass(frozen=True)
class ComponentScore:
    transition: float
    dwell: float
    recovery: float
```

Fit `MiniBatchKMeans` on TRAIN descriptors only. Use additive Dirichlet
`alpha=1.0` for the transition matrix. Compute transition and dwell top-5%
means with at least one element. Recovery is the maximum number of segments
from a transition above TRAIN q95 surprise until a state with TRAIN occupancy
at least 5% is reached; add one when the trace ends without returning.

- [ ] **Step 4: Implement cross-fit percentiles and candidate generation**

```python
def conformal_p_value(reference: Sequence[float], value: float) -> float:
    ref = np.asarray(reference, dtype=float)
    return float((1 + np.count_nonzero(ref >= value)) / (ref.size + 1))

def candidate_indices(p_values: Sequence[float], alpha: float) -> list[int]:
    return [index for index, value in enumerate(p_values) if value <= alpha]
```

For each component, convert TEST scores to conformal p-values from its
cross-fitted TRAIN-normal reference. The composite is the equal-weight mean of
the three TRAIN OOF percentile scores, followed by its own conformal p-value.

- [ ] **Step 5: Run grammar tests**

Run:

```bash
python3 -m unittest experiments.exp393_396.test_grammar_core -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add experiments/exp393_396
git commit -m "feat: implement normal process grammar scores"
```

---

### Task 3: Label-Free DB Input and Frozen Exp393 Prediction

**Files:**
- Create: `experiments/exp393_396/artifact_io.py`
- Create: `experiments/exp393_396/run_experiment_393_396_pipeline.py`
- Create: `experiments/exp393_396/test_structure.py`
- Create: `experiments/exp393_396/test_pipeline.py`

**Interfaces:**
- Consumes: `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Produces: `exp393_predictions.jsonl`, `exp393_prediction_manifest.json`, status JSON, and checkpoint JSONL.

- [ ] **Step 1: Write structural tests**

```python
class StructureTests(unittest.TestCase):
    def test_predict_dataset_has_no_y_test_parameter(self):
        params = inspect.signature(predict_dataset).parameters
        self.assertNotIn("y_test", params)
        self.assertNotIn("labels", params)

    def test_prediction_row_has_no_metrics(self):
        row = build_prediction_row(fake_prediction())
        forbidden = {"y_test", "hard_tp", "hard_fp", "f1", "combined_f1"}
        self.assertTrue(forbidden.isdisjoint(row))
```

Add AST-based checks that the predict stage does not SELECT the label column
and does not import the evaluation loader.

- [ ] **Step 2: Run structural tests and verify failure**

Run:

```bash
python3 -m unittest experiments.exp393_396.test_structure -v
```

Expected: missing pipeline and artifact functions.

- [ ] **Step 3: Implement read-only trace loading**

`load_prediction_inputs()` must open SQLite with:

```python
uri = f"file:{DB_PATH}?mode=ro"
connection = sqlite3.connect(uri, uri=True)
```

Prediction queries may read dataset identifiers, split, and `values_blob`.
They must not SELECT labels. Dataset names are sorted before processing.

- [ ] **Step 4: Implement Exp393 predict phase**

Required signature:

```python
def predict_dataset(
    x_train_normal: Sequence[np.ndarray],
    x_test: Sequence[np.ndarray],
    frozen_config: GrammarConfig,
) -> dict[str, object]:
    ...
```

Each row stores component scores, component p-values, composite scores,
alpha-index maps, target length, effective state count, abstention reason,
runtime, and contract flags. It stores no label or evaluation metric.

Manifest flags must include:

```python
{
    "contains_test_labels": False,
    "contains_evaluation_metrics": False,
    "uses_family_name": False,
    "uses_dataset_name_for_policy": False,
    "uses_test_length_for_budget": False,
    "test_metrics_select_policy": False,
    "eligible_for_deployment": False,
}
```

- [ ] **Step 5: Implement restartable checkpoints and hash verification**

Write each completed dataset atomically to checkpoint JSONL. On resume, verify
config hash and reject duplicate dataset rows. Finalize only when expected
coverage is complete or explicitly marked abstained. Compute SHA256 after
sorting rows by dataset.

- [ ] **Step 6: Run structure and pipeline tests**

Run:

```bash
python3 -m unittest \
  experiments.exp393_396.test_structure \
  experiments.exp393_396.test_pipeline -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add experiments/exp393_396
git commit -m "feat: add frozen Exp393 grammar predictions"
```

---

### Task 4: Exp394 TRAIN-Normal Calibration Audit

**Files:**
- Modify: `experiments/exp393_396/run_experiment_393_396_pipeline.py`
- Create: `experiments/exp393_396/reporting.py`
- Modify: `experiments/exp393_396/test_pipeline.py`

**Interfaces:**
- Consumes: verified Exp393 prediction and embedded TRAIN OOF references.
- Produces: `exp394_calibration_dataset.csv`, `exp394_alpha_summary.csv`, and `exp394_calibration_report.md`.

- [ ] **Step 1: Add failing calibration audit tests**

```python
def test_audit_rejects_modified_prediction(tmp_path):
    prediction, manifest = write_valid_prediction(tmp_path)
    prediction.write_text(prediction.read_text() + "{}\n")
    with self.assertRaisesRegex(ValueError, "SHA256"):
        run_audit(tmp_path)

def test_audit_reports_abstention_and_resolution():
    row = audit_dataset(n_train=4, oof_p_values=[])
    self.assertTrue(row["abstained"])
    self.assertEqual(row["abstention_reason"], "n_train_lt_5")
```

- [ ] **Step 2: Run the targeted test and verify failure**

Run:

```bash
python3 -m unittest experiments.exp393_396.test_pipeline -v
```

Expected: missing audit functions.

- [ ] **Step 3: Implement Exp394 audit**

For each component and alpha report TRAIN pseudo-normal alerts, rate per 100
runs, conformal resolution, one-sided 95% Wilson upper bound, effective state
count, and abstention. Compute pairwise TRAIN OOF Spearman correlations against
available existing evidence score vectors without using TEST labels.

- [ ] **Step 4: Run tests and commit Task 4**

```bash
python3 -m unittest experiments.exp393_396.test_pipeline -v
git add experiments/exp393_396
git commit -m "feat: audit Exp394 train normal calibration"
```

---

### Task 5: Exp395 Locked Complementarity Evaluation

**Files:**
- Modify: `experiments/exp393_396/artifact_io.py`
- Modify: `experiments/exp393_396/run_experiment_393_396_pipeline.py`
- Modify: `experiments/exp393_396/reporting.py`
- Modify: `experiments/exp393_396/test_pipeline.py`

**Interfaces:**
- Consumes: hash-verified Exp393 prediction, Exp394 audit, frozen baseline lane files, then TEST labels.
- Produces: `exp395_dataset_results.csv`, `exp395_candidate_diff.csv`, `exp395_summary.csv`, and `exp395_report.md`.

- [ ] **Step 1: Write failing phase-order tests**

```python
def test_evaluate_verifies_prediction_before_loading_labels(tmp_path):
    order = []
    with mock.patch.object(module, "verify_prediction", side_effect=lambda *_: order.append("verify")):
        with mock.patch.object(module, "load_test_labels", side_effect=lambda *_: order.append("labels")):
            evaluate(tmp_path)
    self.assertLess(order.index("verify"), order.index("labels"))

def test_evaluation_does_not_change_hard_indices():
    result = integrate_review(
        baseline_hard={1, 4},
        baseline_standard=set(),
        baseline_priority=set(),
        npgs_candidates={2, 4},
    )
    self.assertEqual(result.hard, {1, 4})
```

- [ ] **Step 2: Run the targeted tests and verify failure**

Run:

```bash
python3 -m unittest experiments.exp393_396.test_pipeline -v
```

- [ ] **Step 3: Implement locked label evaluation**

Load labels only after Exp393 SHA256, config SHA256, dataset-list SHA256, and
Exp394 contract verification. For every component and alpha calculate
candidate count, recovered baseline FN, added FP, review overlap, review-outside
precision, contributing datasets/lineages, exact candidate signature, and
leave-one-lineage-out stability. Keep autonomous metrics separate from review
diagnostics.

- [ ] **Step 4: Run tests and commit Task 5**

```bash
python3 -m unittest experiments.exp393_396.test_pipeline -v
git add experiments/exp393_396
git commit -m "feat: evaluate Exp395 grammar complementarity"
```

---

### Task 6: Exp396 Review-Only Integration and Independent Reaggregation

**Files:**
- Modify: `experiments/exp393_396/run_experiment_393_396_pipeline.py`
- Create: `experiments/exp393_396/independent_reaggregate_exp393_396.py`
- Modify: `experiments/exp393_396/reporting.py`
- Modify: `experiments/exp393_396/test_pipeline.py`

**Interfaces:**
- Consumes: frozen Exp395 candidate tables and baseline lanes.
- Produces: `exp396_review_integration.csv`, `exp396_lane_transition.csv`, `exp396_summary.csv`, `exp396_report.md`, and `independent_consistency_report.md`.

- [ ] **Step 1: Write failing review-only integration tests**

```python
def test_new_candidates_enter_standard_only():
    result = integrate_review(
        baseline_hard={0},
        baseline_standard={1},
        baseline_priority={2},
        npgs_candidates={3},
    )
    self.assertEqual(result.hard, {0})
    self.assertEqual(result.priority, {2})
    self.assertEqual(result.standard, {1, 3})

def test_lane_overlap_is_zero():
    result = integrate_review({0}, {1}, {2}, {0, 1, 3})
    self.assertFalse(result.hard & result.standard)
    self.assertFalse(result.hard & result.priority)
    self.assertFalse(result.standard & result.priority)
```

- [ ] **Step 2: Implement two review-only variants**

Variant `rank_existing_review` changes only ordering inside Standard review.
Variant `add_to_standard_review` adds NPGS candidates outside every existing
lane to Standard review. Report both for all frozen alphas without selecting a
winner.

- [ ] **Step 3: Implement independent reaggregation**

The reaggregator must read final CSVs only, recompute aggregate counts, assert
candidate equals the disjoint lane union, assert index bounds, and compare its
SHA256-tagged summary with the pipeline manifest.

- [ ] **Step 4: Run tests and commit Task 6**

```bash
python3 -m unittest experiments.exp393_396.test_pipeline -v
git add experiments/exp393_396
git commit -m "feat: integrate Exp396 review only grammar evidence"
```

---

### Task 7: Documentation, Queue Registration, and End-to-End Verification

**Files:**
- Create: `docs/EXP393_396_PREREGISTRATION_KO.md`
- Create: `docs/EXP393_396_RUNBOOK_KO.md`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `experiments/exp393_396/test_structure.py`

**Interfaces:**
- Consumes: complete Exp393-396 pipeline.
- Produces: queue-visible sequential experiment and reproducible smoke/full commands.

- [ ] **Step 1: Add a failing queue registration test**

```python
def test_exp393_396_is_registered_once():
    text = Path("run_rank_experiments_sequential.py").read_text()
    self.assertEqual(text.count('"experiment_393_396_normal_process_grammar"'), 1)
```

- [ ] **Step 2: Register one sequential queue entry**

Add experiment ID `experiment_393_396_normal_process_grammar`. Its command runs
the pipeline with `--stage all --workers 7`. Progress, completion, error,
primary result, secondary result, and log paths must point to the shared output
root. Do not register four independently concurrent queue entries.

- [ ] **Step 3: Write preregistration and runbook**

The preregistration repeats the frozen constants and interpretation limits.
The runbook contains:

```bash
python3 -m unittest \
  experiments.exp393_396.test_preprocessing \
  experiments.exp393_396.test_grammar_core \
  experiments.exp393_396.test_structure \
  experiments.exp393_396.test_pipeline

python3 experiments/exp393_396/run_experiment_393_396_pipeline.py \
  --stage all --dataset-limit 10 --workers 2 \
  --output-dir /Users/minho/Documents/Dataset/exp393_396_normal_process_grammar_smoke

python3 experiments/exp393_396/run_experiment_393_396_pipeline.py \
  --stage all --workers 7 \
  --output-dir /Users/minho/Documents/Dataset/exp393_396_normal_process_grammar
```

- [ ] **Step 4: Run static and unit verification**

```bash
python3 -m py_compile experiments/exp393_396/*.py
python3 -m unittest \
  experiments.exp393_396.test_preprocessing \
  experiments.exp393_396.test_grammar_core \
  experiments.exp393_396.test_structure \
  experiments.exp393_396.test_pipeline
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [ ] **Step 5: Run 10-dataset smoke**

Run the smoke command from the runbook. Verify:

- prediction rows equal 10
- prediction contains no labels or evaluation metrics
- no duplicate or missing datasets
- candidate indices are in range
- lane overlap is zero
- baseline Hard indices are unchanged
- independent reaggregation matches

- [ ] **Step 6: Commit Task 7**

```bash
git add \
  docs/EXP393_396_PREREGISTRATION_KO.md \
  docs/EXP393_396_RUNBOOK_KO.md \
  run_rank_experiments_sequential.py \
  experiments/exp393_396
git commit -m "chore: register Exp393-396 grammar pipeline"
```

---

## Final Execution Gate

The full 1,117-dataset run starts only after all unit tests and the 10-dataset
smoke pass. Full execution remains one sequential queue item. The queue runner
may execute it after registration, but implementation verification must not
claim success merely because a status file says completed.

After the full run, verify:

```bash
python3 experiments/exp393_396/independent_reaggregate_exp393_396.py \
  --output-dir /Users/minho/Documents/Dataset/exp393_396_normal_process_grammar
```

Required completion evidence:

- 1,117 dataset rows or explicit abstention/error rows totaling 1,117
- zero silent omissions
- prediction and manifest hashes match
- no TEST metric in prediction JSONL
- Hard indices identical to frozen baseline
- autonomous and human-assisted metrics reported separately
- no alpha or variant declared optimal
- retrospective and non-deployment status recorded in every summary
