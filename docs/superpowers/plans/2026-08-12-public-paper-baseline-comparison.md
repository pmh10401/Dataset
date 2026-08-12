# Public Paper Baseline Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, label-gated comparison of the frozen Exp448 proposed method against six published one-class anomaly detectors on the same 1,117-dataset population.

**Architecture:** Add an isolated `experiments.paper_public_baselines` package with separate values-only scoring and post-freeze label evaluation processes. Each method emits deterministic per-dataset score shards, TRAIN-OOF conformal alerts, and a frozen manifest; evaluation verifies those bytes before opening TEST labels and produces full-population, common-support, and matched-alert-budget comparisons.

**Tech Stack:** Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0, PyTorch 2.13.0, SQLite, `unittest`, optional MLX metadata only unless all AGENTS.md gates pass.

## Global Constraints

- Work from reviewed Exp448 commit `2f5229ccd471bba907562b3131d293b46b0b3afc` on branch `codex/public-paper-baselines`.
- Use exactly `/Users/mac/Documents/Dataset/univariate_ts.db` for the official run and require SHA-256 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`.
- Use exactly 1,117 datasets and preserve each dataset's contiguous TEST `instance_index` order.
- Scoring code must never import the TEST-label reader or emit labels, TP, FP, FN, TN, precision, recall, F1, or ground-truth fields.
- TEST labels may be opened only after the complete score manifest and every shard digest are verified.
- Report alpha values `0.005`, `0.01`, and `0.02` together; never select one from TEST results.
- Treat Exp404/416, Exp442, and Exp448 only as the author's proposed-method lineage and ablations, never as external baselines.
- Apply the AGENTS.md MLX policy before every full score-source run. Do not rewrite official scikit-learn CPU estimators in MLX.
- Run no more than seven CPU dataset workers and only one MLX/MPS-heavy process at a time.
- Write large run artifacts only below `/Users/mac/Documents/Dataset/paper_public_baselines_v1`; commit code, tests, documents, hashes, and compact reports only.

---

## File Structure

```text
experiments/paper_public_baselines/
  __init__.py             package identity only
  config.py               frozen methods, alphas, seeds, paths, versions
  contracts.py            immutable score/evaluation value objects
  data.py                 values-only SQLite access and dataset population
  preprocessing.py        deterministic repair, alignment, normalization
  calibration.py          OOF planning, conformal p-values, matched top-k
  artifacts.py            canonical JSON, atomic shards, digest verification
  methods/
    __init__.py
    classical.py          k-NN, LOF, OCSVM, Isolation Forest score adapters
    deep_common.py        deterministic torch data and 1D encoder helpers
    deep_svdd.py          Deep SVDD training and scores
    drocc.py              DROCC objective and scores
  acceleration.py         MLX eligibility/parity/speed evidence
  runner.py               resumable scoring orchestration
  label_access.py         evaluation-only SQLite TEST-label capability
  evaluation.py           frozen-score verification and confusion metrics
  statistics.py           macro/micro summaries and paired bootstrap
  reporting.py            JSON/CSV/Markdown paper tables
  cli.py                  prepare, assess, score, freeze, evaluate, report
  tests/                   synthetic unit and boundary tests
docs/PAPER_PUBLIC_BASELINES_PREREGISTRATION_KO.md
docs/PAPER_PUBLIC_BASELINES_RUNBOOK_KO.md
docs/PAPER_PUBLIC_BASELINES_IMPLEMENTATION_REPORT_KO.md
scratch/run_paper_public_baselines.py
```

---

### Task 1: Freeze contracts, values-only data access, and preprocessing

**Files:**
- Create: `experiments/paper_public_baselines/__init__.py`
- Create: `experiments/paper_public_baselines/config.py`
- Create: `experiments/paper_public_baselines/contracts.py`
- Create: `experiments/paper_public_baselines/data.py`
- Create: `experiments/paper_public_baselines/preprocessing.py`
- Create: `experiments/paper_public_baselines/tests/__init__.py`
- Create: `experiments/paper_public_baselines/tests/test_data_boundary.py`
- Create: `experiments/paper_public_baselines/tests/test_preprocessing.py`

**Interfaces:**
- Produces: `FrozenConfig`, `MethodSpec`, `DatasetValues`, `PreparedDataset`, `list_datasets_values_only(db_path)`, `load_dataset_values(db_path, dataset)`, `prepare_dataset(values)`.
- Consumes: no proposed-method artifact and no TEST labels.

- [ ] **Step 1: Write failing data-boundary and preprocessing tests**

```python
class DataBoundaryTests(unittest.TestCase):
    def test_values_loader_never_selects_label_columns(self):
        db = build_toy_database(labels_as_text=True)
        observed_sql = []
        values = load_dataset_values(db, "toy", sql_observer=observed_sql.append)
        self.assertEqual(values.test_indices, (0, 1))
        self.assertFalse(any("label" in sql.lower() for sql in observed_sql))

    def test_population_is_name_sorted_and_exact(self):
        db = build_population_database(("zeta", "alpha"))
        self.assertEqual(list_datasets_values_only(db), ("alpha", "zeta"))

class PreprocessingTests(unittest.TestCase):
    def test_target_length_comes_only_from_train(self):
        raw = DatasetValues("toy", (row(8), row(10)), (0,), (row(200),))
        prepared = prepare_dataset(raw)
        self.assertEqual(prepared.target_length, 9)
        self.assertEqual(prepared.train.shape, (2, 9))
        self.assertEqual(prepared.test.shape, (1, 9))
        self.assertTrue(np.isfinite(prepared.train).all())
```

- [ ] **Step 2: Run tests and confirm missing-module failures**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_data_boundary experiments.paper_public_baselines.tests.test_preprocessing -v`

Expected: FAIL because `experiments.paper_public_baselines` does not exist.

- [ ] **Step 3: Implement frozen contracts and values-only access**

`config.py` must freeze:

```python
ALPHAS = (0.005, 0.01, 0.02)
SEED = 20260812
EXPECTED_DATASETS = 1117
EXPECTED_DB_SHA256 = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
METHOD_IDS = (
    "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest",
    "deep_svdd_1d", "drocc_1d",
)
```

`data.py` must query only `datasets.id`, `datasets.name`, `instances.dataset_id`,
`instances.split`, `instances.instance_index`, and `instances.values_blob`. It must
open SQLite with `mode=ro`, reject duplicate/non-contiguous indices, copy decoded
float32 arrays, and close every connection before returning.

- [ ] **Step 4: Implement deterministic preprocessing**

Reuse the scientific behavior of `sanitize_series`, TRAIN-median target length
clipped to `[8, 2048]`, linear interpolation, and per-row z-normalization. Do not
import Exp448 or its evaluation package.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_data_boundary experiments.paper_public_baselines.tests.test_preprocessing -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add experiments/paper_public_baselines
git commit -m "feat: add public baseline data boundary"
```

---

### Task 2: Implement classical published score adapters and calibration

**Files:**
- Create: `experiments/paper_public_baselines/methods/__init__.py`
- Create: `experiments/paper_public_baselines/methods/classical.py`
- Create: `experiments/paper_public_baselines/calibration.py`
- Create: `experiments/paper_public_baselines/tests/test_classical_methods.py`
- Create: `experiments/paper_public_baselines/tests/test_calibration.py`

**Interfaces:**
- Consumes: `PreparedDataset(train, test, test_indices)` from Task 1.
- Produces: `score_classical(method_id, train, query, seed) -> np.ndarray`, `crossfit_scores(method_id, train, test, seed) -> ScoreResult`, `conformal_alerts(oof, test, alpha) -> tuple[int, ...]`, `matched_top_k(scores, indices, k) -> tuple[int, ...]`.

- [ ] **Step 1: Write failing method tests**

```python
class ClassicalMethodTests(unittest.TestCase):
    def test_all_classical_scores_are_finite_and_higher_means_more_anomalous(self):
        train = np.asarray([[0, 0], [0, .1], [.1, 0], [-.1, 0], [0, -.1]], float)
        query = np.asarray([[0, 0], [9, 9]], float)
        for method in CLASSICAL_METHOD_IDS:
            scores = score_classical(method, train, query, seed=7)
            self.assertEqual(scores.shape, (2,))
            self.assertTrue(np.isfinite(scores).all())
            self.assertGreater(scores[1], scores[0])

    def test_lof_abstains_instead_of_substituting_another_detector(self):
        result = crossfit_scores("lof_novelty", np.zeros((1, 2)), np.zeros((2, 2)), 7)
        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.reason, "insufficient_train_rows")
```

- [ ] **Step 2: Run tests and confirm symbol failures**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_classical_methods experiments.paper_public_baselines.tests.test_calibration -v`

Expected: FAIL because the adapters are absent.

- [ ] **Step 3: Implement the four fixed classical adapters**

Use these exact constructors without TEST tuning:

```python
NearestNeighbors(n_neighbors=min(5, len(fit)))
LocalOutlierFactor(n_neighbors=min(20, len(fit) - 1), novelty=True,
                   contamination="auto", n_jobs=1)
OneClassSVM(kernel="rbf", gamma="scale", nu=0.01, cache_size=256)
IsolationForest(n_estimators=256, max_samples="auto", contamination="auto",
                random_state=seed, n_jobs=1)
```

k-NN score is mean distance to up to five normal neighbors. LOF, OCSVM, and
Isolation Forest return `-score_samples`, so larger always means more anomalous.
Generate TRAIN reference scores with deterministic shuffled K-fold cross-fitting;
never self-score a TRAIN row.

- [ ] **Step 4: Implement conformal and deterministic top-k rules**

```python
def upper_tail_p_values(reference, query):
    return np.asarray([
        (1 + np.count_nonzero(reference >= value)) / (len(reference) + 1)
        for value in query
    ])

def matched_top_k(scores, indices, k):
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), int(indices[i])))
    return tuple(sorted(int(indices[i]) for i in order[:k]))
```

`conformal_alerts` must use `p <= alpha`. Ties are deterministic by TEST index.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_classical_methods experiments.paper_public_baselines.tests.test_calibration -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add experiments/paper_public_baselines
git commit -m "feat: add published classical baseline scores"
```

---

### Task 3: Add immutable score shards, freeze verification, and resume

**Files:**
- Create: `experiments/paper_public_baselines/artifacts.py`
- Create: `experiments/paper_public_baselines/runner.py`
- Create: `experiments/paper_public_baselines/tests/test_artifacts.py`
- Create: `experiments/paper_public_baselines/tests/test_runner.py`

**Interfaces:**
- Consumes: Task 1 dataset loader and Task 2 `crossfit_scores`.
- Produces: `ScoreShard`, `score_dataset(...)`, `write_score_shard_create_only(...)`, `verify_score_root(...) -> FrozenScoreManifest`, `run_methods(...) -> RunSummary`.

- [ ] **Step 1: Write failing artifact and resume tests**

```python
class ArtifactTests(unittest.TestCase):
    def test_score_shard_schema_rejects_evaluation_fields(self):
        payload = valid_score_payload() | {"tp": 1}
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_score_payload(payload)

    def test_verified_complete_shard_is_reused_without_recompute(self):
        with tempfile.TemporaryDirectory() as directory:
            config = toy_config(Path(directory))
            first = run_methods(config, scorer=counting_scorer)
            second = run_methods(config, scorer=counting_scorer)
        self.assertEqual(first.completed, 1)
        self.assertEqual(second.completed, 1)
        self.assertEqual(counting_scorer.calls, 1)

    def test_tampered_shard_blocks_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_complete_toy_score_root(Path(directory))
            root.joinpath("shards/knn_distance/toy.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_score_root(root)
```

- [ ] **Step 2: Run tests and confirm failures**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_artifacts experiments.paper_public_baselines.tests.test_runner -v`

Expected: FAIL because artifact and runner modules are absent.

- [ ] **Step 3: Implement canonical, create-only shard artifacts**

Each shard contains exactly:

```json
{
  "dataset": "name",
  "method": "knn_distance",
  "n_train": 10,
  "n_test": 50,
  "status": "complete",
  "reason": null,
  "test_indices": [0, 1],
  "train_oof_scores": [0.1, 0.2],
  "test_scores": [0.3, 0.4],
  "alerts_by_alpha": {"0.005": [], "0.01": [], "0.02": []},
  "runtime_seconds": 0.1,
  "contains_test_labels": false,
  "contains_evaluation_metrics": false
}
```

Canonical JSON uses sorted keys, compact separators, UTF-8 newline, temporary
sibling files, `fsync`, and `os.replace`. Completed shard reuse requires exact
config hash, DB hash, method, dataset, score finiteness, index identity, and stored
digest. A mismatched existing shard fails closed rather than overwriting.

- [ ] **Step 4: Implement bounded CPU orchestration**

`run_methods` uses `ProcessPoolExecutor(max_workers=min(requested_workers, 7))`
only for classical CPU methods. It records PID, elapsed time, dependency versions,
seed, method config, dataset failures, and exact shard digests. It never catches a
method exception and silently substitutes another method.

- [ ] **Step 5: Run focused tests and the Task 1-3 suite**

Run: `python3 -m unittest discover -s experiments/paper_public_baselines/tests -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add experiments/paper_public_baselines
git commit -m "feat: freeze resumable public baseline scores"
```

---

### Task 4: Implement Deep SVDD, DROCC, and acceleration evidence

**Files:**
- Create: `experiments/paper_public_baselines/methods/deep_common.py`
- Create: `experiments/paper_public_baselines/methods/deep_svdd.py`
- Create: `experiments/paper_public_baselines/methods/drocc.py`
- Create: `experiments/paper_public_baselines/acceleration.py`
- Create: `experiments/paper_public_baselines/tests/test_deep_methods.py`
- Create: `experiments/paper_public_baselines/tests/test_acceleration.py`

**Interfaces:**
- Consumes: normalized `PreparedDataset` arrays.
- Produces: `score_deep_svdd(train, test, seed) -> ScoreResult`, `score_drocc(train, test, seed) -> ScoreResult`, `assess_mlx(method_id, smoke_input) -> AccelerationRecord`.

- [ ] **Step 1: Write failing deterministic deep-method tests**

```python
class DeepMethodTests(unittest.TestCase):
    def test_deep_svdd_same_seed_repeats_scores(self):
        first = score_deep_svdd(train8(), test3(), seed=11, epochs=2)
        second = score_deep_svdd(train8(), test3(), seed=11, epochs=2)
        np.testing.assert_allclose(first.test_scores, second.test_scores, rtol=0, atol=0)

    def test_drocc_returns_finite_scores_and_no_test_fit(self):
        result = score_drocc(train8(), test3(), seed=11, epochs=2)
        self.assertEqual(result.status, "complete")
        self.assertTrue(np.isfinite(result.test_scores).all())
        self.assertEqual(result.fit_rows, len(train8()))

    def test_mlx_ineligible_record_has_required_fallback_evidence(self):
        record = assess_mlx("isolation_forest", smoke_input())
        self.assertEqual(record.selected_backend, "cpu")
        self.assertFalse(record.parity_passed)
        self.assertIn("official scikit-learn", record.fallback_reason)
```

- [ ] **Step 2: Run tests and confirm missing-symbol failures**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_deep_methods experiments.paper_public_baselines.tests.test_acceleration -v`

Expected: FAIL.

- [ ] **Step 3: Implement the frozen 1D encoder and Deep SVDD objective**

Use bias-free layers and deterministic CPU execution:

```python
nn.Sequential(
    nn.Conv1d(1, 16, 5, padding=2, bias=False), nn.LeakyReLU(0.1),
    nn.Conv1d(16, 32, 5, padding=2, bias=False), nn.LeakyReLU(0.1),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 16, bias=False),
)
```

Initialize center `c` from the untrained TRAIN embeddings, clamp components with
absolute value below `0.1` to signed `0.1`, then optimize mean squared distance to
`c` using Adam `lr=1e-3`, weight decay `1e-6`, batch size `min(32, n_train)`, 50
epochs for official runs. Anomaly score is squared distance to `c`. Use the same
cross-fit interface as classical methods for TRAIN OOF calibration.

- [ ] **Step 4: Implement the frozen DROCC objective**

Use the same encoder plus a bias-free scalar head. Train binary-normal loss for
10 warm-up epochs, then 40 DROCC epochs. For every batch, create adversarial
points in the annulus `[r, gamma*r]` with `r=1.0`, `gamma=2.0`, 20 projected
gradient steps of size `0.1`, and loss weight `lambda=1.0`. Fix torch, NumPy, and
Python seeds; use deterministic CPU algorithms. Higher negative normal logit is
the anomaly score. `n_train < 8` produces an explicit abstention.

- [ ] **Step 5: Implement MLX eligibility records**

Record Python, NumPy, scikit-learn, torch, `mlx`, and `mlx-metal` versions,
device, exact stage, condition-1 eligibility, parity, candidate-index equality at
all three alphas, timing including synchronization, speed ratio, selected backend,
and fallback reason. Classical methods fail condition 1 because rewriting official
estimators changes the implementation. Deep methods remain CPU unless an exact
MLX path exists and passes all five AGENTS.md gates; do not create a nominal MLX
port solely for acceleration.

- [ ] **Step 6: Run focused tests**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_deep_methods experiments.paper_public_baselines.tests.test_acceleration -v`

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add experiments/paper_public_baselines
git commit -m "feat: add published deep one-class baselines"
```

---

### Task 5: Add closed CLI and run-bound lifecycle

**Files:**
- Create: `experiments/paper_public_baselines/cli.py`
- Create: `experiments/paper_public_baselines/tests/test_cli.py`
- Create: `scratch/run_paper_public_baselines.py`
- Create: `docs/PAPER_PUBLIC_BASELINES_PREREGISTRATION_KO.md`
- Create: `docs/PAPER_PUBLIC_BASELINES_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: `python3 scratch/run_paper_public_baselines.py {prepare,assess-mlx,score,freeze}`.

- [ ] **Step 1: Write failing CLI import-firewall and lifecycle tests**

```python
class CliTests(unittest.TestCase):
    def test_score_command_imports_no_label_module(self):
        code = (
            "import sys; import experiments.paper_public_baselines.cli as c; "
            "assert 'experiments.paper_public_baselines.label_access' not in sys.modules"
        )
        subprocess.run([sys.executable, "-B", "-c", code], check=True)

    def test_freeze_requires_all_dataset_method_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "coverage"):
                cli_main(["freeze", "--run-root", str(partial_root(Path(directory)))])
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_cli -v`

Expected: FAIL.

- [ ] **Step 3: Implement lazy command dispatch**

`prepare` verifies DB hash and exactly 1,117 datasets. `assess-mlx` writes the
complete acceleration record before any full method run. `score` accepts explicit
`--methods`, `--workers`, `--db-path`, and `--run-root`. Deep methods reject
`--workers` values above one. `freeze` verifies exact method/population coverage
and creates a no-clobber `frozen_score_manifest.json`.

- [ ] **Step 4: Write preregistration and runbook with exact commands**

The official commands are:

```bash
python3 scratch/run_paper_public_baselines.py prepare \
  --db-path /Users/mac/Documents/Dataset/univariate_ts.db \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
python3 scratch/run_paper_public_baselines.py assess-mlx \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
python3 scratch/run_paper_public_baselines.py score \
  --methods knn_distance,lof_novelty,ocsvm_rbf,isolation_forest \
  --workers 7 --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
python3 scratch/run_paper_public_baselines.py score \
  --methods deep_svdd_1d,drocc_1d \
  --workers 1 --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
python3 scratch/run_paper_public_baselines.py freeze \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
```

- [ ] **Step 5: Run CLI and full package tests**

Run: `python3 -m unittest discover -s experiments/paper_public_baselines/tests -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add experiments/paper_public_baselines scratch/run_paper_public_baselines.py docs/PAPER_PUBLIC_BASELINES_*.md
git commit -m "feat: add public baseline run lifecycle"
```

---

### Task 6: Implement post-freeze evaluation, matched budget, and statistics

**Files:**
- Create: `experiments/paper_public_baselines/label_access.py`
- Create: `experiments/paper_public_baselines/evaluation.py`
- Create: `experiments/paper_public_baselines/statistics.py`
- Create: `experiments/paper_public_baselines/tests/test_evaluation_boundary.py`
- Create: `experiments/paper_public_baselines/tests/test_statistics.py`

**Interfaces:**
- Consumes: verified `FrozenScoreManifest` and frozen Exp448 `dataset_metrics_antigravity.jsonl` plus candidate coordinates from the authenticated Exp446/Exp404 publication.
- Produces: `evaluate_frozen_scores(...) -> EvaluationBundle`, `paired_bootstrap(...) -> Interval`, full-population/common-support method summaries.

- [ ] **Step 1: Write failing post-freeze boundary tests**

```python
class EvaluationBoundaryTests(unittest.TestCase):
    def test_labels_are_not_opened_when_manifest_verification_fails(self):
        loader = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "manifest"):
                evaluate_frozen_scores(tampered_root(Path(directory)), loader)
        loader.assert_not_called()

    def test_matched_budget_uses_count_not_exp448_coordinates(self):
        proposal = {"toy": frozenset({7, 9})}
        baseline_scores = np.asarray([.1, .9, .8])
        selected = matched_budget_for_dataset(baseline_scores, (0, 1, 2), proposal["toy"])
        self.assertEqual(selected, (1, 2))

    def test_confusion_and_macro_micro_are_exact(self):
        bundle = evaluate_rows(toy_predictions(), toy_labels())
        self.assertEqual(bundle.micro["tp"], 1)
        self.assertEqual(bundle.micro["fp"], 1)
        self.assertAlmostEqual(bundle.micro["f1"], 0.5)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_evaluation_boundary experiments.paper_public_baselines.tests.test_statistics -v`

Expected: FAIL.

- [ ] **Step 3: Implement the evaluation-only label reader**

`label_access.py` is imported lazily only by the `evaluate` command. It opens the
exact DB read-only, authorizes only dataset name/id plus TEST index/label columns,
accepts exactly TEXT `"0"`/`"1"`, requires contiguous indices, and returns copied
`tuple[(index, int_label), ...]` rows after closing the connection.

- [ ] **Step 4: Implement evaluation protocols**

For every method report:

1. TRAIN-only conformal alerts for all fixed alphas.
2. Matched-budget top-k alerts where `k` is only the number of Exp448 final alerts
   for that dataset.
3. AP, optional explicitly named trapezoidal AUC-PR, and AUROC for complete score
   datasets.
4. Full-population abstention-as-no-alert metrics and common-support metrics.

Validate that Exp448 coordinates are used only to count proposal alerts in the
matched-budget path. Store the proposed method's frozen confusion as a separate
method row, not as an external baseline.

- [ ] **Step 5: Implement paired bootstrap**

Use seed `20260812`, 2,000 shared dataset-level draws, and percentile `[0.025,
0.975]`. Store point estimate, lower, upper, draw count, and seed for every
baseline-minus-proposal F1 comparison. Never choose methods from interval results.

- [ ] **Step 6: Run focused and full tests**

Run: `python3 -m unittest discover -s experiments/paper_public_baselines/tests -v`

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add experiments/paper_public_baselines
git commit -m "feat: evaluate frozen public baselines"
```

---

### Task 7: Generate paper tables and execute the official comparison

**Files:**
- Create: `experiments/paper_public_baselines/reporting.py`
- Create: `experiments/paper_public_baselines/tests/test_reporting.py`
- Create: `docs/PAPER_PUBLIC_BASELINES_IMPLEMENTATION_REPORT_KO.md`
- Modify: `docs/PAPER_PUBLIC_BASELINES_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: immutable evaluation JSON/JSONL from Task 6.
- Produces: `public_baseline_summary.csv`, `public_baseline_dataset_metrics.csv`, `public_baseline_bootstrap.json`, `PAPER_PUBLIC_BASELINE_RESULT_KO.md`, and a compact committed implementation report.

- [ ] **Step 1: Write failing report tests**

```python
class ReportingTests(unittest.TestCase):
    def test_report_separates_public_baselines_and_author_ablation(self):
        text = render_report(toy_evaluation_bundle())
        self.assertIn("공개 기법 비교", text)
        self.assertIn("제안기술 내부 ablation", text)
        self.assertNotIn("Exp404/416", public_comparison_section(text))

    def test_report_never_claims_field_validation(self):
        text = render_report(toy_evaluation_bundle())
        for forbidden in ("양산 검증 완료", "현장 효과 입증", "고장 예방 보장"):
            self.assertNotIn(forbidden, text)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_reporting -v`

Expected: FAIL.

- [ ] **Step 3: Implement canonical CSV/JSON/Markdown reporting**

The main table columns are method, protocol, alpha, coverage, alerts, TP, FP, FN,
TN, precision, recall, micro F1, macro F1, FPR, AP, AUROC. Add separate common-
support and paired-bootstrap tables. The ablation table maps author provenance to
`제안기술 기본 구성`, `후보 확장 적용 구성`, and `안정성 선별 적용 제안기술`.

- [ ] **Step 4: Run full unit suite and smoke population**

Run:

```bash
python3 -m unittest discover -s experiments/paper_public_baselines/tests -v
python3 scratch/run_paper_public_baselines.py prepare \
  --db-path /Users/mac/Documents/Dataset/univariate_ts.db \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_smoke \
  --limit 3
python3 scratch/run_paper_public_baselines.py assess-mlx \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_smoke
python3 scratch/run_paper_public_baselines.py score \
  --methods knn_distance,lof_novelty,ocsvm_rbf,isolation_forest,deep_svdd_1d,drocc_1d \
  --workers 1 --epochs 2 \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_smoke
```

Expected: all six methods create finite complete or preregistered abstention shards;
no score artifact contains TEST labels or evaluation fields.

- [ ] **Step 5: Execute official prepare and MLX assessment**

Run the first two official Task 5 commands. Verify DB hash, exact 1,117 population,
runtime versions, and CPU fallback evidence before scoring.

- [ ] **Step 6: Execute all classical scores**

Run the official classical score command with seven workers. Poll the resumable
status without deleting or overwriting completed shards. On any method failure,
record the exact dataset, exception, and status; do not substitute a detector.

- [ ] **Step 7: Execute Deep SVDD and DROCC sequentially**

Run the official deep score command with one worker. Never overlap Deep SVDD and
DROCC MPS/MLX-heavy work. If projected runtime is impractical after the frozen
pilot, retain the completed classical full comparison and mark deep coverage as
incomplete rather than changing epochs from TEST feedback.

- [ ] **Step 8: Freeze once, evaluate, and report**

```bash
python3 scratch/run_paper_public_baselines.py freeze \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
python3 scratch/run_paper_public_baselines.py evaluate \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1 \
  --exp448-root /Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812
python3 scratch/run_paper_public_baselines.py report \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
```

Expected: immutable score/evaluation hashes, exact method/population coverage,
separate public baseline and author-ablation tables, and no promotion claim.

- [ ] **Step 9: Verify result reproducibility and documentation**

Run:

```bash
python3 -m unittest discover -s experiments/paper_public_baselines/tests -v
python3 scratch/run_paper_public_baselines.py report \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1 \
  --verify-only
git diff --check
```

Expected: PASS; report regeneration matches stored hashes byte-for-byte.

- [ ] **Step 10: Commit Task 7**

```bash
git add experiments/paper_public_baselines docs/PAPER_PUBLIC_BASELINES_*.md scratch/run_paper_public_baselines.py
git commit -m "docs: report public anomaly baseline comparison"
```

---

## Plan Self-Review

- Spec coverage: all six published methods, two threshold protocols, AP/AUROC,
  full/common support, label boundary, resume, bootstrap, MLX evidence, and paper
  wording have an owning task.
- Placeholder scan: no implementation step depends on an unspecified method,
  path, alpha, seed, metric, or command.
- Type consistency: Tasks 2-7 consume `PreparedDataset`, `ScoreResult`,
  `FrozenScoreManifest`, and `EvaluationBundle` introduced in earlier tasks.
- Scope: field validation remains excluded and does not block public benchmark
  comparison.
