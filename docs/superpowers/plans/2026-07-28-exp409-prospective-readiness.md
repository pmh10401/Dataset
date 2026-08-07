# Exp409 Prospective-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, label-blind Exp409 shadow pipeline that freezes the strict Exp408 consensus contract, enforces a 100-candidate publication budget, preserves every operational lane, and records component-level provenance without evaluating TEST performance.

**Architecture:** A new `experiments.exp409_prospective_readiness` package wraps the existing TRAIN-only scoring and consensus functions without loading Exp403-408 checkpoints or evaluation artifacts. Pure contracts, guards, source construction, provenance, and publication modules feed a small orchestration layer; publication is staged in a sibling directory and atomically renamed only after every gate passes.

**Tech Stack:** Python 3.14, NumPy, scikit-learn, PyTorch CPU, SQLite read-only tracing, `unittest`, existing Exp403-408 artifact and model helpers.

## Global Constraints

- Experiment identifier is exactly `Exp409`.
- Frozen variant is exactly `two_new_plus_existing_independent`.
- Frozen alpha is exactly `0.005`.
- Global publication budget is exactly 100 candidates.
- Outputs are review-only Shadow; Hard, Standard, and Priority must remain unchanged.
- Prediction must not read TEST labels, actual anomaly positions, Exp408 evaluation CSVs, or historical TEST performance.
- Exp403-408 checkpoints must not be loaded or accepted as inputs.
- Budget overflow publishes zero candidate artifacts; no top-100 selection is permitted.
- Current work authorizes only a deterministic representative smoke, not a full 1,117-dataset execution.
- Performance is `not_evaluated`; `prospective_validated`, `deployment_eligible`, and `operational_policy_changed` are all `false`.
- Each task uses test-first development and ends in a focused commit.

---

## File Structure

- `experiments/exp409_prospective_readiness/__init__.py`: public Exp409 API only.
- `experiments/exp409_prospective_readiness/config.py`: frozen constants and immutable config serialization.
- `experiments/exp409_prospective_readiness/contracts.py`: candidate, source-status, dataset-result, and batch-result value objects plus schema validation.
- `experiments/exp409_prospective_readiness/guard.py`: recursive forbidden-field validation, SQLite trace enforcement, and forbidden-input rejection.
- `experiments/exp409_prospective_readiness/sources.py`: fresh in-memory construction of thirteen Exp408-compatible sources and their statuses.
- `experiments/exp409_prospective_readiness/provenance.py`: individual file/value hashes, clean-Git requirement, canonical closure, and runtime envelope.
- `experiments/exp409_prospective_readiness/publication.py`: batch gates, deterministic payload rendering, staging-directory transaction, and failure record.
- `experiments/exp409_prospective_readiness/runner.py`: dataset orchestration, deterministic TRAIN-only smoke selection, lane parity, and batch assembly.
- `scratch/run_exp409_prospective_readiness.py`: smoke-only command-line entrypoint.
- `experiments/exp409_prospective_readiness/tests/`: focused unit and integration tests.
- `docs/EXP409_PREREGISTRATION_KO.md`: human-readable frozen experiment contract.
- `docs/EXP409_RUNBOOK_KO.md`: allowed smoke command, checks, artifacts, and stop conditions.

---

### Task 1: Freeze Exp409 Configuration and Output Contracts

**Files:**
- Create: `experiments/exp409_prospective_readiness/__init__.py`
- Create: `experiments/exp409_prospective_readiness/config.py`
- Create: `experiments/exp409_prospective_readiness/contracts.py`
- Create: `experiments/exp409_prospective_readiness/tests/__init__.py`
- Create: `experiments/exp409_prospective_readiness/tests/test_contracts.py`

**Interfaces:**
- Produces: `Exp409FrozenConfig.to_payload() -> dict[str, object]`
- Produces: `FROZEN_CONFIG: Exp409FrozenConfig`
- Produces: `SourceStatus`, `ShadowCandidate`, `DatasetReadinessResult`, and `ReadinessBatch`
- Produces: `validate_candidate(candidate: ShadowCandidate, *, n_test: int, hard_indices: Collection[int]) -> None`
- Produces: `validate_batch(batch: ReadinessBatch, *, review_budget: int) -> None`

- [ ] **Step 1: Write failing frozen-config tests**

```python
class FrozenConfigTests(unittest.TestCase):
    def test_frozen_values_are_exact(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp409")
        self.assertEqual(FROZEN_CONFIG.variant, "two_new_plus_existing_independent")
        self.assertEqual(FROZEN_CONFIG.alpha, "0.005")
        self.assertEqual(FROZEN_CONFIG.review_budget, 100)
        self.assertEqual(FROZEN_CONFIG.performance_evaluation, "not_evaluated")
        self.assertFalse(FROZEN_CONFIG.deployment_eligible)

    def test_config_is_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            FROZEN_CONFIG.alpha = "0.01"
```

- [ ] **Step 2: Run the config tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_contracts
```

Expected: import failure because the Exp409 package does not exist.

- [ ] **Step 3: Implement the immutable configuration**

```python
@dataclass(frozen=True)
class Exp409FrozenConfig:
    experiment_id: str = "Exp409"
    variant: str = "two_new_plus_existing_independent"
    alpha: str = "0.005"
    review_budget: int = 100
    seed: int = 20260727
    independence_threshold: float = 0.90
    minimum_oof_rows: int = 20
    disposition: str = "review_only"
    performance_evaluation: str = "not_evaluated"
    prospective_validated: bool = False
    deployment_eligible: bool = False
    operational_policy_changed: bool = False

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


FROZEN_CONFIG = Exp409FrozenConfig()
```

Validate the exact constants in `__post_init__`; constructing a changed config must raise `ValueError`. This prevents tests or callers from creating an alternate research configuration.

- [ ] **Step 4: Write failing candidate and batch validation tests**

```python
def candidate(index: int) -> ShadowCandidate:
    return ShadowCandidate(
        dataset="sample",
        candidate_index=index,
        variant=FROZEN_CONFIG.variant,
        alpha=FROZEN_CONFIG.alpha,
        supporting_components=(("tri_view",), ("synthetic_a",), ("existing_a",)),
        supporting_source_markers=(("tri_view",), ("synthetic_a",), ("existing_a",)),
        disposition="review_only",
    )


class CandidateContractTests(unittest.TestCase):
    def test_hard_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Hard"):
            validate_candidate(candidate(3), n_test=10, hard_indices={3})

    def test_candidate_outside_test_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "range"):
            validate_candidate(candidate(10), n_test=10, hard_indices=set())

    def test_duplicate_batch_candidate_is_rejected(self) -> None:
        batch = batch_with_candidates((candidate(2), candidate(2)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_batch(batch, review_budget=100)

    def test_budget_101_is_rejected(self) -> None:
        batch = batch_with_candidates(tuple(candidate(index) for index in range(101)))
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_batch(batch, review_budget=100)
```

- [ ] **Step 5: Implement value objects and strict validators**

Use frozen dataclasses. `DatasetReadinessResult` contains dataset identity, TRAIN size, TEST size, original lane tuples, exactly thirteen `SourceStatus` objects, zero or more candidates, source-output hashes, and status. `ReadinessBatch` contains ordered dataset results and no timestamps.

`validate_batch` must:

```python
keys = [(row.dataset, row.candidate_index) for row in batch.candidates]
if len(keys) != len(set(keys)):
    raise ValueError("duplicate Exp409 candidate")
if len(keys) > review_budget:
    raise ValueError("Exp409 review budget exceeded")
```

It also verifies exact variant, alpha, disposition, and thirteen unique source identities per non-fixture dataset.

- [ ] **Step 6: Run Task 1 tests**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_contracts
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add experiments/exp409_prospective_readiness
git commit -m "feat: freeze Exp409 readiness contracts"
```

---

### Task 2: Enforce the Label-Blind Boundary

**Files:**
- Create: `experiments/exp409_prospective_readiness/guard.py`
- Create: `experiments/exp409_prospective_readiness/tests/test_guard.py`

**Interfaces:**
- Consumes: existing `experiments.exp403_408.prediction.find_forbidden_fields`
- Produces: `validate_label_blind_payload(value: object, path: str = "prediction") -> None`
- Produces: `reject_forbidden_inputs(paths: Sequence[Path]) -> None`
- Produces: `LabelBlindSqlTrace`, a callable trace collector with `assert_clean() -> None`
- Produces: `load_values_bundle_label_blind(dataset: str, path: Path, trace: LabelBlindSqlTrace) -> DatasetBundle`

- [ ] **Step 1: Write failing payload and input-path tests**

```python
class GuardTests(unittest.TestCase):
    def test_test_label_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_label_blind_payload({"dataset": "a", "test_labels": [0, 1]})

    def test_exp408_evaluation_csv_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluation input"):
            reject_forbidden_inputs(
                [Path("/data/exp408_variant_candidate_evaluation.csv")]
            )

    def test_exp408_checkpoint_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            reject_forbidden_inputs(
                [Path("/data/exp403_408_feature_ml_validation/checkpoints")]
            )
```

- [ ] **Step 2: Run the guard tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_guard
```

Expected: import failure for `guard`.

- [ ] **Step 3: Implement recursive payload and path guards**

Delegate recursive field detection to `find_forbidden_fields`, then add Exp409-specific forbidden exact names:

```python
FORBIDDEN_INPUT_NAMES = frozenset(
    {
        "exp408_variant_metrics.csv",
        "exp408_variant_candidate_evaluation.csv",
        "exp403_408_frozen_test_labels.csv",
    }
)
```

Reject any path whose components include both `exp403_408_feature_ml_validation` and `checkpoints`.

- [ ] **Step 4: Write failing SQLite trace tests**

```python
def test_normal_train_and_unlabeled_test_values_are_allowed(self) -> None:
    trace = LabelBlindSqlTrace()
    trace("SELECT value FROM train_normal WHERE dataset = 'a'")
    trace("SELECT value FROM test_values WHERE dataset = 'a'")
    trace.assert_clean()


def test_test_label_query_is_rejected(self) -> None:
    trace = LabelBlindSqlTrace()
    trace("SELECT is_anomaly FROM test_labels WHERE dataset = 'a'")
    with self.assertRaisesRegex(ValueError, "TEST label"):
        trace.assert_clean()


def test_test_derived_ordering_is_rejected(self) -> None:
    trace = LabelBlindSqlTrace()
    trace("SELECT dataset FROM test_metadata ORDER BY anomaly_count")
    with self.assertRaisesRegex(ValueError, "TEST-derived"):
        trace.assert_clean()
```

- [ ] **Step 5: Implement normalized SQL trace classification**

Store every statement. Normalize whitespace and lowercase text. Reject label/anomaly truth table or column tokens, TEST metadata population queries, TEST-derived `ORDER BY`, and label blob access. Permit TEST value reads needed to score unlabeled candidates.

- [ ] **Step 6: Write the failing values-only loader integration test**

Create a temporary SQLite fixture whose `instances` table contains both
`values_blob` and `label`. Call `load_values_bundle_label_blind`, then assert
that it returns TRAIN/TEST values and that the recorded SQL contains
`values_blob`, split, dataset identity, and instance ordering but never reads
the `label` column.

- [ ] **Step 7: Run the loader test and verify RED**

Run:

```bash
python3 -m unittest \
  experiments.exp409_prospective_readiness.tests.test_guard.GuardTests.test_values_loader_never_reads_label_column
```

Expected: failure because `load_values_bundle_label_blind` is not defined.

- [ ] **Step 8: Implement the traced values-only loader**

Implement the loader with its own read-only SQLite connection:

```python
connection = sqlite3.connect(
    f"file:{Path(path).resolve(strict=True)}?mode=ro",
    uri=True,
)
connection.set_trace_callback(trace)
```

Use the exact values-only join and `ORDER BY i.instance_index` pattern from
`experiments.exp226_235_hard_alert_quality.database._load_split_values`.
Return `DatasetBundle` and close the connection in `finally`. Do not call the
evaluation label loader.

- [ ] **Step 9: Run Task 2 tests**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_guard
```

Expected: all tests pass.

- [ ] **Step 10: Commit Task 2**

```bash
git add experiments/exp409_prospective_readiness/guard.py experiments/exp409_prospective_readiness/tests/test_guard.py
git commit -m "feat: guard Exp409 label boundary"
```

---

### Task 3: Recompute Thirteen Sources In Memory

**Files:**
- Create: `experiments/exp409_prospective_readiness/sources.py`
- Create: `experiments/exp409_prospective_readiness/tests/test_sources.py`

**Interfaces:**
- Consumes: `DatasetInput`, `FeatureBlock`, `ConsensusSource`, `ShadowSourcePrediction`
- Consumes: `load_feature_blocks`, `score_tri_view`, `shadow_predictions`, `score_synthetic_models`, `synthetic_shadow_predictions`, and the same OOF helpers used by the Exp403-408 pipeline
- Produces: `SourceBatch(sources: tuple[ConsensusSource, ...], statuses: tuple[SourceStatus, ...], output_sha256: Mapping[str, str])`
- Produces: `build_source_batch(dataset: str, bundle: DatasetInput, config: Exp409FrozenConfig = FROZEN_CONFIG) -> SourceBatch`

- [ ] **Step 1: Write failing thirteen-source coverage tests**

Use mocks for model functions so the unit test stays fast:

```python
class SourceBatchTests(unittest.TestCase):
    @patch("experiments.exp409_prospective_readiness.sources.load_feature_blocks")
    @patch("experiments.exp409_prospective_readiness.sources.score_tri_view")
    @patch("experiments.exp409_prospective_readiness.sources.score_synthetic_models")
    def test_records_exact_thirteen_source_statuses(
        self, synthetic_scores, tri_scores, feature_blocks
    ) -> None:
        feature_blocks.return_value = nine_valid_blocks()
        tri_scores.return_value = valid_tri_score()
        synthetic_scores.return_value = three_valid_synthetic_scores()
        result = build_source_batch("sample", sample_bundle())
        self.assertEqual(len(result.statuses), 13)
        self.assertEqual(len({row.source for row in result.statuses}), 13)
```

Add cases for explicit `n_train_below_50`, feature `source_abstained`, source identity mismatch, TRAIN length mismatch, non-finite content, and an unknown abstention reason.

- [ ] **Step 2: Run source tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_sources
```

Expected: import failure for `sources`.

- [ ] **Step 3: Implement existing-feature conversion**

For each registered `FEATURE_SOURCES` entry:

1. verify unique name, expected group, and 64-character source hash;
2. verify one-dimensional finite TRAIN OOF and TEST values with exact bundle lengths;
3. compute empirical upper-tail p-values using only the source's normal TRAIN OOF vector;
4. exclude frozen Hard indices;
5. construct a `ConsensusSource` with kind `existing`;
6. hash the canonical source output separately.

Do not import Exp408 evaluation modules or files.

- [ ] **Step 4: Implement fresh tri-view and synthetic source conversion**

Follow the already verified Exp403-408 call order:

```python
frozen = FrozenConfig(target_length=bundle.target_length)
tri_scores = score_tri_view(bundle.train_normal, bundle.test_values, frozen, device="cpu")
tri_raw = _produce_raw_scores(
    np.asarray(bundle.train_normal),
    np.asarray(bundle.test_values),
    frozen,
    "cpu",
)
tri_predictions = shadow_predictions(dataset, tri_scores, bundle.hard_indices)
```

For each `MODEL_NAMES` entry, derive TRAIN OOF with `_crossfit_normal_scores` using `seed + model_index * 10_000`, then pair it with `score_synthetic_models` and `synthetic_shadow_predictions`.

If TRAIN eligibility is below 50, create the four explicit learned-source abstention statuses and no learned `ConsensusSource` objects. Never replace an abstention with another model.

- [ ] **Step 5: Enforce source identity, reason, and output hashes**

Allowed abstention reasons are the explicit sets already accepted by Exp403-408. Unknown, malformed, or inconsistent outputs raise `ValueError`. Hash each normalized source/status payload with `canonical_json_bytes` and store it under its exact source name.

- [ ] **Step 6: Run Task 3 tests**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_sources
```

Expected: all tests pass without loading checkpoint files.

- [ ] **Step 7: Commit Task 3**

```bash
git add experiments/exp409_prospective_readiness/sources.py experiments/exp409_prospective_readiness/tests/test_sources.py
git commit -m "feat: recompute Exp409 consensus sources"
```

---

### Task 4: Build Component-Level Provenance

**Files:**
- Create: `experiments/exp409_prospective_readiness/provenance.py`
- Create: `experiments/exp409_prospective_readiness/tests/test_provenance.py`

**Interfaces:**
- Consumes: `sha256_file`, `canonical_json_bytes`, `Exp409FrozenConfig`
- Produces: `require_clean_git(repo_root: Path) -> str`
- Produces: `dataset_values_sha256(bundle: DatasetInput) -> str`
- Produces: `build_canonical_provenance(...) -> dict[str, object]`
- Produces: `provenance_closure_sha256(payload: Mapping[str, object]) -> str`
- Produces: `build_runtime_envelope(run_id: str, started_at: str) -> dict[str, object]`

- [ ] **Step 1: Write failing clean-Git and completeness tests**

```python
class ProvenanceTests(unittest.TestCase):
    def test_dirty_git_blocks_publication(self) -> None:
        with patch("...provenance._git_status", return_value=" M file.py"):
            with self.assertRaisesRegex(ValueError, "dirty"):
                require_clean_git(Path("/repo"))

    def test_missing_source_hash_is_rejected(self) -> None:
        payload = valid_provenance()
        del payload["source_output_sha256"]["tri_view"]
        with self.assertRaisesRegex(ValueError, "source provenance"):
            validate_canonical_provenance(payload)

    def test_runtime_envelope_does_not_change_closure(self) -> None:
        payload = valid_provenance()
        first = provenance_closure_sha256(payload)
        second = provenance_closure_sha256(payload)
        self.assertEqual(first, second)
```

- [ ] **Step 2: Run provenance tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_provenance
```

Expected: import failure for `provenance`.

- [ ] **Step 3: Implement clean-Git detection and deterministic value hashing**

Use subprocess arguments, never a shell string:

```python
subprocess.run(
    [git_executable, "-C", str(repo_root), "status", "--porcelain"],
    check=True,
    text=True,
    capture_output=True,
)
```

Resolve the executable with `shutil.which("git")`; permit test injection when Apple Git is unavailable. Hash contiguous float64 TRAIN and TEST arrays with their field names and shapes, matching the proven Exp403-408 value-hash convention.

- [ ] **Step 4: Implement canonical provenance and validation**

Require these exact top-level mappings:

```python
{
    "git": {"commit": ..., "dirty": False},
    "config_sha256": ...,
    "code_sha256": {...},
    "database_snapshot_sha256": {...},
    "lane_artifact_sha256": {...},
    "source_artifact_sha256": {...},
    "dataset_values_sha256": {...},
    "source_output_sha256": {...},
    "upstream_artifact_sha256": {...},
    "candidate_sha256": ...,
    "gates": {
        "label_access": "passed",
        "lane_parity": "passed",
        "review_budget": "passed",
        "determinism": "passed",
    },
}
```

Every digest is lowercase hexadecimal length 64. The closure is SHA256 of this canonical payload. Runtime timestamps, run ID, Python packages, and platform live under a separately hashed `runtime_envelope`.

- [ ] **Step 5: Run Task 4 tests**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_provenance
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add experiments/exp409_prospective_readiness/provenance.py experiments/exp409_prospective_readiness/tests/test_provenance.py
git commit -m "feat: record Exp409 component provenance"
```

---

### Task 5: Add Lane, Budget, and Atomic Publication Gates

**Files:**
- Create: `experiments/exp409_prospective_readiness/publication.py`
- Create: `experiments/exp409_prospective_readiness/tests/test_publication.py`

**Interfaces:**
- Consumes: validated `ReadinessBatch`, canonical provenance, and runtime envelope
- Produces: `publish_batch(batch: ReadinessBatch, output_dir: Path, canonical_provenance: Mapping[str, object], runtime_envelope: Mapping[str, object]) -> Mapping[str, str]`
- Produces: `write_failure_record(output_dir: Path, *, status: str, reason: str) -> Path`

- [ ] **Step 1: Write failing no-partial-publication tests**

```python
class PublicationTests(unittest.TestCase):
    def test_budget_overflow_writes_no_candidate_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "publication"
            with self.assertRaisesRegex(ValueError, "budget"):
                publish_batch(batch_with_count(101), output, valid_provenance(), runtime())
            self.assertFalse(output.exists())
            self.assertFalse((output / "exp409_shadow_candidates.jsonl").exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "publication"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "output"):
                publish_batch(batch_with_count(1), output, valid_provenance(), runtime())
```

- [ ] **Step 2: Run publication tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_publication
```

Expected: import failure for `publication`.

- [ ] **Step 3: Implement deterministic artifact rendering**

Render ordered payloads in memory before touching the destination:

- config JSON;
- candidate JSONL sorted by `(dataset, candidate_index)`;
- source-status JSONL sorted by `(dataset, source)`;
- canonical provenance manifest;
- readiness summary;
- Korean readiness report.

Call `validate_label_blind_payload` on every prediction-side payload. Set the summary claim fields to the exact frozen values from Global Constraints.

- [ ] **Step 4: Implement staging-directory publication**

```python
staging = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.staging")
try:
    staging.mkdir(parents=False)
    # write and verify every staged file
    os.replace(staging, output_dir)
except Exception:
    shutil.rmtree(staging, ignore_errors=True)
    raise
```

Require `output_dir.parent` to exist and `output_dir` not to exist. Validate batch, lanes, budget, payload schemas, file hashes, and provenance closure before `os.replace`.

- [ ] **Step 5: Implement failure records outside the publication directory**

Write only `<output-name>.failure.json` atomically. It contains experiment ID, `blocked` or `error`, a non-sensitive reason, and no candidate rows.

- [ ] **Step 6: Run Task 5 tests**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_publication
```

Expected: all tests pass, including budget 100 success and 101 fail-closed.

- [ ] **Step 7: Commit Task 5**

```bash
git add experiments/exp409_prospective_readiness/publication.py experiments/exp409_prospective_readiness/tests/test_publication.py
git commit -m "feat: publish Exp409 atomically"
```

---

### Task 6: Orchestrate Dataset Readiness and the Smoke CLI

**Files:**
- Create: `experiments/exp409_prospective_readiness/runner.py`
- Create: `scratch/run_exp409_prospective_readiness.py`
- Create: `experiments/exp409_prospective_readiness/tests/test_runner.py`
- Create: `experiments/exp409_prospective_readiness/tests/test_cli.py`

**Interfaces:**
- Consumes: `load_dataset_input`, `load_frozen_lanes`, `build_source_batch`, `build_consensus`, guards, provenance, and publication
- Produces: `run_dataset(dataset: str, *, config: Exp409FrozenConfig = FROZEN_CONFIG) -> DatasetReadinessResult`
- Produces: `select_exp409_smoke_datasets(limit: int = 8) -> tuple[tuple[str, ...], Mapping[str, int]]`
- Produces: `run_smoke(output_dir: Path, *, limit: int = 8) -> Mapping[str, str]`
- Produces CLI: `python3 scratch/run_exp409_prospective_readiness.py --representative-smoke --dataset-limit 8 --output-dir <new-dir>`

- [ ] **Step 1: Write failing strict-consensus and lane-parity tests**

```python
class RunnerTests(unittest.TestCase):
    @patch("...runner.build_source_batch")
    @patch("...runner.load_dataset_input")
    @patch("...runner.load_frozen_lanes")
    def test_run_dataset_exposes_only_frozen_variant_and_alpha(
        self, lanes, loader, source_builder
    ) -> None:
        loader.return_value = sample_bundle()
        lanes.return_value = sample_lane_universe()
        source_builder.return_value = sample_sources()
        result = run_dataset("sample")
        self.assertTrue(all(row.variant == FROZEN_CONFIG.variant for row in result.candidates))
        self.assertTrue(all(row.alpha == FROZEN_CONFIG.alpha for row in result.candidates))

    def test_lane_mutation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "lane parity"):
            validate_lane_parity(original_lanes(), changed_lanes())
```

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_runner
```

Expected: import failure for `runner`.

- [ ] **Step 3: Implement per-dataset orchestration**

`run_dataset` creates a fresh `LabelBlindSqlTrace`, injects
`load_values_bundle_label_blind` into `load_dataset_input`, and performs:

```python
bundle = load_dataset_input(
    dataset,
    loader=partial(
        load_values_bundle_label_blind,
        trace=sql_trace,
    ),
)
source_batch = build_source_batch(dataset, bundle, config)
consensus = build_consensus(
    load_frozen_lanes(),
    source_batch.sources,
    threshold=config.independence_threshold,
    minimum_rows=config.minimum_oof_rows,
)
frozen_rows = consensus.candidates_by_variant[config.variant][config.alpha]
```

If every eligible source explicitly abstains, return an `abstained` dataset result with unchanged lanes and zero candidates. Otherwise require exactly thirteen statuses, validate each candidate, and store only the frozen rows.

- [ ] **Step 4: Write failing TRAIN-only smoke selection tests**

```python
def test_smoke_covers_all_train_tiers(self) -> None:
    datasets, coverage = select_exp409_smoke_datasets(limit=8)
    self.assertEqual(set(coverage), {
        "abstain", "robust_distance", "classical_committee", "learned_methods"
    })
    self.assertTrue(all(count >= 1 for count in coverage.values()))


def test_selector_does_not_accept_test_order_or_labels(self) -> None:
    signature = inspect.signature(select_exp409_smoke_datasets)
    self.assertNotIn("test_labels", signature.parameters)
    self.assertNotIn("test_order", signature.parameters)
```

- [ ] **Step 5: Implement smoke selection and batch execution**

Reuse `select_representative_smoke_datasets` and
`list_datasets_by_train_normal_count`. Pass only the canonical normal TRAIN
count lookup. Require `limit >= 4`; default to 8. Run sequentially on CPU to
keep artifact order and resource use deterministic.

The selector uses a separate explicit TRAIN-count-only query. The dataset
loader owns a traced read-only SQLite connection through
`load_values_bundle_label_blind`. After every dataset, call
`sql_trace.assert_clean()`.

`run_smoke` assembles the selected batch twice in memory from the same frozen
inputs before publication. Compare canonical candidate bytes, per-dataset
source-output hashes, lane payloads, and the canonical provenance inputs. Any
difference raises `ValueError("Exp409 determinism gate failed")`; only a match
sets the determinism gate to `passed`.

- [ ] **Step 6: Write failing CLI authorization tests**

```python
class CliTests(unittest.TestCase):
    def test_full_run_flag_does_not_exist(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ENTRYPOINT), "--help"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertNotIn("--stage all", completed.stdout)
        self.assertIn("--representative-smoke", completed.stdout)

    def test_output_dir_must_not_exist(self) -> None:
        with TemporaryDirectory() as directory:
            completed = run_cli(Path(directory), limit=4)
            self.assertNotEqual(completed.returncode, 0)
```

- [ ] **Step 7: Implement the smoke-only CLI**

Require all of:

```text
--representative-smoke
--dataset-limit 4..20
--output-dir <new path whose parent exists>
```

Do not expose variant, alpha, budget, seed, threshold, device, resume, reset,
or full-run overrides.

- [ ] **Step 8: Run Task 6 tests**

Run:

```bash
python3 -m unittest \
  experiments.exp409_prospective_readiness.tests.test_runner \
  experiments.exp409_prospective_readiness.tests.test_cli
```

Expected: all tests pass without a full historical execution.

- [ ] **Step 9: Commit Task 6**

```bash
git add experiments/exp409_prospective_readiness/runner.py \
  experiments/exp409_prospective_readiness/tests/test_runner.py \
  experiments/exp409_prospective_readiness/tests/test_cli.py \
  scratch/run_exp409_prospective_readiness.py
git commit -m "feat: add Exp409 readiness smoke runner"
```

---

### Task 7: Preregister, Run the Authorized Smoke, and Verify the Branch

**Files:**
- Create: `docs/EXP409_PREREGISTRATION_KO.md`
- Create: `docs/EXP409_RUNBOOK_KO.md`
- Create: `experiments/exp409_prospective_readiness/tests/test_documentation.py`
- Generate outside Git by default: `/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke`

**Interfaces:**
- Consumes: the CLI and artifact schemas from Tasks 1-6
- Produces: frozen Korean preregistration, executable runbook, and verified smoke publication

- [ ] **Step 1: Write failing documentation contract tests**

```python
class DocumentationTests(unittest.TestCase):
    def test_preregistration_contains_every_frozen_boundary(self) -> None:
        text = PREREGISTRATION.read_text(encoding="utf-8")
        for required in (
            "two_new_plus_existing_independent",
            "0.005",
            "100",
            "review-only",
            "not_evaluated",
            "deployment_eligible: false",
            "TEST label",
        ):
            self.assertIn(required, text)

    def test_runbook_has_no_full_or_resume_command(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("--representative-smoke", text)
        self.assertNotIn("--stage all", text)
        self.assertNotIn("--resume", text)
```

- [ ] **Step 2: Run documentation tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp409_prospective_readiness.tests.test_documentation
```

Expected: failure because the documents do not exist.

- [ ] **Step 3: Write preregistration and runbook**

The preregistration copies every exact Global Constraint and states that no
new data currently exists. It separates readiness gates from future
performance evaluation.

The runbook contains only:

```bash
export TIMESERIES_DATA_ROOT="/Users/mac/Documents/Dataset"
python3 scratch/run_exp409_prospective_readiness.py \
  --representative-smoke \
  --dataset-limit 8 \
  --output-dir \
  "/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke"
```

It lists output schemas, SHA verification commands, budget/lane/leakage stop
conditions, and the prohibition on reusing the output directory.

- [ ] **Step 4: Run all Exp409 unit tests**

Run:

```bash
python3 -m unittest discover \
  -s experiments/exp409_prospective_readiness/tests \
  -p "test_*.py"
```

Expected: all Exp409 tests pass.

- [ ] **Step 5: Run all Exp403-408 regression tests**

Run:

```bash
TIMESERIES_DATA_ROOT="/Users/mac/Documents/Dataset" \
python3 -m unittest discover \
  -s experiments/exp403_408/tests \
  -p "test_*.py"
```

Expected: 160 tests pass. A pre-existing SQLite `ResourceWarning` is
non-failing; any test failure blocks the smoke.

- [ ] **Step 6: Compile every affected Python file**

Run:

```bash
python3 -m py_compile \
  experiments/exp409_prospective_readiness/*.py \
  scratch/run_exp409_prospective_readiness.py
```

Expected: exit code 0.

- [ ] **Step 7: Commit the preregistration before the clean-Git smoke gate**

```bash
git add docs/EXP409_PREREGISTRATION_KO.md docs/EXP409_RUNBOOK_KO.md \
  experiments/exp409_prospective_readiness/tests/test_documentation.py
git commit -m "docs: preregister Exp409 readiness smoke"
```

Verify `git status --short` is empty. A dirty tree must block the smoke rather
than being bypassed.

- [ ] **Step 8: Run the authorized representative smoke once**

First confirm the output path does not exist. Then run:

```bash
TIMESERIES_DATA_ROOT="/Users/mac/Documents/Dataset" \
python3 scratch/run_exp409_prospective_readiness.py \
  --representative-smoke \
  --dataset-limit 8 \
  --output-dir \
  "/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke"
```

Expected:

- eight dataset results;
- all four TRAIN eligibility tiers represented;
- no TEST label access;
- exact lane parity;
- candidate count at most 100;
- thirteen source statuses per dataset;
- `performance_evaluation` equals `not_evaluated`;
- no TP/FP/F1 fields in prediction artifacts.

If candidate count exceeds 100, the expected result is a blocked failure
record and no publication directory; do not increase the budget or change the
variant/alpha.

- [ ] **Step 9: Verify the internal deterministic rerun evidence**

Read `exp409_provenance_manifest.json` and independently recompute the
candidate SHA256 and canonical provenance closure:

```bash
shasum -a 256 \
  /Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke/exp409_shadow_candidates.jsonl
```

The observed candidate SHA256 must match `candidate_sha256`, the recomputed
closure must match `closure_sha256`, and
`gates.determinism` must equal `passed`. The CLI has already assembled the
batch twice in memory; do not create or overwrite a second publication.

- [ ] **Step 10: Inspect artifacts for forbidden fields**

Run:

```bash
rg -n -i \
  '"(test_labels|is_true_anomaly|actual_anomaly_positions|tp|fp|fn|f1|precision|recall)"' \
  /Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke
```

Expected: no matches in candidate, source-status, config, or canonical
provenance payloads. The Korean readiness report may mention that performance
is not evaluated but must not contain evaluated metric values.

- [ ] **Step 11: Final branch verification**

Run:

```bash
git status --short --branch
git diff --check
python3 -m unittest discover \
  -s experiments/exp409_prospective_readiness/tests \
  -p "test_*.py"
TIMESERIES_DATA_ROOT="/Users/mac/Documents/Dataset" \
python3 -m unittest discover \
  -s experiments/exp403_408/tests \
  -p "test_*.py"
```

Expected: clean branch, no whitespace errors, all Exp409 tests pass, and all
160 Exp403-408 tests pass.

Do not run the full 1,117-dataset Exp409 workload.
