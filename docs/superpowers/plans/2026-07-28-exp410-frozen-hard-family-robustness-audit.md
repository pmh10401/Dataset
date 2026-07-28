# Exp410 Frozen-Hard Family Robustness Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, read-only Exp410 audit that independently recomputes the frozen Hard result and reports dataset-balanced versus family-balanced robustness without changing predictions, lanes, or policy.

**Architecture:** A verified-input boundary loads only pinned Exp291/292 and Exp403-408 publications. A streaming SQLite scanner derives family-neutral dataset profiles, a pure aggregation layer computes family metrics and clustered bootstrap intervals, and an atomic publisher emits CSV/JSON/Korean reports plus complete provenance. The implementation uses CPU SQLite streaming, `hashlib`, and NumPy; PyTorch/MLX/MPS are assessed and recorded but are not used because the workload is not tensor-heavy and the current MPS/MLX backends are unavailable.

**Tech Stack:** Python 3.14.6, standard-library `sqlite3`/`csv`/`json`/`hashlib`, NumPy 2.3.5, `unittest`, Git

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-07-28-exp410-frozen-hard-family-robustness-audit-design.md`.
- Experiment mode is `read-only retrospective family robustness audit`.
- Dataset population is exactly `1,117`; family population is exactly `128`.
- Frozen Hard is exactly 2,218 alerts with TP/FP/FN `1,915 / 303 / 2,003`.
- Exact dataset-macro mean F1 is `0.6089825086931719`; the 6-decimal display is `0.608983`.
- Prediction SHA256 is `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`.
- Frozen label SHA256 is `b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4`.
- Database SHA256 is `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`.
- Exp291/292 lane SHA256 values are `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848` and `f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b`.
- Bootstrap resamples families, never individual datasets; iterations are `2,000` and seed is `410`.
- Family weight caps are `1`, `3`, `5`, `10`, and unlimited; all are published.
- Robustness classification uses the paired delta-F1 95% interval lower margin `-0.05` and severe point-estimate margin `-0.10`.
- Difficulty stratification is exactly `{"status":"not_evaluated","reason":"source_artifact_unavailable"}`.
- Exp410 does not create predictions, train models, select parameters, mutate lanes, or claim unseen-family, prospective, deployment, or semiconductor field performance.
- Prediction must not use TEST labels, actual anomaly positions, or historical family TEST performance.
- The full 1.6 GB DB audit is not executed during implementation or smoke verification. It requires a later explicit user authorization.
- Every future experiment design must explicitly assess PyTorch, MLX, and MPS eligibility, availability, overhead, determinism/parity, CPU fallback, and the benchmark gate for a canonical accelerated path.

---

## File Map

### New package

- `experiments/exp410_family_robustness/__init__.py`: package marker and public exports.
- `experiments/exp410_family_robustness/config.py`: immutable frozen configuration and hash/count constants.
- `experiments/exp410_family_robustness/contracts.py`: typed immutable inputs, profiles, metrics, result payloads, and canonical family parsing.
- `experiments/exp410_family_robustness/inputs.py`: manifest/hash verification, frozen publication loading, and independent per-dataset Hard metric reconstruction.
- `experiments/exp410_family_robustness/profiles.py`: read-only SQLite integrity/snapshot checks and streaming exact-overlap profiles.
- `experiments/exp410_family_robustness/aggregation.py`: family aggregation, cap sensitivity, influence analysis, strata, clustered bootstrap, and classification.
- `experiments/exp410_family_robustness/reporting.py`: deterministic row schemas, summary payload, and Korean Markdown report rendering.
- `experiments/exp410_family_robustness/publication.py`: complete staged writes, artifact hashing, manifest generation, and atomic directory publication.
- `experiments/exp410_family_robustness/runner.py`: verified smoke/full orchestration and runtime provenance.
- `scratch/run_exp410_family_robustness.py`: explicit `--representative-smoke` or `--full-audit` CLI.

### Tests

- `experiments/exp410_family_robustness/tests/test_config_contracts.py`
- `experiments/exp410_family_robustness/tests/test_inputs.py`
- `experiments/exp410_family_robustness/tests/test_profiles.py`
- `experiments/exp410_family_robustness/tests/test_aggregation.py`
- `experiments/exp410_family_robustness/tests/test_publication.py`
- `experiments/exp410_family_robustness/tests/test_runner_cli.py`

### Documentation and policy

- `docs/EXP410_PREREGISTRATION_KO.md`: frozen experiment contract and claim boundary.
- `docs/EXP410_RUNBOOK_KO.md`: environment, smoke, verification, and separately gated full-run command.
- `AGENTS.md`: persist the user instruction that every design considers PyTorch, MLX, and MPS, while retaining the existing parity/speed gate.

---

### Task 1: Persist Acceleration Policy and Freeze Exp410 Contracts

**Files:**
- Modify: `AGENTS.md`
- Create: `docs/EXP410_PREREGISTRATION_KO.md`
- Create: `experiments/exp410_family_robustness/__init__.py`
- Create: `experiments/exp410_family_robustness/config.py`
- Create: `experiments/exp410_family_robustness/contracts.py`
- Create: `experiments/exp410_family_robustness/tests/__init__.py`
- Create: `experiments/exp410_family_robustness/tests/test_config_contracts.py`

**Interfaces:**
- Consumes: no earlier implementation task
- Produces:
  - `FROZEN_CONFIG: Exp410FrozenConfig`
  - `family_from_dataset(dataset: str) -> str`
  - immutable `FrozenAuditInputs`, `DatasetProfile`, `DatasetMetric`, `FamilyMetric`, and `AuditResult` dataclasses

- [ ] **Step 1: Write failing configuration and family-contract tests**

Create `test_config_contracts.py` with:

```python
import unittest

from experiments.exp410_family_robustness.config import FROZEN_CONFIG
from experiments.exp410_family_robustness.contracts import family_from_dataset


class ConfigContractTests(unittest.TestCase):
    def test_frozen_contract_matches_preregistered_values(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp410")
        self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
        self.assertEqual(FROZEN_CONFIG.expected_families, 128)
        self.assertEqual(FROZEN_CONFIG.expected_hard, 2218)
        self.assertEqual(
            (FROZEN_CONFIG.expected_tp, FROZEN_CONFIG.expected_fp, FROZEN_CONFIG.expected_fn),
            (1915, 303, 2003),
        )
        self.assertEqual(FROZEN_CONFIG.family_weight_caps, (1, 3, 5, 10, None))
        self.assertEqual(FROZEN_CONFIG.bootstrap_iterations, 2000)
        self.assertEqual(FROZEN_CONFIG.bootstrap_seed, 410)
        self.assertFalse(FROZEN_CONFIG.operational_policy_changed)
        self.assertFalse(FROZEN_CONFIG.deployment_eligible)

    def test_family_parser_accepts_only_frozen_normal_dataset_shape(self) -> None:
        self.assertEqual(family_from_dataset("ShapesAll_normal_59"), "ShapesAll")
        for invalid in ("", "ShapesAll", "_normal_0", "ShapesAll_normal_", "ShapesAll_normal_x"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                family_from_dataset(invalid)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
python3 -m unittest experiments.exp410_family_robustness.tests.test_config_contracts -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.exp410_family_robustness'`.

- [ ] **Step 3: Implement immutable configuration and contracts**

Implement `config.py` with a frozen dataclass whose fields and defaults exactly
match Global Constraints:

```python
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Exp410FrozenConfig:
    experiment_id: str = "Exp410"
    expected_datasets: int = 1117
    expected_families: int = 128
    expected_family_size_buckets: tuple[int, int, int, int] = (1, 74, 39, 14)
    expected_hard: int = 2218
    expected_tp: int = 1915
    expected_fp: int = 303
    expected_fn: int = 2003
    expected_dataset_macro_f1: float = 0.6089825086931719
    display_dataset_macro_f1: str = "0.608983"
    prediction_sha256: str = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"
    frozen_labels_sha256: str = "b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4"
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    exp291_sha256: str = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"
    exp292_sha256: str = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"
    family_weight_caps: tuple[int | None, ...] = (1, 3, 5, 10, None)
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 410
    float_tolerance: float = 1e-12
    robustness_margin: float = -0.05
    severe_margin: float = -0.10
    operational_policy_changed: bool = False
    deployment_eligible: bool = False
    unseen_family_validated: bool = False
    prospective_validated: bool = False

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


FROZEN_CONFIG = Exp410FrozenConfig()
```

Implement `contracts.py` with the declared frozen dataclasses and this parser:

```python
def family_from_dataset(dataset: str) -> str:
    if not isinstance(dataset, str) or dataset.count("_normal_") != 1:
        raise ValueError("Exp410 dataset must contain one _normal_ marker")
    family, suffix = dataset.split("_normal_", 1)
    if not family or not suffix.isdecimal():
        raise ValueError("Exp410 dataset family and numeric class suffix are required")
    return family
```

Use these exact field shapes:

```python
@dataclass(frozen=True)
class FrozenAuditInputs:
    labels_by_dataset: Mapping[str, tuple[int, ...]]
    hard_by_dataset: Mapping[str, frozenset[int]]
    input_sha256: Mapping[str, str]

@dataclass(frozen=True)
class DatasetProfile:
    dataset: str
    family: str
    train_normal_count: int
    test_total_count: int
    test_anomaly_count: int
    test_normal_count: int
    exact_overlap_count: int
    exact_overlap_fraction: float

@dataclass(frozen=True)
class DatasetMetric:
    dataset: str
    family: str
    hard_count: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float
    f1: float

@dataclass(frozen=True)
class FamilyMetric:
    family: str
    dataset_count: int
    mean_dataset_f1: float
    mean_dataset_fp: float
    pooled_tp: int
    pooled_fp: int
    pooled_fn: int
    pooled_precision: float
    pooled_recall: float
    pooled_f1: float

@dataclass(frozen=True)
class AuditResult:
    dataset_metrics: tuple[DatasetMetric, ...]
    family_metrics: tuple[FamilyMetric, ...]
    weight_sensitivity: tuple[Mapping[str, object], ...]
    family_influence: tuple[Mapping[str, object], ...]
    stratified_metrics: tuple[Mapping[str, object], ...]
    summary: Mapping[str, object]
```

- [ ] **Step 4: Persist the all-design acceleration assessment rule**

Replace the `AGENTS.md` heading `## MLX acceleration policy` with
`## PyTorch, MLX, and MPS acceleration assessment policy`, retain the current
five MLX parity/speed requirements, and add this paragraph before them:

```markdown
Every experiment design must explicitly assess PyTorch, MLX, and MPS. Record
whether the workload is tensor-eligible, current backend availability,
transfer/compilation overhead, deterministic and numerical parity
requirements, the canonical CPU fallback, and the benchmark required before
an accelerated path becomes canonical.
```

- [ ] **Step 5: Write the Korean preregistration**

Create `docs/EXP410_PREREGISTRATION_KO.md` by translating the approved design
contract without changing values. It must contain the exact input hashes,
counts, caps, bootstrap parameters, strata, claim boundary, difficulty
`not_evaluated` record, acceleration assessment, fail-closed conditions, and
the statement that full execution requires separate explicit authorization.

- [ ] **Step 6: Run focused tests and policy/document checks**

Run:

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
python3 -m unittest experiments.exp410_family_robustness.tests.test_config_contracts -v
rg -n "PyTorch|MLX|MPS|CPU fallback|parity" AGENTS.md docs/EXP410_PREREGISTRATION_KO.md
git diff --check
```

Expected: 2 tests pass; both documents contain the acceleration requirements;
diff check exits 0.

- [ ] **Step 7: Commit Task 1**

```bash
git add AGENTS.md docs/EXP410_PREREGISTRATION_KO.md \
  experiments/exp410_family_robustness
git commit -m "docs: freeze Exp410 family audit contract"
```

---

### Task 2: Verify Frozen Inputs and Independently Recompute Hard Metrics

**Files:**
- Create: `experiments/exp410_family_robustness/inputs.py`
- Create: `experiments/exp410_family_robustness/tests/test_inputs.py`

**Interfaces:**
- Consumes: `Exp410FrozenConfig`, `FrozenAuditInputs`, `DatasetMetric`, and `family_from_dataset`
- Produces:
  - `load_verified_inputs(input_dir: Path, exp291_path: Path, exp292_path: Path, config: Exp410FrozenConfig = FROZEN_CONFIG) -> FrozenAuditInputs`
  - `recompute_dataset_metrics(inputs: FrozenAuditInputs, config: Exp410FrozenConfig = FROZEN_CONFIG) -> tuple[DatasetMetric, ...]`

- [ ] **Step 1: Write failing input-verification tests**

Create tests that use a temporary publication with two datasets:

```python
class FrozenInputTests(unittest.TestCase):
    def test_loader_verifies_hashes_and_recomputes_metrics(self) -> None:
        with TemporaryDirectory() as raw:
            fixture = write_frozen_fixture(Path(raw))
            inputs = load_verified_inputs(
                fixture.input_dir,
                fixture.exp291_path,
                fixture.exp292_path,
                config=fixture.config,
            )
            metrics = recompute_dataset_metrics(inputs, config=fixture.config)
            self.assertEqual([(row.tp, row.fp, row.fn) for row in metrics], [(1, 1, 0), (0, 0, 1)])
            self.assertEqual(metrics[0].precision, 0.5)
            self.assertEqual(metrics[0].f1, 2.0 / 3.0)
            self.assertEqual(metrics[1].precision, None)
            self.assertEqual(metrics[1].f1, 0.0)

    def test_loader_fails_closed_on_rehashed_label_change(self) -> None:
        with TemporaryDirectory() as raw:
            fixture = write_frozen_fixture(Path(raw))
            fixture.labels_path.write_text(
                fixture.labels_path.read_text().replace(",1\\n", ",0\\n", 1)
            )
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_verified_inputs(
                    fixture.input_dir,
                    fixture.exp291_path,
                    fixture.exp292_path,
                    config=fixture.config,
                )
```

Define `FrozenFixture` and `write_frozen_fixture` in the same test module.
The fixture writes:

- contiguous label rows for `Alpha_normal_0` and `Beta_normal_0`;
- Hard rows with identities `("Alpha_normal_0", 0)` and
  `("Alpha_normal_0", 1)`;
- minimal prediction, input, and publication manifests containing the exact
  fixture hashes;
- Exp291/292 JSONL files with matching Hard/Standard/Priority identities;
- `dataclasses.replace(FROZEN_CONFIG, ...)` values matching two datasets, two
  families, two Hard alerts, TP/FP/FN `1/1/1`, and the fixture hashes.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
python3 -m unittest experiments.exp410_family_robustness.tests.test_inputs -v
```

Expected: FAIL because `inputs.py` does not exist.

- [ ] **Step 3: Implement manifest-first frozen input loading**

In `inputs.py`:

- define the exact filenames from the design;
- verify SHA256 before parsing;
- verify both evaluation manifests bind every consumed publication;
- load the label CSV into contiguous `tuple[int, ...]` values per dataset;
- load Hard identities from Exp404 JSONL and reject duplicates;
- verify Exp291/292 SHA256 and lane parity;
- return sorted immutable mappings via `MappingProxyType`.

The loader must execute hash verification before `_read_csv` or `_read_jsonl`.
Use `experiments.exp403_408.artifact_io.sha256_file` and
`canonical_json_bytes`; do not import prediction, evaluation, pipeline, or
consensus helpers.

- [ ] **Step 4: Implement independent per-dataset metrics and global invariants**

Use:

```python
def _metric(dataset: str, labels: tuple[int, ...], hard: frozenset[int]) -> DatasetMetric:
    if not labels or any(value not in (0, 1) for value in labels):
        raise ValueError("Exp410 frozen labels must be nonempty binary values")
    if not all(0 <= index < len(labels) for index in hard):
        raise ValueError("Exp410 Hard index is outside frozen TEST range")
    positives = sum(labels)
    if positives == 0:
        raise ValueError("Exp410 pinned dataset must contain an anomaly")
    tp = sum(labels[index] for index in hard)
    fp = len(hard) - tp
    fn = positives - tp
    precision = None if not hard else tp / len(hard)
    recall = tp / positives
    denominator = 2 * tp + fp + fn
    f1 = 0.0 if denominator == 0 else 2 * tp / denominator
    return DatasetMetric(
        dataset=dataset,
        family=family_from_dataset(dataset),
        hard_count=len(hard),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )
```

After building all rows, require exact dataset/Hard/TP/FP/FN counts and compare
`math.fsum(row.f1 for row in rows) / len(rows)` to the configured exact mean
within `1e-12`.

- [ ] **Step 5: Run focused and Exp403-408 regression tests**

Run:

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
export TIMESERIES_DATA_ROOT="/Users/mac/Documents/Dataset"
python3 -m unittest experiments.exp410_family_robustness.tests.test_inputs -v
python3 -m unittest discover -s experiments/exp403_408/tests -p "test_*.py"
git diff --check
```

Expected: input tests pass; 160 Exp403-408 tests pass; diff check exits 0.
Run the regression command outside the sandbox if macOS semaphore access is
blocked.

- [ ] **Step 6: Commit Task 2**

```bash
git add experiments/exp410_family_robustness/inputs.py \
  experiments/exp410_family_robustness/tests/test_inputs.py
git commit -m "feat: verify Exp410 frozen Hard inputs"
```

---

### Task 3: Stream SQLite Dataset Profiles and Exact Normal Overlap

**Files:**
- Create: `experiments/exp410_family_robustness/profiles.py`
- Create: `experiments/exp410_family_robustness/tests/test_profiles.py`

**Interfaces:**
- Consumes: sorted frozen dataset identities and `Exp410FrozenConfig`
- Produces:
  - `database_snapshot(db_path: Path) -> Mapping[str, str]`
  - `scan_dataset_profiles(db_path: Path, datasets: Sequence[str], config: Exp410FrozenConfig = FROZEN_CONFIG) -> tuple[DatasetProfile, ...]`

- [ ] **Step 1: Write failing profile tests**

Create a temporary SQLite database with `datasets` and `instances` tables and
two datasets. Store float32 blobs using `np.asarray(values,
dtype="<f4").tobytes()`.

```python
class DatasetProfileTests(unittest.TestCase):
    def test_streaming_profile_counts_byte_exact_normal_overlap(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            write_profile_db(path)
            config = replace(FROZEN_CONFIG, expected_datasets=2, expected_families=2)
            rows = scan_dataset_profiles(
                path, ("Alpha_normal_0", "Beta_normal_0"), config=config
            )
            alpha = rows[0]
            self.assertEqual(alpha.train_normal_count, 2)
            self.assertEqual(alpha.test_anomaly_count, 1)
            self.assertEqual(alpha.test_normal_count, 2)
            self.assertEqual(alpha.exact_overlap_count, 1)
            self.assertEqual(alpha.exact_overlap_fraction, 0.5)

    def test_quick_check_or_snapshot_change_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            write_profile_db(path)
            with patch(
                "experiments.exp410_family_robustness.profiles.database_snapshot",
                side_effect=({"db": "a" * 64}, {"db": "b" * 64}),
            ):
                with self.assertRaisesRegex(ValueError, "snapshot changed"):
                    scan_dataset_profiles(
                        path,
                        ("Alpha_normal_0", "Beta_normal_0"),
                        config=replace(FROZEN_CONFIG, expected_datasets=2, expected_families=2),
                    )
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
python3 -m unittest experiments.exp410_family_robustness.tests.test_profiles -v
```

Expected: FAIL because `profiles.py` does not exist.

- [ ] **Step 3: Implement snapshot and integrity verification**

`database_snapshot` returns sorted SHA256 entries for the DB and existing
`-wal`/`-shm` siblings. Open SQLite with URI `file:<quoted-path>?mode=ro`;
execute `PRAGMA quick_check` and require the single result `ok`. Do not set
`immutable=1`, because the reader must observe a valid existing WAL rather
than silently ignore it.

Take one snapshot before opening and one after closing. Reject any difference.
For the real configured DB, require the DB entry to equal
`FROZEN_CONFIG.database_sha256`. Fixture configurations use
`dataclasses.replace` with the fixture DB hash.

- [ ] **Step 4: Implement per-dataset streaming overlap**

For each sorted dataset:

1. query its numeric dataset id;
2. stream normal TRAIN `values_blob` rows and build a set of
   `hashlib.sha256(blob).digest()` values;
3. stream TEST `label, values_blob` rows in `instance_index` order;
4. count anomalies directly from the instance label;
5. hash only normal TEST blobs and count membership in the TRAIN hash set;
6. release the per-dataset hash set before the next dataset.

Never read `labels_blob`. Reject unknown labels, missing datasets, zero normal
TRAIN, zero TEST anomalies, and a scanned universe different from the provided
dataset sequence.

- [ ] **Step 5: Run focused tests and compile**

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
python3 -m unittest experiments.exp410_family_robustness.tests.test_profiles -v
python3 -m py_compile experiments/exp410_family_robustness/*.py
git diff --check
```

Expected: profile tests pass; compilation and diff check exit 0.

- [ ] **Step 6: Commit Task 3**

```bash
git add experiments/exp410_family_robustness/profiles.py \
  experiments/exp410_family_robustness/tests/test_profiles.py
git commit -m "feat: profile Exp410 family and exact overlap"
```

---

### Task 4: Compute Family Metrics, Sensitivity, Influence, Strata, and Bootstrap

**Files:**
- Create: `experiments/exp410_family_robustness/aggregation.py`
- Create: `experiments/exp410_family_robustness/tests/test_aggregation.py`

**Interfaces:**
- Consumes: `tuple[DatasetMetric, ...]`, `tuple[DatasetProfile, ...]`, and `Exp410FrozenConfig`
- Produces:
  - `aggregate_family_robustness(metrics: Sequence[DatasetMetric], profiles: Sequence[DatasetProfile], config: Exp410FrozenConfig = FROZEN_CONFIG) -> AuditResult`

- [ ] **Step 1: Write failing aggregation tests**

Use three families: `Large` with three datasets, `Small` with one, and `Other`
with one. Give `Large` high F1 and the two small families low F1 so dataset and
family weighting visibly differ.

```python
class AggregationTests(unittest.TestCase):
    def test_family_macro_caps_and_influence_are_exact(self) -> None:
        metrics, profiles = aggregation_fixture()
        config = replace(
            FROZEN_CONFIG,
            expected_datasets=5,
            expected_families=3,
            expected_family_size_buckets=(2, 1, 0, 0),
            bootstrap_iterations=50,
        )
        result = aggregate_family_robustness(metrics, profiles, config=config)
        sensitivity = {row["cap"]: row["mean_f1"] for row in result.weight_sensitivity}
        family_scores = {"Large": 1.0, "Small": 0.0, "Other": 0.0}
        self.assertAlmostEqual(sensitivity["1"], sum(family_scores.values()) / 3)
        self.assertAlmostEqual(sensitivity["unlimited"], 3.0 / 5.0)
        self.assertEqual({row.family for row in result.family_metrics}, set(family_scores))
        self.assertEqual(len(result.family_influence), 3)

    def test_bootstrap_is_family_clustered_and_seeded(self) -> None:
        metrics, profiles = aggregation_fixture()
        config = replace(
            FROZEN_CONFIG,
            expected_datasets=5,
            expected_families=3,
            expected_family_size_buckets=(2, 1, 0, 0),
            bootstrap_iterations=50,
        )
        one = aggregate_family_robustness(metrics, profiles, config=config)
        two = aggregate_family_robustness(metrics, profiles, config=config)
        self.assertEqual(one.summary["bootstrap"], two.summary["bootstrap"])
        self.assertEqual(one.summary["bootstrap"]["unit"], "family")
        self.assertEqual(one.summary["bootstrap"]["seed"], 410)
```

The same fixture must cover TRAIN `<20`, `20-49`, `50+`; overlap `<50%`,
`50-<90%`, `>=90%`; and anomaly counts `1`, `2-10`, `11+`.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
python3 -m unittest experiments.exp410_family_robustness.tests.test_aggregation -v
```

Expected: FAIL because `aggregation.py` does not exist.

- [ ] **Step 3: Implement family metrics and cap sensitivity**

Group rows by `DatasetMetric.family`, require one matching profile per dataset,
and compute:

```python
def capped_mean(families: Sequence[FamilyMetric], cap: int | None) -> float:
    weights = [
        row.dataset_count if cap is None else min(row.dataset_count, cap)
        for row in families
    ]
    return math.fsum(
        weight * row.mean_dataset_f1
        for weight, row in zip(weights, families, strict=True)
    ) / sum(weights)
```

Publish cap labels `"1"`, `"3"`, `"5"`, `"10"`, and `"unlimited"`. Assert that
the unlimited value matches the direct dataset-macro F1 within `1e-12`.

- [ ] **Step 4: Implement influence and strata**

For each family, remove all its member datasets and recompute dataset-macro F1
and pooled precision. Record deltas, removed counts, dataset share, and FP
share.

Assign exact buckets:

```python
def train_bucket(n: int) -> str:
    return "<20" if n < 20 else "20-49" if n < 50 else "50+"

def overlap_bucket(value: float) -> str:
    return "<50%" if value < 0.5 else "50-<90%" if value < 0.9 else ">=90%"

def anomaly_bucket(n: int) -> str:
    return "1" if n == 1 else "2-10" if n <= 10 else "11+"
```

Within every stratum, average eligible datasets inside family first, then
average the family values. Record dataset and family denominators.

- [ ] **Step 5: Implement clustered bootstrap and classification**

Use `np.random.default_rng(config.bootstrap_seed)` to draw integer family
indices with shape `(bootstrap_iterations, family_count)`. For each draw,
include all member dataset scores, calculate family-macro and dataset-macro,
and store their difference. Use NumPy quantiles `[0.025, 0.975]`.

Classification:

```python
if point_delta < config.severe_margin:
    classification = "severe_family_imbalance"
elif interval_low < config.robustness_margin:
    classification = "sensitive"
else:
    classification = "robust"
```

Include the exact point, interval, margins, seed, iterations, and unit in the
summary. Include the fixed difficulty `not_evaluated` object.

- [ ] **Step 6: Run focused tests and full Exp410 tests**

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
python3 -m unittest experiments.exp410_family_robustness.tests.test_aggregation -v
python3 -m unittest discover -s experiments/exp410_family_robustness/tests -p "test_*.py"
git diff --check
```

Expected: all aggregation and current Exp410 tests pass; diff check exits 0.

- [ ] **Step 7: Commit Task 4**

```bash
git add experiments/exp410_family_robustness/aggregation.py \
  experiments/exp410_family_robustness/tests/test_aggregation.py
git commit -m "feat: aggregate Exp410 family robustness"
```

---

### Task 5: Render and Atomically Publish the Audit

**Files:**
- Create: `experiments/exp410_family_robustness/reporting.py`
- Create: `experiments/exp410_family_robustness/publication.py`
- Create: `experiments/exp410_family_robustness/tests/test_publication.py`

**Interfaces:**
- Consumes: `AuditResult`, `Exp410FrozenConfig`, input/runtime provenance
- Produces:
  - `render_payloads(result: AuditResult, provenance: Mapping[str, object], config: Exp410FrozenConfig = FROZEN_CONFIG) -> Mapping[str, bytes]`
  - `publish_audit(result: AuditResult, output_dir: Path, provenance: Mapping[str, object], config: Exp410FrozenConfig = FROZEN_CONFIG) -> Mapping[str, str]`

- [ ] **Step 1: Write failing publication tests**

```python
class PublicationTests(unittest.TestCase):
    def test_publication_is_complete_deterministic_and_manifested(self) -> None:
        result, config = publication_fixture()
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            one = parent / "one"
            hashes = publish_audit(result, one, provenance_fixture(), config=config)
            self.assertEqual(set(path.name for path in one.iterdir()), {
                "exp410_frozen_config.json",
                "exp410_dataset_metrics.csv",
                "exp410_family_metrics.csv",
                "exp410_weight_sensitivity.csv",
                "exp410_family_influence.csv",
                "exp410_stratified_metrics.csv",
                "exp410_summary.json",
                "EXP410_RESULT_REPORT_KO.md",
                "MANIFEST.json",
            })
            manifest = json.loads((one / "MANIFEST.json").read_text())
            self.assertEqual(manifest["artifact_sha256"], {
                name: digest for name, digest in hashes.items() if name != "MANIFEST.json"
            })

    def test_failed_publication_leaves_no_partial_directory(self) -> None:
        result, config = publication_fixture()
        with TemporaryDirectory() as raw:
            output = Path(raw) / "blocked"
            with patch(
                "experiments.exp410_family_robustness.publication._write_payload",
                side_effect=OSError("forced"),
            ):
                with self.assertRaisesRegex(OSError, "forced"):
                    publish_audit(result, output, provenance_fixture(), config=config)
            self.assertFalse(output.exists())
```

- [ ] **Step 2: Run publication tests and verify RED**

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
python3 -m unittest experiments.exp410_family_robustness.tests.test_publication -v
```

Expected: FAIL because reporting/publication modules do not exist.

- [ ] **Step 3: Implement deterministic schemas and Korean report**

`reporting.py` serializes rows sorted by:

- dataset;
- family;
- cap order `1, 3, 5, 10, unlimited`;
- influence magnitude then family;
- stratum dimension and fixed bucket order.

The summary must include:

```python
{
    "evaluation_mode": "retrospective_family_robustness_audit",
    "operational_policy_changed": False,
    "deployment_eligible": False,
    "unseen_family_validated": False,
    "prospective_validated": False,
    "difficulty_stratification": {
        "status": "not_evaluated",
        "reason": "source_artifact_unavailable",
    },
}
```

The Korean report explains dataset-macro versus family-macro, every cap,
bootstrap interval/classification, top positive/negative influence families,
strata, missing difficulty artifact, and all claim boundaries. It must not use
“현장 검증”, “전향 검증”, or “성능 개선” as positive claims.

`publish_audit` calls the single public `render_payloads` function and must not
reimplement CSV, JSON, or Markdown rendering.

- [ ] **Step 4: Implement atomic staged publication**

Follow the Exp409 publication pattern:

1. require an existing parent and nonexistent output directory;
2. render every payload in memory;
3. create `.<output>.<uuid>.staging`;
4. write bytes with exclusive creation;
5. hash and reread every staged file;
6. create `MANIFEST.json` over the other eight artifacts plus input/code/runtime
   provenance;
7. `os.replace(staging, output)`;
8. delete staging on every exception.

Do not include `MANIFEST.json` in its own `artifact_sha256`.

- [ ] **Step 5: Run publication and full Exp410 tests**

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
python3 -m unittest experiments.exp410_family_robustness.tests.test_publication -v
python3 -m unittest discover -s experiments/exp410_family_robustness/tests -p "test_*.py"
git diff --check
```

Expected: all tests pass and diff check exits 0.

- [ ] **Step 6: Commit Task 5**

```bash
git add experiments/exp410_family_robustness/reporting.py \
  experiments/exp410_family_robustness/publication.py \
  experiments/exp410_family_robustness/tests/test_publication.py
git commit -m "feat: publish Exp410 audit atomically"
```

---

### Task 6: Add Runner, Explicit CLI, Representative Smoke, and Runbook

**Files:**
- Create: `experiments/exp410_family_robustness/runner.py`
- Create: `experiments/exp410_family_robustness/tests/test_runner_cli.py`
- Create: `scratch/run_exp410_family_robustness.py`
- Create: `docs/EXP410_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: all Task 1-5 public interfaces
- Produces:
  - `run_smoke(output_dir: Path) -> Mapping[str, str]`
  - `run_full_audit(input_dir: Path, db_path: Path, output_dir: Path) -> Mapping[str, str]`
  - CLI requiring exactly one of `--representative-smoke` and `--full-audit`

- [ ] **Step 1: Write failing runner and CLI tests**

```python
class RunnerCliTests(unittest.TestCase):
    def test_cli_requires_explicit_mode_and_new_output(self) -> None:
        script = REPO_ROOT / "scratch" / "run_exp410_family_robustness.py"
        missing = subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO_ROOT, text=True, capture_output=True,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("one of the arguments", missing.stderr)

    def test_representative_smoke_publishes_without_real_full_scan(self) -> None:
        with TemporaryDirectory() as raw:
            output = Path(raw) / "smoke"
            hashes = run_smoke(output)
            summary = json.loads((output / "exp410_summary.json").read_text())
            self.assertTrue(summary["smoke"])
            self.assertEqual(summary["difficulty_stratification"]["status"], "not_evaluated")
            self.assertFalse(summary["operational_policy_changed"])
            self.assertIn("MANIFEST.json", hashes)

    def test_full_runner_is_not_called_by_smoke(self) -> None:
        with TemporaryDirectory() as raw, patch(
            "experiments.exp410_family_robustness.runner.scan_dataset_profiles",
            side_effect=AssertionError("real profile scan forbidden in smoke"),
        ):
            run_smoke(Path(raw) / "smoke")
```

- [ ] **Step 2: Run runner/CLI tests and verify RED**

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
python3 -m unittest experiments.exp410_family_robustness.tests.test_runner_cli -v
```

Expected: FAIL because runner and CLI do not exist.

- [ ] **Step 3: Implement a deterministic fixture-backed smoke**

`run_smoke` creates a temporary, synthetic 32-dataset publication and SQLite
fixture inside the output parent. The fixture must cover:

- family sizes `1`, `2-5`, `6-20`, and `21+` using four families with exactly
  `1`, `3`, `7`, and `21` datasets;
- TRAIN buckets `<20`, `20-49`, `50+`;
- overlap buckets `<50%`, `50-<90%`, `>=90%`;
- anomaly buckets `1`, `2-10`, `11+`;
- at least one no-alert dataset and one dataset with both TP and FP.

The smoke uses fixture-specific `dataclasses.replace(FROZEN_CONFIG, ...)`
values with `expected_datasets=32`, `expected_families=4`, and
`expected_family_size_buckets=(1, 1, 1, 1)`, plus matching fixture
Hard/TP/FP/FN/F1 and SHA256 values. It retains caps, seed, iterations, margins,
and claim flags. It exercises the same input, profile, aggregation, reporting,
and publication components. It must never open the real 1.6 GB DB.

- [ ] **Step 4: Implement the full runner without executing it**

`run_full_audit`:

1. requires a clean Git worktree;
2. calls `load_verified_inputs`;
3. independently recomputes and verifies metrics;
4. records preflight input hashes;
5. calls `scan_dataset_profiles` for the full sorted universe;
6. calls `aggregate_family_robustness`;
7. records Python/NumPy/PyTorch versions, MPS built/available, MLX
   installed/not-installed, platform, Git commit, config SHA256, code SHA256,
   and CPU canonical-path reason;
8. atomically publishes.

It exposes no lane writer, prediction writer, model, source selector, or
threshold parameter.

- [ ] **Step 5: Implement the explicit CLI**

Use a required mutually exclusive group:

```python
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--representative-smoke", action="store_true")
mode.add_argument("--full-audit", action="store_true")
```

Require `--output-dir NEW_PATH` for both modes. Require `--input-dir` and
`--db-path` only with `--full-audit`; reject them in smoke mode. Reject an
existing output directory. The CLI must not provide dataset-limit, cap, seed,
iteration, margin, threshold, alpha, source, or budget overrides.

- [ ] **Step 6: Write the Korean runbook**

`docs/EXP410_RUNBOOK_KO.md` must include:

- the exact venv and `TIMESERIES_DATA_ROOT` activation;
- unit/regression test commands;
- smoke command writing to a new output directory;
- artifact/schema/hash verification commands;
- current acceleration assessment and CPU reason;
- a clearly separated full command headed “사용자의 별도 명시적 승인 후에만”;
- the full command:

```bash
python3 scratch/run_exp410_family_robustness.py \
  --full-audit \
  --input-dir "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation" \
  --db-path "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --output-dir "/Users/mac/Documents/Dataset/exp410_family_robustness"
```

- [ ] **Step 7: Run smoke, tests, compilation, and regression verification**

Use a new temporary smoke output outside Git. Do not run `--full-audit`.

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
export TIMESERIES_DATA_ROOT="/Users/mac/Documents/Dataset"
python3 -m unittest discover -s experiments/exp410_family_robustness/tests -p "test_*.py"
python3 -m unittest discover -s experiments/exp403_408/tests -p "test_*.py"
python3 -m unittest discover -s experiments/exp409_prospective_readiness/tests -p "test_*.py"
python3 -m py_compile experiments/exp410_family_robustness/*.py \
  scratch/run_exp410_family_robustness.py
smoke_dir=$(mktemp -d "/private/tmp/exp410-smoke-parent.XXXXXX")
python3 scratch/run_exp410_family_robustness.py \
  --representative-smoke \
  --output-dir "$smoke_dir/publication"
python3 -m json.tool "$smoke_dir/publication/exp410_summary.json"
python3 -m json.tool "$smoke_dir/publication/MANIFEST.json"
git diff --check
```

Expected:

- all Exp410 tests pass;
- Exp403-408 regression reports 160 passing tests;
- Exp409 regression reports 47 passing tests;
- compilation passes;
- smoke publishes all nine artifacts;
- summary contains `smoke: true` and all claim flags are false;
- manifest hashes match the eight non-manifest artifacts;
- no full real-DB scan is executed.

- [ ] **Step 8: Commit Task 6**

```bash
git add experiments/exp410_family_robustness/runner.py \
  experiments/exp410_family_robustness/tests/test_runner_cli.py \
  scratch/run_exp410_family_robustness.py \
  docs/EXP410_RUNBOOK_KO.md
git commit -m "feat: add Exp410 smoke and audit runner"
```

---

## Final Verification Before Full-Run Decision

Run fresh:

```bash
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
export TIMESERIES_DATA_ROOT="/Users/mac/Documents/Dataset"
python3 -m unittest discover -s experiments/exp410_family_robustness/tests -p "test_*.py"
python3 -m unittest discover -s experiments/exp403_408/tests -p "test_*.py"
python3 -m unittest discover -s experiments/exp409_prospective_readiness/tests -p "test_*.py"
python3 -m py_compile experiments/exp410_family_robustness/*.py \
  scratch/run_exp410_family_robustness.py
git diff --check
git status --short --branch
```

Then verify the most recent smoke:

```bash
exp410_verify_parent=$(mktemp -d "/private/tmp/exp410-final-smoke.XXXXXX")
python3 scratch/run_exp410_family_robustness.py \
  --representative-smoke \
  --output-dir "$exp410_verify_parent/publication"
python3 -m json.tool "$exp410_verify_parent/publication/exp410_summary.json"
python3 -m json.tool "$exp410_verify_parent/publication/MANIFEST.json"
shasum -a 256 "$exp410_verify_parent/publication"/*
python3 - "$exp410_verify_parent/publication" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text())
for name, expected in manifest["artifact_sha256"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA256 mismatch: {name}")
print("Exp410 smoke manifest hashes: OK")
PY
```

Expected:

- all test suites and compilation pass;
- worktree is clean after six intentional commits;
- smoke artifacts and manifest hashes verify;
- no `exp410_family_robustness` full output directory has been created;
- no full 1.6 GB scan has run.

Stop and report the verified implementation and smoke evidence. Ask for
explicit authorization before executing the full command in the runbook.
