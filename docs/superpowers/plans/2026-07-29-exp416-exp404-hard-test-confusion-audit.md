# Exp416 Exp404 Hard TEST Confusion Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 동결된 Exp404 Hard 좌표 2,218개를 그 이후에만 여는 frozen TEST label 195,900개와 대조하여, 1,117개 모집단의 실제 포인트 단위 혼동행렬과 dataset-balanced 보조 지표를 별도 Exp416 산출물로 게시한다.

**Architecture:** Prediction 검증과 label loading을 타입과 함수 경계로 분리하여 label loader가 `FrozenPrediction` 없이는 호출될 수 없게 한다. 순수 계산 계층이 micro 및 dataset별 TP/FP/FN/TN과 metric을 만들고, reporting 계층은 결정론적인 세 payload를 렌더링하며, publication 계층은 manifest를 더해 새 디렉터리로 원자 게시한다. 기존 Exp403-408 및 Exp414 코드는 수정하지 않는다.

**Tech Stack:** Python 3.14, standard-library `csv`/`json`/`hashlib`, frozen dataclasses, `unittest`, canonical JSON/SHA256, atomic filesystem publication

## Global Constraints

- Exp404 prediction 경로는 `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl`, SHA256은 `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`이다.
- Exp404 manifest 경로는 `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json`, SHA256은 `41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8`이다.
- Frozen TEST label 경로는 `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp403_408_frozen_test_labels.csv`, SHA256은 `b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4`이다.
- Source DB reference SHA256은 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`이다. Exp416은 DB에서 label을 다시 읽지 않는다.
- Prediction은 정확히 2,218개 고유 좌표와 888개 prediction 보유 dataset을 가진다.
- Manifest와 frozen label은 정확히 1,117개 dataset을 가진다.
- Frozen label은 정확히 195,900개 고유 좌표이며 label은 exact integer `0` 또는 `1`이다.
- Manifest `frozen_hard_lanes`의 비어 있지 않은 좌표 집합과 prediction JSONL 좌표 집합은 정확히 같아야 한다.
- Prediction identity를 동결하기 전에는 TEST label 파일을 열지 않는다.
- Positive class는 실제 이상이며, predicted positive는 Exp404 Hard 좌표다.
- Undefined metric은 `null`/빈 CSV field로 남기며 0으로 대체하지 않는다.
- Macro metric은 정의된 dataset만 평균하고 `defined_dataset_count`를 함께 기록한다.
- TEST 결과를 보고 alpha, threshold, source, budget 또는 lane을 재선택하지 않는다.
- Exp415 prediction을 생성하지 않고 운영 정책을 변경하지 않는다.
- 출력 디렉터리는 `/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit`이며 기존 경로를 덮어쓰지 않는다.
- 최종 산출물은 `exp416_confusion_matrix.json`, `exp416_dataset_metrics.csv`, `EXP416_RESULT_REPORT_KO.md`, `MANIFEST.json` 정확히 네 파일이다.

---

## File Structure

- `docs/EXP416_PREREGISTRATION_KO.md`: 입력 SHA, prediction-first 순서, 혼동행렬 정의, 금지사항과 산출물을 실행 전에 동결한다.
- `experiments/exp416_exp404_hard_test_confusion_audit/__init__.py`: 공개 API를 최소 범위로 노출한다.
- `experiments/exp416_exp404_hard_test_confusion_audit/config.py`: Exp416 frozen constants와 exact-default 검증을 담당한다.
- `experiments/exp416_exp404_hard_test_confusion_audit/contracts.py`: prediction, label, confusion count, metric과 audit result 타입을 정의한다.
- `experiments/exp416_exp404_hard_test_confusion_audit/inputs.py`: prediction-first freeze, 별도 TEST label load와 snapshot 재검증을 담당한다.
- `experiments/exp416_exp404_hard_test_confusion_audit/audit.py`: micro/dataset 혼동행렬과 macro metric을 순수 계산한다.
- `experiments/exp416_exp404_hard_test_confusion_audit/reporting.py`: 세 non-manifest payload를 결정론적으로 렌더링하고 다시 검증한다.
- `experiments/exp416_exp404_hard_test_confusion_audit/publication.py`: 세 payload와 manifest를 exact-file atomic publication한다.
- `experiments/exp416_exp404_hard_test_confusion_audit/tests/`: 각 계층의 TDD와 실패 경로 테스트를 둔다.
- `scratch/run_exp416_exp404_hard_test_confusion_audit.py`: one-shot real audit CLI와 orchestration을 제공한다.
- `docs/EXP416_RUNBOOK_KO.md`: clean commit 확인, 실행과 산출물 검증 명령을 기록한다.

---

### Task 1: Frozen Governance, Configuration, and Contracts

**Files:**
- Create: `docs/EXP416_PREREGISTRATION_KO.md`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/__init__.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/config.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/contracts.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/tests/__init__.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/tests/test_config.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/tests/test_contracts.py`

**Interfaces:**
- Consumes: approved design `docs/superpowers/specs/2026-07-29-exp416-exp404-hard-test-confusion-audit-design.md`.
- Produces: `Exp416FrozenConfig`, `FROZEN_CONFIG`, `FrozenPrediction`, `FrozenLabels`, `LabelRow`, `ConfusionCounts`, `ConfusionMetrics`, `MacroMetric`, `DatasetAuditRow`, `AuditResult`.

- [ ] **Step 1: Write frozen-config and contract tests**

Create tests with exact values and type rejection:

```python
from dataclasses import replace
import unittest

from experiments.exp416_exp404_hard_test_confusion_audit.config import (
    FROZEN_CONFIG,
)
from experiments.exp416_exp404_hard_test_confusion_audit.contracts import (
    ConfusionCounts,
    LabelRow,
)


class ConfigTests(unittest.TestCase):
    def test_frozen_values_match_preregistration(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp416")
        self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
        self.assertEqual(FROZEN_CONFIG.expected_prediction_datasets, 888)
        self.assertEqual(FROZEN_CONFIG.expected_prediction_coordinates, 2218)
        self.assertEqual(FROZEN_CONFIG.expected_label_rows, 195900)
        self.assertEqual(
            FROZEN_CONFIG.prediction_sha256,
            "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
        )
        FROZEN_CONFIG.validate()

    def test_mutated_frozen_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(FROZEN_CONFIG, expected_prediction_coordinates=2217).validate()


class ContractTests(unittest.TestCase):
    def test_label_requires_exact_binary_integer(self) -> None:
        self.assertEqual(LabelRow("d", 1, 1).is_true_anomaly, 1)
        for invalid in (True, False, 0.0, 1.0, "1", -1, 2):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    LabelRow("d", 1, invalid)  # type: ignore[arg-type]

    def test_counts_require_exact_nonnegative_integers(self) -> None:
        self.assertEqual(ConfusionCounts(tp=1, fp=2, fn=3, tn=4).total, 10)
        with self.assertRaises(ValueError):
            ConfusionCounts(tp=True, fp=0, fn=0, tn=0)
        with self.assertRaises(ValueError):
            ConfusionCounts(tp=-1, fp=0, fn=0, tn=0)
```

- [ ] **Step 2: Run the focused tests and verify the package is absent**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_config \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_contracts -v
```

Expected: FAIL with `ModuleNotFoundError` for the Exp416 package.

- [ ] **Step 3: Implement exact frozen config**

Implement a frozen dataclass whose `validate()` compares exact type and value
against a fresh default instance:

```python
@dataclass(frozen=True)
class Exp416FrozenConfig:
    experiment_id: str = "Exp416"
    expected_datasets: int = 1117
    expected_prediction_datasets: int = 888
    expected_prediction_coordinates: int = 2218
    expected_label_rows: int = 195900
    prediction_sha256: str = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"
    manifest_sha256: str = "41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8"
    labels_sha256: str = "b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4"
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"

    def validate(self) -> None:
        expected = type(self)()
        for field in fields(self):
            actual = getattr(self, field.name)
            wanted = getattr(expected, field.name)
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError(f"Exp416 frozen config mismatch: {field.name}")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


FROZEN_CONFIG = Exp416FrozenConfig()
```

- [ ] **Step 4: Implement immutable contracts**

Define and validate these exact interfaces:

```python
Coordinate = tuple[str, int]

@dataclass(frozen=True)
class FrozenPrediction:
    coordinates: tuple[Coordinate, ...]
    dataset_scope: tuple[str, ...]
    prediction_dataset_count: int
    identity_sha256: str
    prediction_sha256: str
    manifest_sha256: str

@dataclass(frozen=True)
class LabelRow:
    dataset: str
    candidate_index: int
    is_true_anomaly: int

@dataclass(frozen=True)
class FrozenLabels:
    rows: tuple[LabelRow, ...]
    dataset_scope: tuple[str, ...]
    labels_sha256: str

@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

@dataclass(frozen=True)
class ConfusionMetrics:
    precision: float | None
    recall: float | None
    specificity: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    f1: float | None

@dataclass(frozen=True)
class MacroMetric:
    value: float | None
    defined_dataset_count: int

@dataclass(frozen=True)
class DatasetAuditRow:
    dataset: str
    counts: ConfusionCounts
    metrics: ConfusionMetrics

@dataclass(frozen=True)
class AuditResult:
    micro_counts: ConfusionCounts
    micro_metrics: ConfusionMetrics
    dataset_rows: tuple[DatasetAuditRow, ...]
    macro_metrics: Mapping[str, MacroMetric]
    label_rows: int
    prediction_coordinates: int
```

Every contract `__post_init__` must reject empty dataset names, negative
indices, bool-as-int, non-finite metric values, unsorted/duplicate coordinates,
and inconsistent count totals.

- [ ] **Step 5: Write the preregistration before any real label load**

`docs/EXP416_PREREGISTRATION_KO.md` must include:

- all four frozen input hashes;
- 2,218 / 888 / 195,900 / 1,117 counts;
- the prediction-first six-step freeze sequence;
- TP/FP/FN/TN and six metric formulas;
- macro undefined-value policy;
- no alpha/threshold/source/budget/lane reselection;
- `Exp415 prediction not created`;
- exact four outputs and no-overwrite atomic publication.

No real command in Tasks 1-4 may open
`exp403_408_frozen_test_labels.csv`.

- [ ] **Step 6: Run focused tests and documentation guards**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_config \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_contracts -v
rg -n "437dc637|41a91341|b483ba00|195,900|2,218|TEST-informed" \
  docs/EXP416_PREREGISTRATION_KO.md
git diff --check
```

Expected: tests pass, all frozen guards are present, diff check is silent.

- [ ] **Step 7: Commit the preregistered boundary**

```bash
git add docs/EXP416_PREREGISTRATION_KO.md \
  experiments/exp416_exp404_hard_test_confusion_audit
git commit -m "docs: preregister Exp416 TEST confusion audit"
```

---

### Task 2: Prediction-First Input Freeze and Separate Label Loader

**Files:**
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/inputs.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/tests/test_inputs.py`

**Interfaces:**
- Consumes: `FrozenPrediction`, `FrozenLabels`, `LabelRow`, `Exp416FrozenConfig`, `FROZEN_CONFIG`.
- Produces:
  - `verify_database_reference(database_path: Path, *, config: Exp416FrozenConfig = FROZEN_CONFIG) -> str`
  - `freeze_prediction_inputs(prediction_path: Path, manifest_path: Path, *, config: Exp416FrozenConfig = FROZEN_CONFIG) -> FrozenPrediction`
  - `load_frozen_test_labels(labels_path: Path, frozen_prediction: FrozenPrediction, *, config: Exp416FrozenConfig = FROZEN_CONFIG) -> FrozenLabels`
  - `verify_input_snapshots(database_path: Path, prediction_path: Path, manifest_path: Path, labels_path: Path, database_sha256: str, frozen_prediction: FrozenPrediction, frozen_labels: FrozenLabels, *, config: Exp416FrozenConfig = FROZEN_CONFIG) -> None`

- [ ] **Step 1: Write prediction-freeze failure tests**

Use synthetic temp files and patch only frozen expected values:

```python
from collections.abc import Iterator
from contextlib import contextmanager

@contextmanager
def fixture_config_validation() -> Iterator[None]:
    with mock.patch.object(
        Exp416FrozenConfig,
        "validate",
        autospec=True,
        return_value=None,
    ):
        yield

def test_freeze_rejects_duplicate_and_manifest_mismatch(self) -> None:
    rows = [
        prediction_row("A", 3),
        prediction_row("A", 3),
    ]
    prediction, manifest, config = write_inputs(
        rows,
        frozen_hard_lanes={"A": [3], "B": []},
        expected_datasets=2,
        expected_prediction_datasets=1,
        expected_prediction_coordinates=2,
    )
    with fixture_config_validation():
        with self.assertRaisesRegex(ValueError, "duplicate"):
            freeze_prediction_inputs(prediction, manifest, config=config)

def test_prediction_forbids_label_or_metric_fields(self) -> None:
    for field, value in (
        ("contains_test_labels", True),
        ("contains_evaluation_metrics", True),
    ):
        with self.subTest(field=field):
            prediction, manifest, config = write_single_prediction(
                row_overrides={field: value}
            )
            with fixture_config_validation():
                with self.assertRaises(ValueError):
                    freeze_prediction_inputs(prediction, manifest, config=config)
```

Also test:

- wrong database-reference SHA without opening SQLite tables;
- wrong prediction and manifest SHA independently;
- bool/negative candidate index;
- manifest scope duplicate or wrong count;
- lane coordinate absent from JSONL;
- JSONL coordinate absent from lanes;
- second-pass hash mutation after parse.

- [ ] **Step 2: Write label-loader ordering and validation tests**

The loader requires a real `FrozenPrediction` positional argument. Verify the
label file is not opened when that object is invalid:

```python
def test_labels_cannot_load_without_frozen_prediction(self) -> None:
    labels = self.root / "labels.csv"
    labels.write_text(
        "dataset,candidate_index,is_true_anomaly\nA,0,1\n",
        encoding="utf-8",
    )
    with mock.patch.object(Path, "open", side_effect=AssertionError("opened")):
        with self.assertRaises((TypeError, ValueError)):
            load_frozen_test_labels(
                labels,
                object(),  # type: ignore[arg-type]
                config=self.config,
            )

def test_labels_reject_bool_like_and_duplicate_coordinates(self) -> None:
    frozen = valid_frozen_prediction(("A",))
    for payload in (
        "dataset,candidate_index,is_true_anomaly\nA,0,true\n",
        "dataset,candidate_index,is_true_anomaly\nA,0,1\nA,0,0\n",
    ):
        with self.subTest(payload=payload):
            labels, config = write_labels(payload, expected_rows=1)
            with fixture_config_validation():
                with self.assertRaises(ValueError):
                    load_frozen_test_labels(labels, frozen, config=config)
```

Also test exact header, negative index, blank dataset, row count, dataset scope,
prediction-coordinate coverage and before/after label hash mutation.

- [ ] **Step 3: Run input tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_inputs -v
```

Expected: FAIL because `inputs.py` and its functions do not exist.

- [ ] **Step 4: Implement canonical prediction identity**

Use canonical JSON bytes for the sorted coordinate array:

```python
def _coordinate_identity(coordinates: Sequence[Coordinate]) -> str:
    payload = canonical_json_bytes(
        [[dataset, index] for dataset, index in coordinates]
    )
    return sha256(payload).hexdigest()

def _exact_index(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Exp416 {field} must be a nonnegative exact integer")
    return value
```

`freeze_prediction_inputs` must:

1. verify both frozen SHA values before parse;
2. parse JSONL one row at a time;
3. require exact `false` TEST/evaluation flags;
4. reject duplicate coordinates;
5. parse manifest with exact top-level flags/counts;
6. build all lane coordinates and compare exact set equality;
7. compare coordinate/dataset/population counts;
8. re-hash both source files after parse;
9. return sorted immutable coordinates and scope.

`verify_database_reference` streams the DB file only through `sha256_file`,
compares the observed digest with `config.database_sha256`, and returns the
observed digest. It must not connect to SQLite or read TEST tables.

- [ ] **Step 5: Implement separate frozen label loader**

Parse `csv.DictReader` with exact headers and literal string values:

```python
_LABEL_FIELDS = ("dataset", "candidate_index", "is_true_anomaly")

def _parse_binary_label(raw: str) -> int:
    if raw not in {"0", "1"}:
        raise ValueError("Exp416 TEST label must be literal 0 or 1")
    return int(raw)
```

Before opening `labels_path`, require:

```python
if not isinstance(frozen_prediction, FrozenPrediction):
    raise ValueError("Exp416 labels require a frozen prediction")
```

Verify label SHA before and after streaming, exact row/dataset counts, unique
coordinates, exact scope equality and `prediction_coordinates <= label_universe`.
Return rows sorted by `(dataset, candidate_index)`.

- [ ] **Step 6: Implement final snapshot verification**

`verify_input_snapshots` must recompute the DB reference and all three artifact
SHA values and also verify:

```python
if database_sha256 != config.database_sha256:
    raise ValueError("Exp416 database reference snapshot mismatch")
if frozen_prediction.prediction_sha256 != config.prediction_sha256:
    raise ValueError("Exp416 frozen prediction snapshot mismatch")
if frozen_prediction.manifest_sha256 != config.manifest_sha256:
    raise ValueError("Exp416 frozen manifest snapshot mismatch")
if frozen_labels.labels_sha256 != config.labels_sha256:
    raise ValueError("Exp416 frozen label snapshot mismatch")
```

It must not rebuild or mutate the frozen objects.

- [ ] **Step 7: Run focused and package tests**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_inputs -v
.venv/bin/python -m unittest discover \
  -s experiments/exp416_exp404_hard_test_confusion_audit/tests \
  -p "test_*.py"
git diff --check
```

Expected: all tests pass and diff check is silent.

- [ ] **Step 8: Commit input isolation**

```bash
git add experiments/exp416_exp404_hard_test_confusion_audit
git commit -m "feat: freeze Exp416 prediction before TEST labels"
```

---

### Task 3: Exact Micro and Dataset-Balanced Confusion Audit

**Files:**
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/audit.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/tests/test_audit.py`

**Interfaces:**
- Consumes: `FrozenPrediction`, `FrozenLabels`, all confusion contracts.
- Produces:
  - `metrics_from_counts(counts: ConfusionCounts) -> ConfusionMetrics`
  - `compute_confusion_audit(frozen_prediction: FrozenPrediction, frozen_labels: FrozenLabels) -> AuditResult`

- [ ] **Step 1: Write a hand-calculated micro matrix test**

```python
def test_computes_exact_micro_confusion_matrix(self) -> None:
    prediction = frozen_prediction(
        coordinates=(("A", 0), ("A", 2), ("B", 1)),
        dataset_scope=("A", "B"),
    )
    labels = frozen_labels(
        (
            LabelRow("A", 0, 1),  # TP
            LabelRow("A", 1, 1),  # FN
            LabelRow("A", 2, 0),  # FP
            LabelRow("B", 0, 0),  # TN
            LabelRow("B", 1, 1),  # TP
            LabelRow("B", 2, 0),  # TN
        ),
        dataset_scope=("A", "B"),
    )
    result = compute_confusion_audit(prediction, labels)
    self.assertEqual(
        result.micro_counts,
        ConfusionCounts(tp=2, fp=1, fn=1, tn=2),
    )
    self.assertAlmostEqual(result.micro_metrics.precision, 2 / 3)
    self.assertAlmostEqual(result.micro_metrics.recall, 2 / 3)
    self.assertAlmostEqual(result.micro_metrics.specificity, 2 / 3)
    self.assertAlmostEqual(result.micro_metrics.f1, 2 / 3)
```

- [ ] **Step 2: Write undefined and macro-policy tests**

```python
def test_undefined_metrics_remain_none(self) -> None:
    metrics = metrics_from_counts(ConfusionCounts(tp=0, fp=0, fn=0, tn=5))
    self.assertIsNone(metrics.precision)
    self.assertIsNone(metrics.recall)
    self.assertEqual(metrics.specificity, 1.0)
    self.assertEqual(metrics.false_positive_rate, 0.0)
    self.assertIsNone(metrics.false_negative_rate)
    self.assertIsNone(metrics.f1)

def test_macro_uses_only_defined_dataset_values(self) -> None:
    result = compute_confusion_audit(
        prediction_with_no_coordinates(("A", "B")),
        frozen_labels(
            (
                LabelRow("A", 0, 1),
                LabelRow("B", 0, 0),
            ),
            dataset_scope=("A", "B"),
        ),
    )
    self.assertEqual(result.macro_metrics["recall"].defined_dataset_count, 1)
    self.assertEqual(result.macro_metrics["recall"].value, 0.0)
    self.assertEqual(
        result.macro_metrics["specificity"].defined_dataset_count,
        1,
    )
    self.assertEqual(result.macro_metrics["specificity"].value, 1.0)
```

Also test input scope mismatch, prediction coordinate absent from labels,
dataset sort order and micro count equals sum of dataset counts.

- [ ] **Step 3: Run audit tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_audit -v
```

Expected: FAIL because `audit.py` is missing.

- [ ] **Step 4: Implement count-derived metrics**

Use one guarded ratio helper and the exact formulas:

```python
def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None

def metrics_from_counts(counts: ConfusionCounts) -> ConfusionMetrics:
    return ConfusionMetrics(
        precision=_ratio(counts.tp, counts.tp + counts.fp),
        recall=_ratio(counts.tp, counts.tp + counts.fn),
        specificity=_ratio(counts.tn, counts.tn + counts.fp),
        false_positive_rate=_ratio(counts.fp, counts.fp + counts.tn),
        false_negative_rate=_ratio(counts.fn, counts.fn + counts.tp),
        f1=_ratio(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn),
    )
```

- [ ] **Step 5: Implement exact per-coordinate counting**

For each label row:

```python
predicted = (row.dataset, row.candidate_index) in predicted_coordinates
actual = row.is_true_anomaly == 1
if predicted and actual:
    counts["tp"] += 1
elif predicted:
    counts["fp"] += 1
elif actual:
    counts["fn"] += 1
else:
    counts["tn"] += 1
```

Build one `DatasetAuditRow` for every sorted dataset, then sum those rows into
micro counts. Fail if:

- dataset scopes differ;
- a prediction coordinate is outside label universe;
- dataset rows are not 1:1 with scope;
- sum of dataset counts differs from micro counts;
- total differs from label row count.

Macro calculation must iterate the six named metric fields and average only
non-`None` finite values, recording exact defined count.

- [ ] **Step 6: Run focused and package tests**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_audit -v
.venv/bin/python -m unittest discover \
  -s experiments/exp416_exp404_hard_test_confusion_audit/tests \
  -p "test_*.py"
git diff --check
```

Expected: all tests pass.

- [ ] **Step 7: Commit the pure audit engine**

```bash
git add experiments/exp416_exp404_hard_test_confusion_audit
git commit -m "feat: compute Exp416 TEST confusion matrix"
```

---

### Task 4: Deterministic Reporting and Atomic Publication

**Files:**
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/reporting.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/publication.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/tests/test_reporting.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/tests/test_publication.py`

**Interfaces:**
- Consumes: `AuditResult`, `FrozenPrediction`, `FrozenLabels`, `Exp416FrozenConfig`.
- Produces:
  - `render_payloads(result: AuditResult, frozen_prediction: FrozenPrediction, frozen_labels: FrozenLabels, database_sha256: str, git_commit: str, *, config: Exp416FrozenConfig = FROZEN_CONFIG) -> Mapping[str, bytes]`
  - `verify_rendered_payloads(payloads: Mapping[str, bytes], *, config: Exp416FrozenConfig = FROZEN_CONFIG) -> None`
  - `publish_payloads(payloads: Mapping[str, bytes], output_dir: Path, provenance: Mapping[str, object], *, pre_publish_verify: Callable[[], None] | None = None, config: Exp416FrozenConfig = FROZEN_CONFIG) -> Mapping[str, str]`

- [ ] **Step 1: Write exact rendering tests**

Assert exact three-file coverage and count-derived values:

```python
@contextmanager
def fixture_config_validation() -> Iterator[None]:
    with mock.patch.object(
        Exp416FrozenConfig,
        "validate",
        autospec=True,
        return_value=None,
    ):
        yield

def test_renders_exact_payloads_and_governance_flags(self) -> None:
    with fixture_config_validation():
        payloads = render_payloads(
            hand_audit_result(),
            hand_frozen_prediction(),
            hand_frozen_labels(),
            self.config.database_sha256,
            "a" * 40,
            config=self.config,
        )
    self.assertEqual(
        set(payloads),
        {
            "exp416_confusion_matrix.json",
            "exp416_dataset_metrics.csv",
            "EXP416_RESULT_REPORT_KO.md",
        },
    )
    summary = json.loads(payloads["exp416_confusion_matrix.json"])
    self.assertEqual(summary["micro"]["counts"], {
        "tp": 2, "fp": 1, "fn": 1, "tn": 2,
    })
    self.assertFalse(summary["prediction_modified"])
    self.assertTrue(summary["test_labels_loaded_after_prediction_freeze"])
    self.assertFalse(summary["test_informed_reselection"])
    self.assertFalse(summary["operational_policy_changed"])
    self.assertFalse(summary["exp415_prediction_created"])
```

CSV tests must assert fixed header, sorted 1,117-style dataset ordering in a
small fixture, and empty cells for undefined values. Markdown must include the
integer 2×2 matrix, metric values and no-reselection boundary.

- [ ] **Step 2: Write reporting tamper tests**

For each mutation, `verify_rendered_payloads` must reject:

- one count changed without metric change;
- total count not equal to label rows;
- macro defined count greater than dataset count;
- NaN/Infinity or numeric string;
- missing/extra payload;
- governance flag changed to true;
- CSV dataset order or count mismatch.

- [ ] **Step 3: Write atomic publication failure tests**

Use a new temp output path and assert no final directory for:

```python
def test_callback_mutation_leaves_no_output(self) -> None:
    output = self.root / "publication"
    def mutate() -> None:
        staging = next(self.root.glob(".publication.*.staging"))
        (staging / "exp416_confusion_matrix.json").write_bytes(b"changed")
    with self.assertRaises(ValueError):
        publish_payloads(
            valid_payloads(),
            output,
            valid_provenance(),
            pre_publish_verify=mutate,
            config=self.config,
        )
    self.assertFalse(output.exists())
```

Also cover extra file, staging file symlink, staging-root symlink substitution,
output appearing during staging, parent fsync failure rollback and existing
output no-clobber.

- [ ] **Step 4: Run reporting/publication tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_reporting \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_publication -v
```

Expected: FAIL because reporting and publication modules are absent.

- [ ] **Step 5: Implement deterministic payload rendering**

Use constants:

```python
SUMMARY_FILE = "exp416_confusion_matrix.json"
DATASET_FILE = "exp416_dataset_metrics.csv"
REPORT_FILE = "EXP416_RESULT_REPORT_KO.md"
MANIFEST_FILE = "MANIFEST.json"
_PAYLOAD_FILES = {SUMMARY_FILE, DATASET_FILE, REPORT_FILE}
_DATASET_FIELDS = (
    "dataset", "tp", "fp", "fn", "tn",
    "precision", "recall", "specificity",
    "false_positive_rate", "false_negative_rate", "f1",
)
```

Summary JSON must include:

```python
summary = {
    "experiment_id": "Exp416",
    "scope": "Exp404 frozen Hard point-level TEST audit",
    "git_commit": git_commit,
    "prediction_identity_sha256": frozen_prediction.identity_sha256,
    "input_sha256": {
        "database_reference": database_sha256,
        "prediction": frozen_prediction.prediction_sha256,
        "manifest": frozen_prediction.manifest_sha256,
        "frozen_test_labels": frozen_labels.labels_sha256,
    },
    "dataset_count": len(result.dataset_rows),
    "prediction_dataset_count": frozen_prediction.prediction_dataset_count,
    "prediction_coordinate_count": result.prediction_coordinates,
    "frozen_label_row_count": result.label_rows,
    "micro": {
        "counts": asdict(result.micro_counts),
        "metrics": asdict(result.micro_metrics),
    },
    "macro": {
        name: asdict(metric)
        for name, metric in sorted(result.macro_metrics.items())
    },
    "prediction_modified": False,
    "test_labels_loaded_after_prediction_freeze": True,
    "test_informed_reselection": False,
    "operational_policy_changed": False,
    "exp415_prediction_created": False,
}
```

Serialize JSON with `allow_nan=False`, sorted keys and compact separators.
Serialize CSV with `lineterminator="\n"` and `""` for `None`. Render a Korean
Markdown report from the same summary object without recalculating counts.

- [ ] **Step 6: Implement independent payload verification**

Parse the emitted JSON and CSV back. Reconstruct `ConfusionCounts` from JSON,
recompute all metrics with `metrics_from_counts`, sum CSV counts, and compare:

- exact filenames;
- exact schemas and types;
- fixed counts;
- micro total equals 195,900 for production config;
- CSV count sum equals JSON micro counts;
- macro values and defined counts match CSV recomputation;
- governance flags are exact false/true values;
- Markdown contains the four exact micro counts.

- [ ] **Step 7: Implement fail-closed atomic publication**

Adapt Exp414 publication semantics but require exactly three payloads:

1. create a unique sibling staging directory;
2. write each file with mode `xb`, flush and fsync;
3. verify bytes, SHA and no symlink;
4. render/write `MANIFEST.json`;
5. verify exact four-file staging set;
6. invoke `pre_publish_verify`;
7. fsync staging;
8. repeat exact staging verification;
9. `os.replace(staging, output)`;
10. fsync parent or roll output back;
11. remove only the exact staging directory on failure.

- [ ] **Step 8: Run focused, package, and regression checks**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_reporting \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_publication -v
.venv/bin/python -m unittest discover \
  -s experiments/exp416_exp404_hard_test_confusion_audit/tests \
  -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp414_missing_aware_near_repeat/tests \
  -p "test_*.py"
git diff --check
```

Expected: all tests pass; Exp414 remains green.

- [ ] **Step 9: Commit reporting and publication**

```bash
git add experiments/exp416_exp404_hard_test_confusion_audit
git commit -m "feat: publish Exp416 confusion audit atomically"
```

---

### Task 5: One-Shot CLI, Runbook, and Clean-Commit Real Audit

**Files:**
- Create: `scratch/run_exp416_exp404_hard_test_confusion_audit.py`
- Create: `experiments/exp416_exp404_hard_test_confusion_audit/tests/test_cli.py`
- Create: `docs/EXP416_RUNBOOK_KO.md`
- External create after clean commit: `/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit/`

**Interfaces:**
- Consumes: all Task 1-4 public functions.
- Produces:
  - CLI `main(argv: list[str] | None = None) -> int`
  - one immutable four-file real output directory.

- [ ] **Step 1: Write CLI argument and ordering tests**

The CLI accepts exactly five required paths:

```text
--database-reference PATH
--prediction PATH
--prediction-manifest PATH
--frozen-test-labels PATH
--output-dir NEW_PATH
```

Test missing paths, existing output, nonexistent parent, forbidden extra mode,
and call ordering:

```python
def test_cli_freezes_prediction_before_loading_labels(self) -> None:
    events: list[str] = []
    with (
        mock.patch.object(cli, "verify_database_reference",
            side_effect=lambda *a, **k: events.append("database") or self.database_sha),
        mock.patch.object(cli, "freeze_prediction_inputs",
            side_effect=lambda *a, **k: events.append("freeze") or self.prediction),
        mock.patch.object(cli, "load_frozen_test_labels",
            side_effect=lambda *a, **k: events.append("labels") or self.labels),
        mock.patch.object(cli, "compute_confusion_audit",
            side_effect=lambda *a, **k: events.append("audit") or self.result),
        mock.patch.object(cli, "render_payloads",
            side_effect=lambda *a, **k: events.append("render") or self.payloads),
        mock.patch.object(cli, "publish_payloads",
            side_effect=lambda *a, **k: events.append("publish") or self.hashes),
    ):
        self.assertEqual(cli.main(self.argv), 0)
    self.assertEqual(
        events,
        ["database", "freeze", "labels", "audit", "render", "publish"],
    )
```

Add failure tests proving label load is not called after prediction failure and
publication is not called after label/audit/render failure.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp416_exp404_hard_test_confusion_audit.tests.test_cli -v
```

Expected: FAIL because the CLI is absent.

- [ ] **Step 3: Implement clean-Git one-shot orchestration**

The CLI must:

1. resolve the DB reference and three artifact input files, and require a new output path;
2. run `git status --porcelain` and require no tracked/untracked entries;
3. record `git rev-parse HEAD`;
4. call `verify_database_reference` without opening SQLite;
5. call `freeze_prediction_inputs`;
6. call `load_frozen_test_labels` with the returned object;
7. call `compute_confusion_audit`;
8. render and independently verify payloads;
9. publish with `pre_publish_verify=lambda: verify_input_snapshots(...)`;
10. print canonical JSON containing artifact hashes and micro counts.

Do not accept alpha, threshold, source, budget, family filter or dataset subset
arguments.

- [ ] **Step 4: Write the Korean runbook**

`docs/EXP416_RUNBOOK_KO.md` must contain:

- fixed SHA verification commands;
- `git status --short --branch` and exact clean-commit rule;
- test commands;
- a statement that preregistration commit predates real label load;
- the exact one-shot command:

```bash
.venv/bin/python scratch/run_exp416_exp404_hard_test_confusion_audit.py \
  --database-reference "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --prediction "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl" \
  --prediction-manifest "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json" \
  --frozen-test-labels "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp403_408_frozen_test_labels.csv" \
  --output-dir "/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit"
```

- manifest hash verification;
- no rerun/overwrite rule;
- no TEST-informed reselection or operational promotion.

- [ ] **Step 5: Run all tests before commit**

Run:

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp416_exp404_hard_test_confusion_audit/tests \
  -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp414_missing_aware_near_repeat/tests \
  -p "test_*.py"
.venv/bin/python -m compileall -q \
  experiments/exp416_exp404_hard_test_confusion_audit \
  scratch/run_exp416_exp404_hard_test_confusion_audit.py
git diff --check
```

Expected: Exp416 and Exp414 suites pass; compile/diff checks are silent.

- [ ] **Step 6: Commit CLI and runbook before real TEST load**

```bash
git add scratch/run_exp416_exp404_hard_test_confusion_audit.py \
  experiments/exp416_exp404_hard_test_confusion_audit/tests/test_cli.py \
  docs/EXP416_RUNBOOK_KO.md
git commit -m "feat: prepare Exp416 frozen TEST audit"
```

- [ ] **Step 7: Verify clean commit and frozen real inputs**

Run:

```bash
git status --short --branch
git rev-parse HEAD
shasum -a 256 \
  "/Users/mac/Documents/Dataset/univariate_ts.db" \
  "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl" \
  "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json" \
  "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp403_408_frozen_test_labels.csv"
test ! -e "/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit"
```

Expected: clean branch, four exact frozen hashes, output absent.

- [ ] **Step 8: Run the real Exp416 audit once**

Run the exact runbook command synchronously. It should complete quickly because
it streams 2,218 prediction rows and 195,900 label rows without model fitting.

Expected:

- exit code 0;
- output contains exactly four files;
- 1,117 dataset CSV rows plus header;
- JSON micro counts sum to 195,900;
- no input file changes.

- [ ] **Step 9: Independently verify published artifacts**

Run:

```bash
output="/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit"
test "$(find "$output" -maxdepth 1 -type f | wc -l | tr -d ' ')" = "4"
cd "$output"
jq -r '.artifact_sha256 | to_entries[] | "\(.value)  \(.key)"' \
  MANIFEST.json | shasum -a 256 -c -
test "$(wc -l < exp416_dataset_metrics.csv | tr -d ' ')" = "1118"
jq '.micro.counts | .tp + .fp + .fn + .tn' \
  exp416_confusion_matrix.json
```

Expected: all three payload hashes report `OK`, CSV lines equal 1,118, count
sum prints `195900`.

- [ ] **Step 10: Report without reselection**

Report:

- exact TP, FP, FN, TN;
- precision, recall, specificity, FPR, FNR, F1;
- macro values and their defined dataset counts;
- prediction identity SHA and manifest SHA;
- output manifest SHA;
- explicit statements that Exp415 prediction was not created, Exp404 was not
  modified, and no policy was reselected.

Do not recommend a new alpha or threshold from the observed TEST result. Any
future change requires a new experiment number and preregistration.
