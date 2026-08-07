# Exp414 Missing-Aware Near-Repeat Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exp411의 연구 계약을 유지하면서 정상 TRAIN의 비유한값을 결정론적으로 복구하고, 1,117개 전체 모집단에서 Exp414 near-repeat 자격 심사를 안전하게 재개할 수 있는 별도 파이프라인을 만든다.

**Architecture:** `experiments.exp411_near_repeat_qualification`의 동결된 fold·거리·교란·qualification 엔진은 수정하지 않는다. 새 `experiments.exp414_missing_aware_near_repeat` 패키지가 TRAIN-only 결측 복구, 복구 후 identity, Exp411 checkpoint의 검증된 이전, Exp414 checkpoint, subgroup 집계와 9개 산출물 게시를 담당한다. 원본 DB, Exp411 checkpoint와 Exp411 출력 경로는 읽기 전용으로 보존한다.

**Tech Stack:** Python 3.14, NumPy float32/float64, Numba 12-thread CPU, SQLite read-only URI, `unittest`, canonical JSON/SHA256, atomic filesystem writes, macOS `launchctl`

## Global Constraints

- 입력 DB SHA256은 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`이다.
- 모집단 manifest SHA256은 `41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8`이다.
- 모집단은 정확히 1,117 datasets와 128 families다.
- 영향 inventory는 정확히 19 datasets, 164 TRAIN runs, 983 non-finite values, 0 all-missing runs다.
- 영향 dataset 정렬 배열의 canonical JSON SHA256은 `ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9`다.
- 내부 결측은 선형 보간하고 시작·끝 결측은 가장 가까운 유한값으로 연장한다.
- 복구 결과는 little-endian float32이며 길이는 변경하지 않는다.
- 전체 결측 run이 하나라도 있는 dataset은 `missing_support_abstain`으로 처리한다.
- tail replacement, 길이 보간, padding과 truncation을 사용하지 않는다.
- TEST split, `labels_blob`, 실제 이상 위치와 과거 TEST 성능을 prediction/qualification에서 읽지 않는다.
- Exp411의 seed `411`, quantile `0.01 / 0.05 / 0.10`, 거리 view, fold, 합성 교란과 gate를 변경하지 않는다.
- NumPy CPU가 수치 정본이며 Numba는 12 threads, 중첩 수치 라이브러리는 1 thread를 사용한다.
- NumPy와 Numba의 최대 허용 오차는 `1e-6`이고 tag 및 selected quantile이 같아야 한다.
- MLX, PyTorch와 MPS는 Exp414 실행 경로에서 사용하지 않는다.
- Exp411 checkpoint는 읽기 전용이며 Exp414 checkpoint와 출력 디렉터리는 별도다.
- Exp412와 Exp413은 시작하지 않는다. Exp414 통과 뒤의 prediction/audit 번호는 Exp415/416이다.
- 전체 1,117개 실행은 구현, 검토, benchmark 완료 후 사용자의 별도 명시적 승인을 받아야 한다.

---

## File Structure

- `docs/EXP411_FAILURE_REPORT_KO.md`: 중단 증거, 187 record와 stale progress 차이, 주장 경계를 동결한다.
- `docs/EXP414_PREREGISTRATION_KO.md`: 실행 전 입력, 복구, gate, 출력과 금지사항을 사전등록한다.
- `experiments/exp414_missing_aware_near_repeat/config.py`: Exp414 고정값과 연구 계약 검증을 담당한다.
- `experiments/exp414_missing_aware_near_repeat/contracts.py`: repair, inventory, prepared dataset과 checkpoint record 타입을 정의한다.
- `experiments/exp414_missing_aware_near_repeat/repair.py`: 단일 blob의 결정론적 복구와 audit hash를 계산한다.
- `experiments/exp414_missing_aware_near_repeat/inputs.py`: TRAIN-only SQLite stream과 전체 결측 inventory를 검증한다.
- `experiments/exp414_missing_aware_near_repeat/checkpoints.py`: Exp414 dataset record와 progress를 원자적으로 저장한다.
- `experiments/exp414_missing_aware_near_repeat/migration.py`: Exp411 source checkpoint를 검증하고 clean record만 이전한다.
- `experiments/exp414_missing_aware_near_repeat/runner.py`: smoke, full stream, subgroup 집계와 게시 전 재검증을 조정한다.
- `experiments/exp414_missing_aware_near_repeat/reporting.py`: repair audit CSV를 포함한 8개 non-manifest payload를 렌더링한다.
- `experiments/exp414_missing_aware_near_repeat/publication.py`: 8개 payload와 `MANIFEST.json`을 원자적으로 게시한다.
- `experiments/exp414_missing_aware_near_repeat/benchmark.py`: 기존 3개 속도 dataset과 1개 결측 dataset의 gate를 실행한다.
- `scratch/run_exp414_missing_aware_near_repeat.py`: audit, smoke, benchmark와 full CLI를 제공한다.
- `scratch/launch_exp414_full.sh`: preflight 후 one-shot launchd job을 제출한다.
- `scratch/run_exp414_launchd_once.sh`: 종료 시 자신의 launchd label을 제거한다.
- `docs/EXP414_RUNBOOK_KO.md`: 검증, benchmark, 승인 후 실행과 결과 확인 명령을 기록한다.

---

### Task 1: Failure Freeze, Preregistration, and Frozen Contracts

**Files:**
- Create: `docs/EXP411_FAILURE_REPORT_KO.md`
- Create: `docs/EXP414_PREREGISTRATION_KO.md`
- Create: `experiments/exp414_missing_aware_near_repeat/__init__.py`
- Create: `experiments/exp414_missing_aware_near_repeat/config.py`
- Create: `experiments/exp414_missing_aware_near_repeat/contracts.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/__init__.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_config.py`

**Interfaces:**
- Consumes: `experiments.exp411_near_repeat_qualification.config.FROZEN_CONFIG`.
- Produces: `Exp414FrozenConfig`, `FROZEN_CONFIG`, `RunRepairAudit`, `DatasetRepairAudit`, `MissingInventory`, `PreparedDataset`.

- [ ] **Step 1: Write the frozen-config tests**

```python
class ConfigTests(unittest.TestCase):
    def test_frozen_values_match_preregistration(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp414")
        self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
        self.assertEqual(FROZEN_CONFIG.expected_families, 128)
        self.assertEqual(FROZEN_CONFIG.expected_missing_datasets, 19)
        self.assertEqual(FROZEN_CONFIG.expected_missing_runs, 164)
        self.assertEqual(FROZEN_CONFIG.expected_nonfinite_values, 983)
        self.assertEqual(FROZEN_CONFIG.expected_all_missing_runs, 0)
        self.assertEqual(FROZEN_CONFIG.seed, 411)
        self.assertEqual(FROZEN_CONFIG.quantiles, (0.01, 0.05, 0.10))
        FROZEN_CONFIG.validate()

    def test_contract_rejects_mutation(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(FROZEN_CONFIG, expected_missing_datasets=18).validate()
```

- [ ] **Step 2: Run the focused test and verify the package is missing**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_config -v
```

Expected: FAIL with `ModuleNotFoundError` for `exp414_missing_aware_near_repeat`.

- [ ] **Step 3: Define exact immutable contracts**

Create frozen dataclasses with these signatures:

```python
@dataclass(frozen=True)
class RunRepairAudit:
    dataset: str
    instance_index: int
    raw_values_sha256: str
    repaired_values_sha256: str | None
    length: int
    nan_count: int
    posinf_count: int
    neginf_count: int
    missing_positions_sha256: str
    repair_method: str

@dataclass(frozen=True)
class DatasetRepairAudit:
    dataset: str
    runs: tuple[RunRepairAudit, ...]
    raw_train_identity_sha256: str
    repaired_train_identity_sha256: str | None
    status: str

@dataclass(frozen=True)
class MissingInventory:
    datasets: int
    affected_datasets: int
    affected_runs: int
    nonfinite_values: int
    all_missing_runs: int
    affected_datasets_sha256: str

@dataclass(frozen=True)
class PreparedDataset:
    dataset: str
    runs: tuple[TrainRun, ...]
    repair_audit: DatasetRepairAudit
```

`Exp414FrozenConfig` must carry every Global Constraint constant, the Exp411 source commit `5c4268fb89ac806dac6bdccd31f5724ee520c63a`, and `to_payload()` using `asdict`. `validate()` must compare against a fresh default instance exactly.

- [ ] **Step 4: Write the failure report and preregistration**

`docs/EXP411_FAILURE_REPORT_KO.md` must record:

- final publication absent;
- 187 valid dataset records;
- stale `progress.json` value 186 and why it is not authoritative;
- first failing input `DodgerLoopDay_normal_1`, run 6, 71 leading NaNs;
- full TRAIN-only scan `19 / 164 / 983 / 0`;
- no TEST access and no operational lane change.

`docs/EXP414_PREREGISTRATION_KO.md` must reproduce the design's input hashes, repair algorithm, repaired-identity grouping, unchanged Exp411 gate, subgroup diagnostic-only rule, migration conditions, nine outputs and full-run approval gate.

- [ ] **Step 5: Run config tests and documentation guards**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_config -v
rg -n "19|164|983|ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9" \
  docs/EXP414_PREREGISTRATION_KO.md
git diff --check
```

Expected: tests pass, all four frozen inventory values are present, and `git diff --check` is silent.

- [ ] **Step 6: Commit the governance boundary**

```bash
git add docs/EXP411_FAILURE_REPORT_KO.md docs/EXP414_PREREGISTRATION_KO.md \
  experiments/exp414_missing_aware_near_repeat
git commit -m "docs: preregister Exp414 missing-aware qualification"
```

---

### Task 2: Deterministic Repair and TRAIN-Only Inventory

**Files:**
- Create: `experiments/exp414_missing_aware_near_repeat/repair.py`
- Create: `experiments/exp414_missing_aware_near_repeat/inputs.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_repair.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_inputs.py`

**Interfaces:**
- Consumes: `RunRepairAudit`, `DatasetRepairAudit`, `MissingInventory`, `PreparedDataset`, Exp411 `TrainRun`, `database_snapshot`, `trace_sql_contract`, and Exp414 `FROZEN_CONFIG`.
- Produces: `repair_train_blob(dataset: str, instance_index: int, payload: bytes) -> tuple[TrainRun | None, RunRepairAudit]`, `build_dataset_repair_audit(dataset: str, runs: Sequence[TrainRun], audits: Sequence[RunRepairAudit]) -> DatasetRepairAudit`, `iter_prepared_train_datasets(db_path: Path, datasets: Sequence[str], *, config: Exp414FrozenConfig = FROZEN_CONFIG) -> Iterator[PreparedDataset]`, `scan_missing_inventory(db_path: Path, datasets: Sequence[str], *, config: Exp414FrozenConfig = FROZEN_CONFIG) -> MissingInventory`.

- [ ] **Step 1: Write exact repair tests**

```python
def test_repairs_internal_and_edges_as_little_endian_float32(self) -> None:
    raw = np.asarray([np.nan, 1.0, np.nan, 5.0, np.inf], dtype="<f4").tobytes()
    run, audit = repair_train_blob("Alpha_normal_0", 7, raw)
    self.assertIsNotNone(run)
    np.testing.assert_array_equal(
        np.asarray(run.values, dtype="<f4"),
        np.asarray([1.0, 1.0, 3.0, 5.0, 5.0], dtype="<f4"),
    )
    self.assertEqual(audit.repair_method, "linear_and_edge_nearest")
    self.assertEqual((audit.nan_count, audit.posinf_count, audit.neginf_count), (2, 1, 0))

def test_finite_blob_is_byte_identical(self) -> None:
    raw = np.asarray([0.25, 1.5, -2.0], dtype="<f4").tobytes()
    run, audit = repair_train_blob("Alpha_normal_0", 0, raw)
    self.assertEqual(audit.raw_values_sha256, audit.repaired_values_sha256)
    self.assertEqual(np.asarray(run.values, dtype="<f4").tobytes(), raw)
    self.assertEqual(audit.repair_method, "unchanged")

def test_all_missing_returns_no_run(self) -> None:
    raw = np.asarray([np.nan, np.inf, -np.inf], dtype="<f4").tobytes()
    run, audit = repair_train_blob("Alpha_normal_0", 0, raw)
    self.assertIsNone(run)
    self.assertIsNone(audit.repaired_values_sha256)
    self.assertEqual(audit.repair_method, "all_missing_abstain")
```

- [ ] **Step 2: Write SQLite isolation and inventory tests**

Create a temporary DB containing two normal TRAIN rows and a TEST row whose
`values_blob` and `labels_blob` are sentinel bytes. Assert:

```python
prepared = tuple(iter_prepared_train_datasets(path, ("Alpha_normal_0",), config=config))
self.assertEqual(len(prepared), 1)
self.assertEqual(prepared[0].repair_audit.status, "ready")
self.assertNotIn(b"TEST_SENTINEL", captured_payloads)
self.assertEqual(inventory.affected_datasets, 1)
self.assertEqual(inventory.affected_runs, 1)
self.assertEqual(inventory.nonfinite_values, 1)
```

Also assert duplicate indices, non-zero TRAIN labels, malformed blob widths, WAL/SHM sidecars and post-repair non-finite values raise `ValueError`.

- [ ] **Step 3: Run tests and verify missing interfaces**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_repair \
  experiments.exp414_missing_aware_near_repeat.tests.test_inputs -v
```

Expected: FAIL on missing `repair_train_blob` and input stream interfaces.

- [ ] **Step 4: Implement byte-stable repair**

Use this algorithm without calling the older generic sanitizer:

```python
values = np.frombuffer(payload, dtype="<f4")
finite = np.isfinite(values)
positions = np.arange(len(values), dtype=np.float64)
if finite.all():
    repaired_payload = payload
elif not finite.any():
    return None, all_missing_audit
else:
    repaired = values.astype(np.float64, copy=True)
    repaired[~finite] = np.interp(
        positions[~finite],
        positions[finite],
        values[finite].astype(np.float64),
    )
    repaired_payload = np.asarray(repaired, dtype="<f4").tobytes()
```

Hash missing positions as canonical JSON of sorted integer indices. Set
`TrainRun.values_sha256` to the repaired payload SHA256 so repaired-identical
waveforms share an identity group. Preserve the raw digest only in
`RunRepairAudit`.

- [ ] **Step 5: Implement the streaming TRAIN-only loader**

Open SQLite with `file:<resolved>?mode=ro`, enable a trace callback, run
`PRAGMA quick_check`, and execute only:

```sql
SELECT id FROM datasets WHERE name = ?
```

```sql
SELECT COUNT(*), SUM(CASE WHEN label = '0' THEN 1 ELSE 0 END)
FROM instances
WHERE dataset_id = ? AND split = 'TRAIN'
```

```sql
SELECT instance_index, values_blob
FROM instances
WHERE dataset_id = ? AND split = 'TRAIN' AND label = '0'
ORDER BY instance_index
```

Call `trace_sql_contract` before closing. A dataset containing any
`all_missing_abstain` audit gets `DatasetRepairAudit.status =
"missing_support_abstain"` and an empty `runs` tuple; otherwise it gets
`status="ready"`.

- [ ] **Step 6: Validate the real TRAIN inventory read-only**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from experiments.exp411_near_repeat_qualification.runner import load_population_manifest
from experiments.exp414_missing_aware_near_repeat.inputs import scan_missing_inventory

manifest = Path("/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json")
datasets = load_population_manifest(manifest)
print(scan_missing_inventory(
    Path("/Users/mac/Documents/Dataset/univariate_ts.db"),
    datasets,
))
PY
```

Expected exact values: `1117`, `19`, `164`, `983`, `0`, and affected digest
`ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9`.

- [ ] **Step 7: Run focused regressions and commit**

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_repair \
  experiments.exp414_missing_aware_near_repeat.tests.test_inputs \
  experiments.exp411_near_repeat_qualification.tests.test_inputs -v
git diff --check
git add experiments/exp414_missing_aware_near_repeat
git commit -m "feat: add Exp414 TRAIN-only missing repair"
```

---

### Task 3: Exp414 Checkpoints and Verified Exp411 Migration

**Files:**
- Create: `experiments/exp414_missing_aware_near_repeat/checkpoints.py`
- Create: `experiments/exp414_missing_aware_near_repeat/migration.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_checkpoints.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_migration.py`

**Interfaces:**
- Consumes: Exp411 `CheckpointProvenance`, `CheckpointStore`, `qualify_dataset`; Exp414 `PreparedDataset`, repair audits and config.
- Produces: `Exp414CheckpointProvenance`, `Exp414CheckpointStore.has_dataset(dataset: str) -> bool`, `Exp414CheckpointStore.progress() -> Mapping[str, object]`, `open_exp411_checkpoint(root: Path, datasets: Sequence[str]) -> CheckpointStore`, `verify_migration_representatives(source: CheckpointStore, prepared_by_dataset: Mapping[str, PreparedDataset]) -> Mapping[str, str]`, `migrate_if_eligible(prepared: PreparedDataset, source: CheckpointStore, target: Exp414CheckpointStore, representative_evidence: Mapping[str, str]) -> bool`.

- [ ] **Step 1: Write atomic checkpoint tests**

Tests must create a two-dataset store and assert:

```python
store.write_dataset(
    row,
    backend_audit,
    raw_train_identity_sha256="1" * 64,
    repaired_train_identity_sha256="1" * 64,
    repair_audit_sha256="2" * 64,
    execution_source="exp414_computed",
    elapsed_seconds=0.25,
)
self.assertEqual(store.progress()["completed_datasets"], 1)
self.assertEqual(store.load_dataset_record(row.dataset)["execution_source"], "exp414_computed")
```

Corrupt payload hash, foreign dataset, boolean counter, wrong provenance,
invalid execution source and non-atomic duplicate writes must fail closed.
`require_complete(128)` must require exact dataset and family coverage.

- [ ] **Step 2: Write migration eligibility tests**

Use an Exp411 fixture checkpoint and prepared datasets to assert:

- clean matching identity migrates with `migrated_exp411_verified`;
- repaired dataset is never migrated;
- raw identity mismatch recomputes by returning `False`;
- invalid Exp411 root provenance fails the migration operation;
- stale Exp411 `progress.json` is ignored and valid dataset records are counted;
- fixed representatives are exactly `ACSF1_normal_0`, `CBF_normal_2`,
  `DistalPhalanxTW_normal_8`;
- a representative qualification mismatch blocks all migration.

- [ ] **Step 3: Run tests and verify missing modules**

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_checkpoints \
  experiments.exp414_missing_aware_near_repeat.tests.test_migration -v
```

Expected: FAIL with missing checkpoint and migration interfaces.

- [ ] **Step 4: Implement Exp414 record schema**

Each canonical record must contain exactly:

```python
{
    "schema_version": 1,
    "dataset": row.dataset,
    "raw_train_identity_sha256": raw_digest,
    "repaired_train_identity_sha256": repaired_digest,
    "repair_audit_sha256": repair_digest,
    "execution_source": execution_source,
    "qualification": asdict(row),
    "backend_audit": asdict(audit),
    "elapsed_seconds": float(elapsed_seconds),
    "payload_sha256": digest_without_this_field,
}
```

Root provenance must additionally bind `missing_repair_contract_sha256`,
`missing_inventory_sha256`, `qualification_contract_sha256`, and
`exp411_migration_source_sha256`. Reuse Exp411's strict counter and
perturbation-key validation behavior in Exp414-owned validators; do not weaken
the Exp411 module.

- [ ] **Step 5: Implement source opening and representative parity**

`open_exp411_checkpoint` must parse the source root, require Git commit
`5c4268fb89ac806dac6bdccd31f5724ee520c63a`, require the pinned DB and
population hashes, instantiate the public Exp411 `CheckpointStore`, and load
records through its validators.

For each fixed representative, load current clean TRAIN through Exp414,
recompute with NumPy `qualify_dataset`, and require equality with the Exp411
qualification record. Return a canonical mapping of representative dataset to
record payload hash for provenance.

- [ ] **Step 6: Implement per-dataset migration**

`migrate_if_eligible` returns `False` without writing when the dataset has any
repair, source record is absent, or raw/repaired identity differs from the
source identity. It writes the already validated qualification and backend
audit to the Exp414 store only after the global representative parity gate
passed. Any source corruption raises instead of silently recomputing.

- [ ] **Step 7: Run regressions and commit**

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_checkpoints \
  experiments.exp414_missing_aware_near_repeat.tests.test_migration \
  experiments.exp411_near_repeat_qualification.tests.test_checkpoints -v
git diff --check
git add experiments/exp414_missing_aware_near_repeat
git commit -m "feat: add verified Exp414 checkpoint migration"
```

---

### Task 4: Qualification Orchestration and Missing-Support Abstention

**Files:**
- Create: `experiments/exp414_missing_aware_near_repeat/runner.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_runner.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_structure.py`

**Interfaces:**
- Consumes: Exp411 `qualify_dataset`, `aggregate_qualification`, `configure_runtime`; Exp414 input stream, checkpoint and migration interfaces.
- Produces: `missing_support_abstention(prepared: PreparedDataset) -> DatasetQualification`, `qualify_prepared_dataset(prepared: PreparedDataset, *, accelerated: bool, backend_audit: list[DatasetBackendAudit] | None = None) -> DatasetQualification`, `aggregate_subgroups(rows: Sequence[DatasetQualification], audits: Sequence[DatasetRepairAudit], *, config: Exp414FrozenConfig = FROZEN_CONFIG) -> Mapping[str, object]`, `run_representative_smoke(output_dir: Path) -> Mapping[str, str]`, `run_full_qualification(db_path: Path, population_manifest: Path, output_dir: Path, *, exp411_checkpoint_dir: Path, checkpoint_dir: Path, benchmark_report: Path) -> Mapping[str, str]`.

- [ ] **Step 1: Write missing-support and clean parity tests**

```python
def test_all_missing_run_abstains_entire_dataset(self) -> None:
    row = missing_support_abstention(prepared)
    self.assertEqual(row.normal_total, 6)
    self.assertEqual(row.normal_abstained, 6)
    self.assertTrue(all(value == 0 for value in row.normal_near.values()))
    self.assertTrue(all(value == 0 for value in row.synthetic_total.values()))

def test_clean_prepared_dataset_matches_exp411(self) -> None:
    expected = qualify_dataset(dataset=name, family=family, runs=prepared.runs)
    observed = qualify_prepared_dataset(prepared, accelerated=False)
    self.assertEqual(observed, expected)
```

- [ ] **Step 2: Write orchestration and forbidden-access tests**

Assert a fake three-dataset stream:

- migrates one clean record;
- computes one repaired record;
- creates one missing-support abstention;
- calls aggregate only after checkpoint coverage is complete;
- does not create output if stream, inventory, benchmark, Git or DB recheck fails.

Parse `inputs.py`, `runner.py`, `repair.py`, `migration.py` and
`benchmark.py` with `ast`. Reject source constants or SQL containing
`labels_blob`, `split = 'TEST'`, `is_true_anomaly`, TP, FP, FN or F1 field
access.

- [ ] **Step 3: Run tests and verify runner is missing**

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_runner \
  experiments.exp414_missing_aware_near_repeat.tests.test_structure -v
```

Expected: FAIL on missing runner interfaces.

- [ ] **Step 4: Implement deterministic abstention and subgroup aggregation**

Construct abstention counts for string quantile keys `"0.01"`, `"0.05"`,
`"0.10"` and all seven `perturbation_types()`. `normal_total` is the number of
raw TRAIN audit rows, `normal_abstained` is the same count, and all near and
synthetic counters are zero.

`aggregate_subgroups` must emit diagnostic summaries for:

```python
("all", "nonmissing", "missing_repaired", "DodgerLoop",
 "MelbournePedestrian", "missing_support_abstain")
```

Only `aggregate_qualification(all_rows, config=EXP411_FROZEN_CONFIG)` may choose
the global quantile. Subgroup aggregates must carry
`selection_authority=False`.

- [ ] **Step 5: Implement checkpointed stream execution**

For each `PreparedDataset`, calculate raw/repaired identity and repair-audit
hash, then follow:

```python
if target_store.has_dataset(dataset):
    record = target_store.load_dataset_record(dataset)
    if (
        record["raw_train_identity_sha256"] != raw_digest
        or record["repaired_train_identity_sha256"] != repaired_digest
        or record["repair_audit_sha256"] != repair_digest
    ):
        raise ValueError(f"Exp414 checkpoint TRAIN identity mismatch: {dataset}")
elif migration_enabled and migrate_if_eligible(
    prepared,
    source_store,
    target_store,
    representative_evidence,
):
    pass
elif prepared.repair_audit.status == "missing_support_abstain":
    target_store.write_dataset(
        missing_support_abstention(prepared),
        DatasetBackendAudit(
            backend="numpy_cpu",
            sampled_pairs=(),
            max_abs_error=0.0,
            fallback_reason="missing_support_abstain",
        ),
        raw_train_identity_sha256=raw_digest,
        repaired_train_identity_sha256=repaired_digest,
        repair_audit_sha256=repair_digest,
        execution_source="exp414_computed",
        elapsed_seconds=0.0,
    )
else:
    target_store.write_progress(current_dataset=dataset)
    started = perf_counter()
    result = qualify_dataset(
        dataset=dataset,
        family=family,
        runs=prepared.runs,
        accelerated=True,
        backend_audit=audits,
        config=EXP411_FROZEN_CONFIG,
    )
    target_store.write_dataset(
        result,
        audits[0],
        raw_train_identity_sha256=raw_digest,
        repaired_train_identity_sha256=repaired_digest,
        repair_audit_sha256=repair_digest,
        execution_source="exp414_computed",
        elapsed_seconds=perf_counter() - started,
    )
```

Never use `progress.json` as completion authority. Dataset record validation is
the only completion authority.

- [ ] **Step 6: Implement smoke and full pre-publication gates**

Smoke must include clean, internal-gap, edge-gap, repaired-identity collision
and all-missing synthetic datasets. Full mode must require:

- clean Git commit before and after;
- exact DB snapshot and manifest hash;
- exact missing inventory;
- successful Exp414 benchmark bound to the same commit and contracts;
- runtime `numba_threads=12`, `nested_threads=1`;
- exact 1,117/128 checkpoint coverage;
- output path absent.

- [ ] **Step 7: Run focused and Exp411 parity regressions**

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_runner \
  experiments.exp414_missing_aware_near_repeat.tests.test_structure \
  experiments.exp411_near_repeat_qualification.tests.test_qualification \
  experiments.exp411_near_repeat_qualification.tests.test_acceleration -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit orchestration**

```bash
git add experiments/exp414_missing_aware_near_repeat
git commit -m "feat: orchestrate Exp414 missing-aware qualification"
```

---

### Task 5: Nine-Artifact Reporting and Atomic Publication

**Files:**
- Create: `experiments/exp414_missing_aware_near_repeat/reporting.py`
- Create: `experiments/exp414_missing_aware_near_repeat/publication.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_reporting.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_publication.py`
- Modify: `experiments/exp414_missing_aware_near_repeat/runner.py`

**Interfaces:**
- Consumes: completed qualification records, per-run repair audits, global aggregate, subgroup aggregates, backend assessment and provenance.
- Produces: `render_payloads(records: Sequence[Mapping[str, object]], repair_audits: Sequence[DatasetRepairAudit], aggregate: Mapping[str, object], subgroups: Mapping[str, object], backend_assessment: Mapping[str, object], provenance: Mapping[str, object], *, full_qualification_executed: bool, config: Exp414FrozenConfig = FROZEN_CONFIG) -> Mapping[str, bytes]`, `publish_payloads(payloads: Mapping[str, bytes], output_dir: Path, provenance: Mapping[str, object], *, pre_publish_verify: Callable[[], None] | None = None) -> Mapping[str, str]`.

- [ ] **Step 1: Write exact artifact coverage tests**

Assert `render_payloads` returns exactly:

```python
{
    "exp414_frozen_config.json",
    "exp414_missing_repair_audit.csv",
    "exp414_dataset_qualification.csv",
    "exp414_quantile_summary.csv",
    "exp414_backend_assessment.json",
    "exp414_selected_rule.json",
    "exp414_summary.json",
    "EXP414_RESULT_REPORT_KO.md",
}
```

Assert the repair CSV contains one row for every raw TRAIN run and columns:

```python
(
    "dataset", "instance_index", "raw_values_sha256",
    "repaired_values_sha256", "length", "nan_count", "posinf_count",
    "neginf_count", "missing_positions_sha256", "repair_method",
)
```

Assert the dataset CSV includes `execution_source` and
`repair_status`. Assert summary has `exp412_prediction_created=false`,
`exp413_audit_created=false`, `test_values_used=false`,
`test_labels_used=false`, and subgroup rows with
`selection_authority=false`.

- [ ] **Step 2: Write atomic publication failure tests**

Use a temporary parent and assert:

- exactly 8 non-manifest payloads are required;
- existing output is rejected;
- staged byte/hash mismatch is rejected;
- `pre_publish_verify` failure leaves no final directory;
- success creates exactly 9 files;
- `MANIFEST.json` hashes the other 8 and contains Exp414 provenance.

- [ ] **Step 3: Run tests and verify reporting modules are missing**

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_reporting \
  experiments.exp414_missing_aware_near_repeat.tests.test_publication -v
```

Expected: FAIL on missing `render_payloads` and `publish_payloads`.

- [ ] **Step 4: Implement deterministic rendering**

Sort dataset and repair rows by `(dataset, instance_index)`. Use
`csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n",
extrasaction="raise")`. JSON cells must use `sort_keys=True`,
`separators=(",", ":")`, and
`allow_nan=False`. The Korean report must state that Exp414 is TRAIN-only,
missing-aware qualification and does not change operational policy.

- [ ] **Step 5: Implement atomic publication**

Adapt Exp411 publication semantics with Exp414 messages and an exact
eight-payload requirement. Write each file with `xb`, `flush`, `fsync`,
verify its SHA256, write the manifest, call `pre_publish_verify`, and only then
`os.replace(staging, output)`. Remove staging on every exception.

- [ ] **Step 6: Connect runner rendering and pre-publish revalidation**

Before publication recheck Git, DB snapshot, manifest, inventory, benchmark
SHA256, checkpoint completeness and all dataset raw/repaired identities.
Include migration representative hashes and counts by `execution_source` in
provenance.

- [ ] **Step 7: Run reporting, runner and manifest regressions**

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_reporting \
  experiments.exp414_missing_aware_near_repeat.tests.test_publication \
  experiments.exp414_missing_aware_near_repeat.tests.test_runner -v
git diff --check
```

Expected: all tests pass and whitespace check is silent.

- [ ] **Step 8: Commit publication**

```bash
git add experiments/exp414_missing_aware_near_repeat
git commit -m "feat: publish Exp414 qualification atomically"
```

---

### Task 6: Benchmark, CLI, One-Shot Launcher, and Runbook

**Files:**
- Create: `experiments/exp414_missing_aware_near_repeat/benchmark.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_benchmark.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_cli.py`
- Create: `experiments/exp414_missing_aware_near_repeat/tests/test_launch_contract.py`
- Create: `scratch/run_exp414_missing_aware_near_repeat.py`
- Create: `scratch/run_exp414_launchd_once.sh`
- Create: `scratch/launch_exp414_full.sh`
- Create: `docs/EXP414_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: Exp414 inventory, repair, runner and publication interfaces; Exp411 acceleration engine.
- Produces: `BENCHMARK_DATASETS`, `run_fixed_train_benchmark(db_path: Path) -> Mapping[str, object]`, `require_acceleration_adoption(report, expected_commit) -> None`, four-mode CLI and safe detached full launcher.

- [ ] **Step 1: Write benchmark contract tests**

Freeze:

```python
BENCHMARK_DATASETS = {
    "small": "ACSF1_normal_0",
    "medium": "HandOutlines_normal_0",
    "large": "StarLightCurves_normal_3",
    "missing": "DodgerLoopDay_normal_1",
}
```

Require speedup `>= 2.0` only for medium and large; require error `<= 1e-6`,
qualification/tag/selected-quantile identity for every dataset; require the
missing row to report one repaired run and 71 repaired values; require runtime
12/1 and matching Git, DB, repair and qualification contract hashes.

- [ ] **Step 2: Write CLI and launcher tests**

CLI modes are mutually exclusive:

```text
--input-audit
--representative-smoke
--acceleration-benchmark
--full-qualification
```

Assert full mode requires DB, population manifest, Exp411 source checkpoint,
Exp414 checkpoint, benchmark report and new output path. Smoke forbids real
paths. Benchmark refuses an existing report.

`launch_exp414_full.sh --print-command` must contain:

```text
NUMBA_NUM_THREADS=12
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
com.openai.exp414.full
/private/tmp/exp414-full.stdout.log
/private/tmp/exp414-full.stderr.log
```

The submitted executable must be `run_exp414_launchd_once.sh`, whose EXIT trap
removes only `com.openai.exp414.full`. This prevents launchd keepalive
respawning without polling.

- [ ] **Step 3: Run tests and verify files are missing**

```bash
.venv/bin/python -m unittest \
  experiments.exp414_missing_aware_near_repeat.tests.test_benchmark \
  experiments.exp414_missing_aware_near_repeat.tests.test_cli \
  experiments.exp414_missing_aware_near_repeat.tests.test_launch_contract -v
```

Expected: FAIL on missing benchmark, CLI and launcher files.

- [ ] **Step 4: Implement the four-dataset benchmark**

Load all four through the Exp414 TRAIN-only input path. For each, run NumPy,
Numba cold and Numba warm qualification with parity traces. The missing
dataset uses repaired identities. Store timings, maximum backend audit error,
tag identity, selected-quantile identity, repair counts, runtime and all
contract hashes. Call `require_acceleration_adoption` before returning.

- [ ] **Step 5: Implement CLI modes and atomic benchmark write**

`--input-audit` prints canonical inventory JSON without writing. Smoke writes
only to a new temporary-style output path. Benchmark writes one new JSON file
atomically. Full mode delegates to `run_full_qualification` and contains no
implicit approval or automatic retry.

- [ ] **Step 6: Implement preflight and one-shot launch scripts**

Use fixed absolute paths:

```text
/Users/mac/Documents/Dataset/univariate_ts.db
/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json
/Users/mac/Documents/Dataset/exp411_near_repeat_qualification.checkpoint
/Users/mac/Documents/Dataset/exp414_missing_aware_near_repeat.checkpoint
/Users/mac/Documents/Dataset/exp414_acceleration_benchmark.json
/Users/mac/Documents/Dataset/exp414_missing_aware_near_repeat
```

Preflight requires a clean worktree, exact Git-bound benchmark, exact DB hash,
SQLite quick check, exact inventory and absent final output. It must refuse an
already loaded `com.openai.exp414.full` job. The helper installs:

```bash
trap 'launchctl remove com.openai.exp414.full >/dev/null 2>&1 || true' EXIT
```

and invokes the Exp414 Python CLI without `exec`, so the trap always runs.

- [ ] **Step 7: Write the Korean runbook**

Document environment verification, all unit tests, compile check, input audit,
representative smoke, benchmark, manifest verification, printed launch
command, separately approved full launch, one-time status inspection and
checkpoint resume. State explicitly that automated monitoring and polling are
not used.

- [ ] **Step 8: Run the complete pre-benchmark verification**

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp414_missing_aware_near_repeat/tests -p "test_*.py" -v
.venv/bin/python -m unittest discover \
  -s experiments/exp411_near_repeat_qualification/tests -p "test_*.py" -v
.venv/bin/python -m compileall -q \
  experiments/exp414_missing_aware_near_repeat \
  scratch/run_exp414_missing_aware_near_repeat.py
bash -n scratch/launch_exp414_full.sh
bash -n scratch/run_exp414_launchd_once.sh
git diff --check
```

Expected: all tests pass, compilation and shell syntax succeed, and diff check
is silent.

- [ ] **Step 9: Run representative smoke in a new temporary parent**

```bash
exp414_smoke_parent=$(mktemp -d /private/tmp/exp414-smoke-parent.XXXXXX)
.venv/bin/python scratch/run_exp414_missing_aware_near_repeat.py \
  --representative-smoke \
  --output-dir "$exp414_smoke_parent/publication"
```

Expected: exactly 9 files, manifest hashes valid, TEST-use flags false, clean
parity true, and missing-repair smoke coverage true.

- [ ] **Step 10: Commit the execution tooling**

```bash
git add experiments/exp414_missing_aware_near_repeat \
  scratch/run_exp414_missing_aware_near_repeat.py \
  scratch/run_exp414_launchd_once.sh scratch/launch_exp414_full.sh \
  docs/EXP414_RUNBOOK_KO.md
git commit -m "feat: prepare Exp414 full qualification"
```

---

### Task 7: Clean-Commit Benchmark and Full-Run Readiness Gate

**Files:**
- Verify only: repository files from Tasks 1-6
- Local-only output: `/Users/mac/Documents/Dataset/exp414_acceleration_benchmark.json`

**Interfaces:**
- Consumes: committed Exp414 code and the pinned local DB.
- Produces: a commit-bound benchmark report and an evidence-backed readiness decision; it does not start the 1,117-dataset run.

- [ ] **Step 1: Require a clean committed state**

```bash
git status --short --branch
git rev-parse HEAD
```

Expected: no porcelain entries and one 40-character commit.

- [ ] **Step 2: Run the real read-only input audit**

```bash
.venv/bin/python scratch/run_exp414_missing_aware_near_repeat.py \
  --input-audit \
  --db-path "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --population-manifest "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json"
```

Expected: exact `1117 / 19 / 164 / 983 / 0` and affected digest
`ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9`.

- [ ] **Step 3: Run the Exp414 benchmark once**

```bash
NUMBA_NUM_THREADS=12 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scratch/run_exp414_missing_aware_near_repeat.py \
  --acceleration-benchmark \
  --db-path "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --benchmark-report "/Users/mac/Documents/Dataset/exp414_acceleration_benchmark.json"
```

Expected: medium and large warm speedup at least 2.0, all numerical errors at
most `1e-6`, all tag and selected-quantile identity flags true, and missing
repair `1 run / 71 values`.

- [ ] **Step 4: Verify benchmark hash and launch preflight without submission**

```bash
shasum -a 256 "/Users/mac/Documents/Dataset/exp414_acceleration_benchmark.json"
bash scratch/launch_exp414_full.sh --print-command
```

Expected: the benchmark validates against the current clean commit and the
launcher prints one fixed-label one-shot command without submitting it.

- [ ] **Step 5: Report readiness and stop at the approval gate**

Report test counts, smoke manifest status, benchmark timings/speedups/error,
input inventory, current commit and checkpoint migration eligibility count.
Do not run `bash scratch/launch_exp414_full.sh` until the user explicitly
approves the Exp414 1,117-dataset full run.
