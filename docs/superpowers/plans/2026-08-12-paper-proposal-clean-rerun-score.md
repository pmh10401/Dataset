# 논문용 제안기술 Clean Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 최종 제안기술을 원본 공개 benchmark DB에서 label-free로 재실행하고, 원 경보 좌표를 보존한 연속 근거 score와 AP/AUROC·F1 비교 결과를 동결한다.

**Architecture:** 독립된 `paper/equipment_anomaly_detection/` namespace에 기존 연구 코드 중 필요한 과학 로직을 provenance와 함께 복사하고 모든 신규 파일을 `*_paper`로 명명한다. 논문 실행 경로는 내부 Exp package를 runtime dependency로 사용하지 않으며, 과거 prediction 좌표는 새 경보 생성이 끝난 뒤 parity oracle로만 사용한다. 제안기술 score root와 공개 baseline score root를 각각 동결한 다음 하나의 lazy evaluator가 처음으로 TEST label을 열고, 같은 dataset bootstrap draw로 결과를 집계한다.

**Tech Stack:** Python 3, NumPy, SciPy, scikit-learn, aeon, pycatch22, PyWavelets, PyTorch, MLX 선택적 smoke, SQLite read-only URI, `unittest`, canonical JSON/JSONL, SHA256

## Global Constraints

- DB는 `/Users/mac/Documents/Dataset/univariate_ts.db`, SHA256 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`로 고정한다.
- 모집단은 정렬된 1,117개 데이터셋과 195,900 TEST 행이며 `CornellWhaleChallenge`, `Wafer_normal_1`을 제외한다.
- 새 run identity는 `paper-proposal-clean-rerun-20260812`, root는 `/Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812`다.
- 과거 prediction 좌표는 새 score 또는 route 입력으로 금지하고 freeze 뒤 parity 비교에만 사용한다.
- TEST value는 score 단계에서 허용하지만 TEST label은 `evaluate` 전까지 import, SQL, file open 모두 금지한다.
- 주 구성은 Hard + stable 26이며 ablation 경보 수는 `2218`, `2248`, `2244`, `2222`여야 한다.
- score adapter는 empirical upper-tail `p`, `strength=1-p`, nonalert `0.5*strength`, alert `0.5+0.5*strength`를 사용한다.
- p-value 판정은 `p <= alpha`, Exp440 raw threshold는 `score > threshold`를 그대로 유지한다.
- bootstrap은 dataset 단위 20,000회, `PCG64(20260812)`, `int64`, `endpoint=False`다.
- 공개 비교군은 k-NN, LOF, RBF OCSVM, Isolation Forest, Deep SVDD 1D, DROCC 1D뿐이다.
- 고전 네 기법은 전체 실행하고 심층 두 기법은 합계 48시간 예산 안에서 완주한 결과만 주 표에 허용한다.
- PyTorch, MLX, MPS의 tensor eligibility·availability·overhead를 각각 평가한다. MLX/MPS는 CPU와 같은 input/seed, 모든 alpha candidate parity, finite score, 절대오차 `1e-5`, 전송·동기화를 포함한 1.1배 속도 조건을 모두 통과할 때만 사용한다.
- 새 CPU score stage는 `max(1, os.cpu_count() - 1)` dataset worker를 사용하고 감지 CPU 수와 선택 worker 수를 provenance에 기록하며, worker당 수치 thread는 1개다.
- 기존 experiment package와 `experiments.paper_public_baselines`의 과학 코드는 수정하지 않는다. 필요한 제안기술 로직은 원본 파일·symbol·SHA256를 기록한 뒤 paper 폴더에 복사하고 원본과의 parity test를 둔다.
- 논문용 신규 파일은 `paper/equipment_anomaly_detection/` 아래에 두고 파일명에 `paper`를 포함한다. 이 사용자 승인 규칙은 기존 `antigravity` 신규 파일명 정책에 대한 이 논문 폴더의 명시적 예외다.
- 논문 버전의 code map은 `codemap_paper.html`, `codemap_paper.json`, `codemap_paper.lock` 세 파일을 함께 갱신한다.
- 논문 본문에는 내부 Exp 번호와 `Stable-26`을 쓰지 않고 artifact provenance에만 남긴다.

---

## File Structure

```text
paper/equipment_anomaly_detection/
  package_paper.py                    namespace metadata and public identity
  config_paper.py                      frozen identity, constants, paths, counts
  contracts_paper.py                   immutable values, score, route contracts
  lineage_paper.py                     raw-computable/frozen-rule/parity-oracle lock
  lineage_lock_paper.json              machine-readable exact dependency closure
  codemap_paper.html                   human-readable paper dependency map
  codemap_paper.json                   machine-readable callers/effects/tests map
  codemap_paper.lock                   hashes binding the complete paper code map
  data_paper.py                        values-only SQLite and prepare population
  label_access_paper.py                lazy exact TEXT TEST-label loader
  calibration_paper.py                 p-values, margin, strength, Boolean aggregation
  score_adapter_paper.py               route-preserving binary/continuous score
  sources_paper/
    package_paper.py
    hard_sources_paper.py              raw Exp40/55/56/84/Block-B/canonical scores
    minirocket_paper.py                raw 1024/4096 two-seed MiniROCKET evidence
    low_frequency_paper.py             full TEST Exp440 multi-depth score replay
    stability_paper.py                 exact 30-candidate Exp444 omission replay
  hard_route_paper.py                  legacy route chain through guarded Hard
  routing_paper.py                     four final ablation policies
  artifacts_paper.py                   canonical create-only shards and manifests
  runner_paper.py                      ordered/resumable scoring and freeze lifecycle
  acceleration_paper.py               per-stage CPU/MLX assessment and fallback
  baseline_bridge_paper.py             verified public baseline score adapter
  evaluation_paper.py                 post-freeze metrics only
  statistics_paper.py                 paired dataset bootstrap
  independent_reaggregate_paper.py    immutable result independent verification
  reporting_paper.py                  paper tables and limitation text
  cli_paper.py                         prepare/assess-mlx/score/freeze/parity/evaluate/report
  tests_paper/
    test_config_lineage_paper.py
    test_data_boundary_paper.py
    test_calibration_adapter_paper.py
    test_hard_sources_paper.py
    test_hard_route_paper.py
    test_minirocket_paper.py
    test_low_frequency_paper.py
    test_stability_paper.py
    test_artifacts_runner_paper.py
    test_acceleration_paper.py
    test_baseline_bridge_paper.py
    test_evaluation_statistics_paper.py
    test_reporting_cli_paper.py
paper/equipment_anomaly_detection/run_paper.py
paper/equipment_anomaly_detection/docs_paper/preregistration_paper_ko.md
paper/equipment_anomaly_detection/docs_paper/runbook_paper_ko.md
paper/equipment_anomaly_detection/docs_paper/result_paper_ko.md
```

### Task 1: Frozen configuration, contracts, and dependency closure

**Files:**
- Create: `paper/equipment_anomaly_detection/package_paper.py`
- Create: `paper/equipment_anomaly_detection/config_paper.py`
- Create: `paper/equipment_anomaly_detection/contracts_paper.py`
- Create: `paper/equipment_anomaly_detection/lineage_paper.py`
- Create: `paper/equipment_anomaly_detection/lineage_lock_paper.json`
- Create: `paper/equipment_anomaly_detection/codemap_paper.html`
- Create: `paper/equipment_anomaly_detection/codemap_paper.json`
- Create: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_config_lineage_paper.py`

**Interfaces:**
- Consumes: approved design and existing source module paths/SHA256.
- Produces: `FROZEN_CONFIG: FrozenPaperConfig`, `LINEAGE_NODES: tuple[LineageNode, ...]`, `validate_lineage_lock(path: Path) -> tuple[LineageNode, ...]`, complete paper codemap triplet.

- [ ] **Step 1: Write failing identity and forbidden-input tests**

```python
class ConfigLineageTests(unittest.TestCase):
    def test_frozen_identity_and_counts(self) -> None:
        self.assertEqual(FROZEN_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_CONFIG.test_row_count, 195900)
        self.assertEqual(FROZEN_CONFIG.ablation_alert_counts,
                         (2218, 2248, 2244, 2222))

    def test_lineage_has_no_prediction_coordinate_input(self) -> None:
        nodes = validate_lineage_lock(LINEAGE_LOCK_PATH)
        self.assertFalse(any(node.input_kind == "prediction_coordinates" for node in nodes))
        self.assertTrue(all(node.label_access is False for node in nodes))

    def test_paper_codemap_triplet_is_complete_and_hash_bound(self) -> None:
        codemap = validate_codemap_triplet(PAPER_ROOT)
        self.assertEqual(set(codemap.required_fields),
                         {"callers", "effects", "tests", "source_origins"})

    def test_all_new_static_files_use_paper_naming(self) -> None:
        names = tuple(path.name for path in PAPER_ROOT.rglob("*") if path.is_file())
        self.assertTrue(all("paper" in name for name in names))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper -v`

Expected: FAIL because the package and frozen contracts do not exist.

- [ ] **Step 3: Implement immutable contracts and the exact lineage registry**

```python
@dataclass(frozen=True, slots=True)
class LineageNode:
    node_id: str
    stage: str
    input_kind: str
    module: str
    symbols: tuple[str, ...]
    source_sha256: str | None
    paper_module: str | None
    label_access: bool
    parity_oracle: str | None = None

@dataclass(frozen=True, slots=True)
class FrozenPaperConfig:
    run_identity: str = "paper-proposal-clean-rerun-20260812"
    dataset_count: int = 1117
    test_row_count: int = 195900
    seed: int = 20260812
    ablation_alert_counts: tuple[int, int, int, int] = (2218, 2248, 2244, 2222)
```

`lineage_lock_paper.json`에는 `rocket_exp40`, `exp55_best`, `exp56_best`, `exp84_family_guard`, `exp84_count_cap`, `rocket_block_b`, `canonical_normal_state`, `exp234_three_source`, `minirocket_1024_seed_intersection`, `minirocket_4096_seed_intersection`, `raw_exact_normal_guard`, `low_frequency_two_depth`, `hard_empty_budget_30`, `train_omission_stability`를 모두 넣는다. 각 scoring node의 `input_kind`는 `raw_train_test_values`, route constant는 `frozen_rule`, 기존 artifact는 `parity_oracle`로 기록한다. `module`과 `symbols`는 복사 원본, `source_sha256`는 원본 파일 hash, `paper_module`은 Task 4–7에서 만들 독립 paper 목적지를 나타낸다.

- [ ] **Step 4: Add import and symbol validation**

```python
def validate_lineage_lock(path: Path) -> tuple[LineageNode, ...]:
    nodes = parse_nodes(read_strict_json(path))
    if any(node.input_kind == "prediction_coordinates" for node in nodes):
        raise ValueError("old prediction coordinates are forbidden inputs")
    for node in nodes:
        if node.input_kind == "raw_train_test_values":
            module = importlib.import_module(node.module)
            if any(not hasattr(module, symbol) for symbol in node.symbols):
                raise ValueError(f"raw scorer symbol is unavailable: {node.node_id}")
            if sha256_file(Path(module.__file__)) != node.source_sha256:
                raise ValueError(f"raw scorer source SHA differs: {node.node_id}")
    return nodes
```

같은 module에서 strict JSON loader와 dataclass 변환도 정의해 숨은 helper를 남기지 않는다.

```python
def read_strict_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"non-finite JSON constant: {value}")))

def parse_nodes(payload: object) -> tuple[LineageNode, ...]:
    if not isinstance(payload, list):
        raise ValueError("lineage lock must be a JSON list")
    expected = {"node_id", "stage", "input_kind", "module", "symbols",
                "source_sha256", "paper_module", "label_access", "parity_oracle"}
    if any(not isinstance(record, dict) or set(record) != expected for record in payload):
        raise ValueError("lineage node schema is invalid")
    nodes = tuple(LineageNode(
        node_id=record["node_id"], stage=record["stage"],
        input_kind=record["input_kind"], module=record["module"],
        symbols=tuple(record["symbols"]), label_access=record["label_access"],
        source_sha256=record["source_sha256"], paper_module=record["paper_module"],
        parity_oracle=record["parity_oracle"],
    ) for record in payload)
    if not nodes or len({node.node_id for node in nodes}) != len(nodes):
        raise ValueError("lineage node ids must be non-empty and unique")
    return nodes
```

`codemap_paper.json`은 새 paper module 각각에 대해 `callers`, `effects`, `tests`, `source_origins`를 기록하고, HTML은 같은 전체 graph를 사람이 읽을 수 있게 렌더링한다. lock은 lineage/config와 HTML/JSON 각각의 SHA256를 strict JSON으로 고정한다. 세 파일 가운데 하나라도 누락되거나 hash가 다르면 테스트가 실패해야 한다.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection
git commit -m "feat: lock clean paper rerun lineage"
```

### Task 2: Values-only data boundary and lazy label loader

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/data_paper.py`
- Create: `paper/equipment_anomaly_detection/label_access_paper.py`
- Modify: `paper/equipment_anomaly_detection/contracts_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_data_boundary_paper.py`

**Interfaces:**
- Consumes: `FROZEN_CONFIG`, read-only SQLite DB.
- Produces: `load_population(db_path: Path) -> tuple[str, ...]`, `load_dataset_values(db_path: Path, dataset: str) -> RawDataset`, `load_test_labels(db_path: Path, datasets: tuple[str, ...]) -> Mapping[str, tuple[int, ...]]`.

- [ ] **Step 1: Write failing SQL and type-boundary tests**

```python
class DataBoundaryTests(unittest.TestCase):
    def _replace_label(self, replacement: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE instances SET label=? WHERE dataset_id="
                "(SELECT id FROM datasets WHERE name='toy') "
                "AND split='TEST' AND instance_index=1",
                (replacement,),
            )

    def test_test_value_sql_cannot_select_label(self) -> None:
        observed: list[str] = []
        values = load_dataset_values(self.db_path, "toy", sql_observer=observed.append)
        self.assertEqual(values.test_indices, (0, 1))
        self.assertTrue(all("label" not in sql.casefold() for sql in observed if "TEST" in sql))

    def test_label_loader_accepts_only_exact_text_binary(self) -> None:
        self.assertEqual(load_test_labels(self.db_path, ("toy",))["toy"], (0, 1))
        self._replace_label(" 1")
        with self.assertRaisesRegex(ValueError, "exact TEXT"):
            load_test_labels(self.db_path, ("toy",))
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_data_boundary_paper -v`

Expected: FAIL because the new data boundary is absent.

- [ ] **Step 3: Implement descriptor-pinned values loading**

```python
@dataclass(frozen=True, slots=True)
class RawDataset:
    dataset: str
    train_indices: tuple[int, ...]
    train_blobs: tuple[bytes, ...]
    train_rows: tuple[np.ndarray, ...]
    test_indices: tuple[int, ...]
    test_blobs: tuple[bytes, ...]
    test_rows: tuple[np.ndarray, ...]

TEST_VALUES_SQL = """SELECT i.instance_index, i.values_blob
FROM instances i JOIN datasets d ON d.id=i.dataset_id
WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index"""
```

TRAIN은 `split='TRAIN' AND label='0'`으로 정상 행만 읽는다. TEST query에는 `label`과 `labels_blob` token이 없어야 한다. blob은 little-endian float32로 복사하고 원 bytes도 exact-normal guard를 위해 보존한다.

- [ ] **Step 4: Implement lazy exact TEXT label conversion**

```python
def _convert(raw: object) -> int:
    if type(raw) is not str or raw not in {"0", "1"}:
        raise ValueError("TEST label must be exact TEXT '0' or '1'")
    return 0 if raw == "0" else 1
```

`data_paper.py`는 `label_access_paper.py`를 import하지 않는다. 테스트는 `sys.modules`에서 eager import가 없음을 확인한다.

- [ ] **Step 5: Run focused and existing boundary tests, then commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_data_boundary_paper experiments.paper_public_baselines.tests.test_data_boundary experiments.paper_public_baselines.tests.test_evaluation_boundary -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection
git commit -m "feat: add clean rerun data boundary"
```

### Task 3: Continuous evidence and route-preserving score math

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/calibration_paper.py`
- Create: `paper/equipment_anomaly_detection/score_adapter_paper.py`
- Modify: `paper/equipment_anomaly_detection/contracts_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_calibration_adapter_paper.py`

**Interfaces:**
- Consumes: TRAIN OOF score, TEST score, native boundary booleans.
- Produces: `upper_tail_pvalues`, `leaf_evidence`, `combine_evidence`, `route_preserving_scores`.

- [ ] **Step 1: Write RED tests for p-value equality, strict boundary, Boolean strength, and bands**

```python
class CalibrationAdapterTests(unittest.TestCase):
    def test_empirical_p_equality_passes(self) -> None:
        leaf = leaf_evidence(np.arange(99.0), np.asarray([100.0]), alpha=0.01)
        self.assertEqual(leaf.native_pass.tolist(), [True])
        self.assertGreaterEqual(leaf.margin[0], 0.0)

    def test_strict_raw_equality_fails(self) -> None:
        self.assertFalse(strict_native_pass(np.asarray([3.0]), np.asarray([3.0]))[0])

    def test_route_score_reproduces_alerts(self) -> None:
        scores = route_preserving_scores(np.asarray([0.1, 0.9]), np.asarray([False, True]))
        np.testing.assert_array_equal(scores >= 0.5, np.asarray([False, True]))
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_calibration_adapter_paper -v`

Expected: FAIL with missing functions.

- [ ] **Step 3: Implement finite-sample p-values and immutable evidence**

```python
@dataclass(frozen=True, slots=True)
class LeafEvidence:
    pvalue: np.ndarray
    margin: np.ndarray
    strength: np.ndarray
    native_pass: np.ndarray

def upper_tail_pvalues(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.asarray([(1 + np.count_nonzero(reference >= value)) / (len(reference) + 1)
                       for value in query], dtype=np.float64)

def leaf_evidence(reference: np.ndarray, query: np.ndarray, alpha: float) -> LeafEvidence:
    p = upper_tail_pvalues(reference, query)
    return LeafEvidence(p, np.log10(float(alpha) / p), 1.0 - p, p <= float(alpha))
```

- [ ] **Step 4: Implement AND/OR/k-of-n and score bands without index jitter**

```python
def combine_evidence(values: Sequence[np.ndarray], mode: str, k: int = 1) -> np.ndarray:
    matrix = np.vstack(values)
    if mode == "and": return np.min(matrix, axis=0)
    if mode == "or": return np.max(matrix, axis=0)
    if mode == "k_of_n": return np.sort(matrix, axis=0)[-int(k)]
    raise ValueError("unknown evidence mode")

def route_preserving_scores(strength: np.ndarray, alerts: np.ndarray) -> np.ndarray:
    u = np.asarray(strength, dtype=np.float64)
    a = np.asarray(alerts, dtype=bool)
    return np.where(a, 0.5 + 0.5 * u, 0.5 * u)
```

Missing evidence는 호출 전에 `u=0.5`와 `evidence_missing=true`로 명시하고 임의 perturbation을 금지한다.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_calibration_adapter_paper -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection
git commit -m "feat: add route preserving evidence score"
```

### Task 4: Standalone paper copies of raw Hard score sources

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/sources_paper/package_paper.py`
- Create: `paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_hard_sources_paper.py`

**Interfaces:**
- Consumes: `RawDataset`, `LineageNode`, fixed seeds and source configs.
- Produces: `score_hard_sources(values: RawDataset) -> HardSourceBundle` containing TRAIN OOF, TEST score, p-value, threshold, native pass, eligibility for every source.

- [ ] **Step 1: Write failing label-free deterministic source tests**

```python
class HardSourceTests(unittest.TestCase):
    def test_source_bundle_is_deterministic_and_full_length(self) -> None:
        first = score_hard_sources(self.values)
        second = score_hard_sources(self.values)
        self.assertEqual(first.source_names, EXPECTED_HARD_SOURCES)
        for name in first.source_names:
            np.testing.assert_array_equal(first.by_name[name].test_score,
                                          second.by_name[name].test_score)
            self.assertEqual(len(first.by_name[name].test_score), len(self.values.test_indices))

    def test_adapter_module_never_imports_label_loader(self) -> None:
        self.assertNotIn("paper.equipment_anomaly_detection.label_access_paper", sys.modules)
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_hard_sources_paper -v`

Expected: FAIL because `score_hard_sources` is absent.

- [ ] **Step 3: Copy the four historical raw scorers without their CSV/label wrappers**

```python
RAW_PRIMARY_SOURCES = ("rocket_exp40", "exp55_best", "exp56_best", "exp84")
EXPECTED_HARD_SOURCES = RAW_PRIMARY_SOURCES + ("rocket_block_b", "canonical_normal_state")

def _historical_source(name: str, train: np.ndarray, test: np.ndarray, record: Mapping[str, object]):
    return crossfit_source_paper(name, train, test, record)
```

원본의 `crossfit_source`와 `source_score_pair` 과학 로직만 `hard_sources_paper.py`에 복사하고 provenance header에 원본 module, symbol, source SHA256를 기록한다. `load_test_labels`, `load_candidate_predictions`, `prediction_bundle`, `row_with_metrics`는 복사하거나 호출하지 않는다. `RawDataset`에서 길이 정렬 metadata를 만든다. 원 source config는 Exp40 `rocket_256_knn3_local_gap`, Exp55 `train_global_minmax_clip_spectrogram_32_pca32_knn3`, Exp56 `glcm_rp_32_pca32_knn3`, Exp84 `aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3`다. paper scientific module의 AST에는 `experiments.exp` runtime import가 없어야 한다.

- [ ] **Step 4: Add Block-B and canonical normal-state sources**

```python
block_b = crossfit_confirmation_scores_paper(train, test)
canonical_oof, canonical_test = crossfit_scores_paper(
    canonical_features_paper(train_matrix),
    canonical_features_paper(test_matrix),
    backend="knn_local_gap", seed=PAPER_SEED,
)
```

Block-B와 canonical feature/crossfit 함수도 같은 provenance 규칙으로 paper file에 복사한다. 각 source가 실행 불가능하면 원 eligibility 이유와 빈 score를 기록한다. 다른 detector로 대체하지 않는다. synthetic fixture에서 paper copy와 원본 함수의 score/native-pass parity를 확인한다.

- [ ] **Step 5: Run source tests and commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_hard_sources_paper experiments.exp177_185.test_exp177_185_core -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection
git commit -m "feat: replay raw hard score sources"
```

### Task 5: Legacy Hard route closure and exact-normal guard

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/hard_route_paper.py`
- Create: `paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_hard_route_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_minirocket_paper.py`

**Interfaces:**
- Consumes: `HardSourceBundle`, `RawDataset` blobs.
- Produces: `replay_hard_route(values, sources) -> HardRouteResult` with base lanes, Exp234 expansion, MiniROCKET promotion, exact-normal exclusion, and `automatic_hard`.

- [ ] **Step 1: Write synthetic route and exact-blob RED tests**

```python
class HardRouteTests(unittest.TestCase):
    def test_route_preserves_stage_counts_and_exact_guard(self) -> None:
        result = replay_hard_route(self.values, self.sources)
        self.assertEqual(result.exp236_hard, frozenset({1, 2}))
        self.assertEqual(result.exp253_hard, frozenset({1, 2, 3}))
        self.assertEqual(result.automatic_hard, frozenset({2, 3}))

    def test_old_coordinate_artifact_cannot_be_passed(self) -> None:
        with self.assertRaises(TypeError):
            replay_hard_route(self.values, self.sources, frozen_coordinates={9})
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_hard_route_paper paper.equipment_anomaly_detection.tests_paper.test_minirocket_paper -v`

Expected: FAIL because route replay is absent.

- [ ] **Step 3: Implement the original set/rank chain as pure functions**

`hard_route_paper.py`는 다음 stage를 순서대로 호출하고 각 중간 coordinate identity를 기록한다.

```python
def replay_hard_route(values: RawDataset, sources: HardSourceBundle) -> HardRouteResult:
    exp93 = replay_exp74d_89_90_93_119(sources, values)
    exp137 = partition_block_b_confidence(exp93, sources.by_name["rocket_block_b"])
    exp234 = minimum_source_expansion(exp137, sources, required=3, alpha=0.01)
    exp236 = union_three_source_hard(exp137, exp234)
    mini = score_minirocket_confirmations(values)
    exp253 = exp236.hard | (exp236.standard & (mini.k1024 | mini.k4096))
    automatic = exact_normal_guard(exp253, values.train_blobs, values.test_blobs)
    return HardRouteResult(exp137, exp236, exp253, automatic, sources.continuous_strength)
```

Exp74d→89→90→93→119 route의 pure rank/set 함수와 고정 selector constant를 provenance와 함께 `hard_route_paper.py`로 복사한다. runtime에는 해당 내부 Exp package import가 없어야 하고 원본 route 함수와 synthetic set parity를 검사한다. CSV row loader와 metric writer는 복사하지 않는다. 중간 전체 parity 목표는 Exp137 Hard `2005`, Exp236 Hard `2116`, Exp253 Hard `2253`, exact guard 이후 Hard `2218`이다.

- [ ] **Step 4: Implement raw MiniROCKET two-seed intersection**

```python
def score_minirocket_confirmations(values: RawDataset) -> MiniRocketResult:
    configs = ((1024, 1008), (4096, 4032))
    # score_configuration for seeds 20260721 and 20261721
    # candidate is p<=0.005 under both seeds
```

원본 `score_configuration`과 필요한 pure helper를 `minirocket_paper.py`로 복사해 두 seed OOF/TEST score를 새로 만들고 `p <= 0.005` 교집합을 계산한다. runtime에는 `experiments.exp249_250...` import가 없어야 하며 synthetic fixture에서 원본과 paper copy의 score/candidate parity를 검사한다. 기존 Exp249 prediction JSONL은 읽지 않는다.

- [ ] **Step 5: Add label-free parity smoke against old artifacts**

새로 계산한 10-dataset smoke output을 먼저 canonical bytes로 freeze한 후에만 기존 Exp137/236/253/291 좌표와 비교한다. 비교 함수 signature는 `compare_parity(new: Mapping[str, frozenset[int]], oracle_path: Path) -> ParityReport`이고 새 route 함수에 oracle path를 전달할 수 없게 한다.

- [ ] **Step 6: Run route tests and commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_hard_route_paper paper.equipment_anomaly_detection.tests_paper.test_minirocket_paper experiments.exp253_guarded_standard_to_hard_expansion.tests.test_prediction experiments.exp290_train_raw_exact_normal_guard.tests.test_prediction -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection
git commit -m "feat: reconstruct guarded hard route"
```

### Task 6: Full TEST low-frequency replay and 30-candidate selection

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py`
- Modify: `paper/equipment_anomaly_detection/contracts_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_low_frequency_paper.py`

**Interfaces:**
- Consumes: `RawDataset`, frozen Hard coordinates produced in Task 5.
- Produces: `score_low_frequency(values, hard) -> LowFrequencyResult` with every TEST row's depth evidence, missing reason, provisional status, and selected 30 candidates.

- [ ] **Step 1: Write RED tests for strict consensus and exact selector order**

```python
class LowFrequencyTests(unittest.TestCase):
    def test_equal_threshold_is_not_a_hit(self) -> None:
        self.assertFalse(strict_two_depth_pass(
            np.asarray([3.0, 4.0]), np.asarray([3.0, 4.0])))

    def test_hard_empty_one_per_dataset_global_30(self) -> None:
        selected = select_source_candidates(self.candidates, self.hard, budget=30)
        self.assertLessEqual(len(selected), 30)
        self.assertEqual(len({row.dataset for row in selected}), len(selected))
        self.assertTrue(all(not self.hard[row.dataset] for row in selected))
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_low_frequency_paper -v`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Copy exact low-frequency numerical primitives and run them over all TEST rows**

```python
group = fit_group_calibration_paper(train_group)
depth_scores, candidate = score_test_row_paper(
    dataset, instance_index, test_row, group
)

def strict_two_depth_pass(scores: np.ndarray, thresholds: np.ndarray) -> bool:
    return bool(np.count_nonzero(np.asarray(scores) > np.asarray(thresholds)) >= 2)
```

Exp440의 feature/calibration/scoring/budget pure 함수와 frozen constants를 `low_frequency_paper.py`에 provenance와 함께 복사한다. runtime에는 `experiments.exp440...` import가 없어야 하며 inherited fixture에서 원본과 paper copy의 score/candidate parity를 확인한다. 정상 TRAIN은 raw length별 exact group, whole-row nonfinite 제외, 최소 4행을 유지한다. scoreable하지 않은 TEST 행도 row record를 만들고 `evidence_missing=true`, `missing_reason`을 저장한다. 원 depth score, threshold, strict pass, empirical rank와 second excess/rank를 보존한다.

- [ ] **Step 4: Implement original Hard-empty and budget ordering**

dataset별 key는 `(-second_excess_ratio, second_empirical_rank, instance_index)`, 전역 key는 `(-second_excess_ratio, second_empirical_rank, dataset, instance_index)`다. Hard가 비어 있는 dataset만 허용하고 1개씩 고른 뒤 상위 30개를 선택한다.

- [ ] **Step 5: Run focused and inherited Exp440 tests, then commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_low_frequency_paper experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_features_antigravity experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_scoring_antigravity experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_budget_antigravity -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection
git commit -m "feat: replay full low frequency evidence"
```

### Task 7: Stability replay and final four policy routes

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/sources_paper/stability_paper.py`
- Create: `paper/equipment_anomaly_detection/routing_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_stability_paper.py`

**Interfaces:**
- Consumes: selected 30 low-frequency candidates and raw normal TRAIN groups.
- Produces: `classify_stability(...) -> StabilityPartition`, `build_policies(...) -> PolicyBundle`.

- [ ] **Step 1: Write RED tests for 7/8 survival and exact partitions**

```python
class StabilityTests(unittest.TestCase):
    def test_fourteen_of_sixteen_is_stable(self) -> None:
        partition = classify_survival((True,) * 14 + (False,) * 2)
        self.assertTrue(partition)

    def test_policies_partition_source_candidates(self) -> None:
        policies = build_policies(frozenset({1}), frozenset({2, 3}), frozenset({4}))
        self.assertEqual(policies.hard_plus_source, frozenset({1, 2, 3, 4}))
        self.assertEqual(policies.hard_plus_stable, frozenset({1, 2, 3}))
        self.assertEqual(policies.hard_plus_unstable, frozenset({1, 4}))
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_stability_paper -v`

Expected: FAIL because stability adapter and policy bundle are absent.

- [ ] **Step 3: Copy Exp444 omission masks and candidate replay into the paper module**

```python
masks = generate_omission_masks_paper(dataset, raw_length, train_indices)
required = required_survival_count_paper(len(masks))
stable = sum(view.survived for view in views) >= required
```

Exp444의 omission mask, survival threshold, candidate replay pure 함수를 `stability_paper.py`에 provenance와 함께 복사한다. runtime에는 `experiments.exp444...` import가 없어야 하며 원본 fixture parity를 검사한다. 각 view는 retained TRAIN만으로 paper low-frequency calibration을 새로 만들고 원 candidate 한 행만 점수화한다. 원 30개 외 행에 stability를 확장하지 않는다.

- [ ] **Step 4: Build exact four policies and route-preserving scores**

```python
POLICY_ORDER = ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")
EXPECTED_COUNTS = (2218, 2248, 2244, 2222)
```

Hard와 low-frequency continuous strength의 elementwise max를 evidence strength로 사용하고, 각 policy alert mask마다 `route_preserving_scores`를 별도로 저장한다.

- [ ] **Step 5: Run focused and inherited stability tests, then commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_stability_paper experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_masks_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_stability_antigravity -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection
git commit -m "feat: build stable proposal policy routes"
```

### Task 8: Create-only artifacts, resume, CLI, and parity gate

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/artifacts_paper.py`
- Create: `paper/equipment_anomaly_detection/runner_paper.py`
- Create: `paper/equipment_anomaly_detection/cli_paper.py`
- Create: `paper/equipment_anomaly_detection/run_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_artifacts_runner_paper.py`

**Interfaces:**
- Consumes: tasks 1–7.
- Produces: CLI `prepare`, `score`, `freeze`, `parity`; `verify_score_root(root: Path) -> FrozenProposalManifest`.

- [ ] **Step 1: Write RED tests for no-clobber, resume, tamper rejection, and label firewall**

```python
class ArtifactRunnerTests(unittest.TestCase):
    def test_freeze_rejects_incomplete_inventory(self) -> None:
        with self.assertRaisesRegex(ValueError, "coverage"):
            freeze_score_root(self.root)

    def test_parity_failure_does_not_open_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "parity"):
            verify_parity(self.root)
        self.assertNotIn("paper.equipment_anomaly_detection.label_access_paper", sys.modules)
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper -v`

Expected: FAIL because lifecycle code is absent.

- [ ] **Step 3: Implement atomic dataset shards and exact-prefix resume**

Shard schema에는 dataset, TRAIN/TEST identity, every-source score/p/strength/native pass, Hard route stages, low-frequency evidence, stability result, four policies, backend decision만 허용한다. `label`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `auroc`, `average_precision` field가 있으면 publication을 거절한다.

- [ ] **Step 4: Implement freeze-before-parity lifecycle**

```python
def freeze_score_root(root: Path) -> FrozenProposalManifest:
    verify_exact_1117_shards(root)
    verify_total_test_rows(root, 195900)
    return write_create_only_manifest(root)

def verify_parity(root: Path) -> ParityReport:
    manifest = verify_score_root(root)
    report = compare_all_four_policies_with_oracles(manifest)
    if not report.exact:
        raise ValueError("proposal parity failed")
    if not manifest.limited_for_smoke and report.alert_counts != (2218, 2248, 2244, 2222):
        raise ValueError("full proposal alert counts differ")
    return write_create_only_parity(root, report)
```

- [ ] **Step 5: Add closed CLI and launcher**

```text
prepare --db-path DB --run-root ROOT [--limit N]
assess-acceleration --run-root ROOT
score --run-root ROOT --workers auto
freeze --run-root ROOT
parity --run-root ROOT
```

`evaluate`는 Task 11, `report`는 Task 12에서 lazy import로 추가한다.

- [ ] **Step 6: Run tests and commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection paper/equipment_anomaly_detection/run_paper.py
git commit -m "feat: freeze clean proposal score lifecycle"
```

### Task 9: PyTorch/MLX/MPS assessment and truthful CPU fallback

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/acceleration_paper.py`
- Modify: `paper/equipment_anomaly_detection/cli_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_acceleration_paper.py`

**Interfaces:**
- Consumes: exact dense-distance stages from hard source and low-frequency adapters.
- Produces: `assess_stage(stage_id, cpu, candidates, smoke_input) -> tuple[AccelerationRecord, ...]`, `select_cpu_workers() -> WorkerRecord`, CLI `assess-acceleration`.

- [ ] **Step 1: Write RED tests for every MLX gate**

```python
class AccelerationTests(unittest.TestCase):
    def test_numeric_or_candidate_mismatch_selects_cpu(self) -> None:
        record = assess_stage("dense_distance", self.cpu, self.bad_mlx, self.input)
        self.assertEqual(record.selected_backend, "cpu")
        self.assertFalse(record.candidate_parity)

    def test_speed_below_1p1_selects_cpu(self) -> None:
        record = decide_backend(cpu_seconds=1.0, mlx_seconds=0.95,
                                finite=True, candidate_parity=True, max_abs_error=0.0)
        self.assertEqual(record.selected_backend, "cpu")

    def test_cpu_worker_count_is_logical_cpu_minus_one(self) -> None:
        record = select_cpu_workers(detected_logical_cpus=12)
        self.assertEqual(record.dataset_workers, 11)

    def test_assessment_names_all_three_acceleration_families(self) -> None:
        self.assertEqual({row.backend_family for row in self.records},
                         {"pytorch", "mlx", "mps"})
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_acceleration_paper -v`

Expected: FAIL because acceleration records are absent.

- [ ] **Step 3: Implement exact policy checks**

`AccelerationRecord`는 stage, backend family, tensor eligibility, availability, transfer/compile overhead, CPU/device seconds, speedup, finite, max abs error, alpha별 candidate SHA, versions, device, selected backend와 fallback reason을 갖는다. PyTorch, MLX, MPS 세 family 모두 record를 만들고 official aeon/scikit-learn/PyTorch transformer를 다시 작성해야 하는 backend는 `algorithm_rewrite_required` 사유로 CPU를 선택한다. `WorkerRecord`에는 감지 logical CPU와 `max(1, cpu_count - 1)` worker를 기록한다.

- [ ] **Step 4: Add `assess-acceleration` gate before score**

`score`는 `acceleration_paper.json`과 SHA256가 없거나 lineage/config hash가 다르면 실행을 거절한다. official run의 `--workers auto`는 frozen `WorkerRecord`와 일치해야 하며 임의 worker override를 거절한다. MLX/MPS heavy process는 worker 1을 강제한다.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_acceleration_paper experiments.paper_public_baselines.tests.test_acceleration -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection
git commit -m "feat: gate clean rerun acceleration"
```

### Task 10: Public baseline bridge and fresh score rerun

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/baseline_bridge_paper.py`
- Modify: `paper/equipment_anomaly_detection/cli_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_baseline_bridge_paper.py`
- Do not modify: `experiments/paper_public_baselines/`

**Interfaces:**
- Consumes: verified proposal score root and one or more verified fresh public baseline score roots.
- Produces: `freeze_baseline_policy_scores(proposal_root, baseline_roots, output_root) -> FrozenBaselinePolicyManifest`, `run_deep_budgeted(jobs, budget_seconds, epochs) -> DeepBudgetReport`.

- [ ] **Step 1: Write RED tests proving counts-only matched budget and common adapter**

```python
class BaselineBridgeTests(unittest.TestCase):
    def test_matched_budget_never_copies_proposal_coordinates(self) -> None:
        result = build_baseline_policy(self.baseline, proposal_alerts=frozenset({7, 9}))
        self.assertEqual(len(result.alerts), 2)
        self.assertNotEqual(frozenset(result.alerts), frozenset({7, 9}))

    def test_policy_score_uses_one_minus_empirical_p(self) -> None:
        result = build_baseline_policy(self.baseline, proposal_alerts=frozenset({1}))
        expected = route_preserving_scores(1.0 - result.pvalues, result.alert_mask)
        np.testing.assert_array_equal(result.route_score, expected)

    def test_deep_controller_never_starts_after_shared_deadline(self) -> None:
        report = run_deep_budgeted(self.jobs, budget_seconds=10, epochs=50,
                                   clock=self.fake_clock, run=self.fake_run)
        self.assertEqual(report.started_methods, ("deep_svdd_1d",))
        self.assertEqual(report.excluded_methods, ("drocc_1d",))
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_baseline_bridge_paper -v`

Expected: FAIL because the bridge is absent.

- [ ] **Step 3: Implement verified shard conversion without labels**

각 completed baseline shard의 `train_oof_scores`, `test_scores`, `test_indices`를 읽고 empirical p-value와 native strength를 만든다. proposal에서는 dataset별 alert count만 읽는다. conformal alpha `0.005`, `0.01`, `0.02`와 matched-budget alert를 모두 저장한다.

고전 네 방법은 하나의 root에, 딥 방법은 방법별 root에 준비한다. `run_deep_budgeted`는 `time.monotonic()`의 단일 시작시각과 `172800`초 deadline을 공유하고, 각 public CLI subprocess에 남은 시간만 `timeout`으로 준다. 완주한 딥 root만 즉시 freeze하며 timeout/실패/미시작 root는 bridge 입력에서 제외한다. 과학 코드는 호출만 하고 수정하지 않는다.

`cli_paper.py`에 반복 가능한 `--baseline-root`를 받는 `bridge-baselines`, 그리고 `run-deep-budget --deep-svdd-root ROOT --drocc-root ROOT --budget-seconds 172800 --epochs 50 --report-path PATH`를 추가한다. bridge는 모든 입력 root의 frozen manifest와 method 집합이 중복되지 않는지 확인한 뒤 output을 create-only로 동결한다. budget report에는 시작/종료 monotonic elapsed, method별 subprocess exit/timeout, frozen manifest SHA, 포함/제외 사유를 기록한다.

- [ ] **Step 4: Run a fresh public baseline smoke root**

```bash
python3 -m experiments.paper_public_baselines.cli prepare --db-path /Users/mac/Documents/Dataset/univariate_ts.db --run-root /Users/mac/Documents/Dataset/paper_public_baselines_clean_20260812_smoke --limit 10 --methods knn_distance,lof_novelty,ocsvm_rbf,isolation_forest,deep_svdd_1d,drocc_1d
python3 -m experiments.paper_public_baselines.cli assess-mlx --run-root /Users/mac/Documents/Dataset/paper_public_baselines_clean_20260812_smoke
PAPER_CPU_WORKERS=$(python3 -c 'import os; print(max(1, (os.cpu_count() or 1) - 1))')
python3 -m experiments.paper_public_baselines.cli score --run-root /Users/mac/Documents/Dataset/paper_public_baselines_clean_20260812_smoke --methods knn_distance,lof_novelty,ocsvm_rbf,isolation_forest --workers "$PAPER_CPU_WORKERS"
python3 -m experiments.paper_public_baselines.cli score --run-root /Users/mac/Documents/Dataset/paper_public_baselines_clean_20260812_smoke --methods deep_svdd_1d,drocc_1d --workers 1 --epochs 50
python3 -m experiments.paper_public_baselines.cli freeze --run-root /Users/mac/Documents/Dataset/paper_public_baselines_clean_20260812_smoke
```

Expected: frozen 60-pair smoke manifest with zero label access.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_baseline_bridge_paper -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection
git commit -m "feat: bridge fresh public baseline scores"
```

### Task 11: Post-freeze evaluation and paired statistics

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/evaluation_paper.py`
- Create: `paper/equipment_anomaly_detection/statistics_paper.py`
- Create: `paper/equipment_anomaly_detection/independent_reaggregate_paper.py`
- Modify: `paper/equipment_anomaly_detection/cli_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_evaluation_statistics_paper.py`

**Interfaces:**
- Consumes: verified proposal manifest, parity success, verified baseline-policy manifest.
- Produces: `evaluate_frozen_roots(...) -> EvaluationBundle`, `paired_bootstrap(...) -> tuple[BootstrapRow, ...]`, `independent_reaggregate.main(argv=None) -> int`.

- [ ] **Step 1: Write RED tests for label order, AP definition, undefined macro, and freeze gate**

```python
class EvaluationStatisticsTests(unittest.TestCase):
    def test_labels_are_not_called_before_parity_success(self) -> None:
        loader = Mock()
        with self.assertRaisesRegex(ValueError, "parity"):
            evaluate_frozen_roots(self.bad_proposal, self.baseline, label_loader=loader)
        loader.assert_not_called()

    def test_average_precision_is_not_trapezoidal_pr_auc(self) -> None:
        row = evaluate_scores(np.asarray([1, 0, 1]), np.asarray([0.9, 0.8, 0.1]))
        self.assertAlmostEqual(row.average_precision, average_precision_score([1, 0, 1], [0.9, 0.8, 0.1]))

    def test_label_indices_must_match_frozen_score_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "index order"):
            join_score_labels((0, 1), (1, 0), (1, 0))

    def test_single_class_macro_auroc_is_none(self) -> None:
        row = evaluate_dataset(np.asarray([0, 0]), np.asarray([0.2, 0.1]), np.asarray([False, False]))
        self.assertIsNone(row.auroc)
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_evaluation_statistics_paper -v`

Expected: FAIL because evaluator and bootstrap are absent.

- [ ] **Step 3: Implement exact binary and score metrics**

각 policy/method/protocol/support에 TP, FP, FN, TN, precision, recall, specificity, FPR, micro F1, macro precision/recall/F1/FPR, pooled AP/AUROC, macro AP/AUROC, score coverage, tied/missing row 수를 저장한다. `binary-policy`, `route-preserving`, `native` score type을 별도 field로 둔다.

평가기는 label을 연 직후 dataset/index/label과 각 비교 대상의 alert/score를 canonical `score_label_rows.jsonl`에 create-only로 기록한다. 따라서 pooled AP/AUROC를 SQLite 재접속 없이 독립 재계산할 수 있다. dataset aggregate와 bootstrap 결과도 별도 canonical JSONL로 기록하고 artifact map에 모두 포함한다.

- [ ] **Step 4: Implement shared-draw dataset bootstrap**

```python
rng = np.random.Generator(np.random.PCG64(20260812))
draws = rng.integers(0, 1117, size=(20000, 1117), dtype=np.int64, endpoint=False)
```

draw bytes SHA256를 저장하고 모든 제안기술-minus-baseline 차이에 같은 draw matrix를 사용한다. source별 native score 중 최고 TEST 값을 대표값으로 선택하는 함수는 만들지 않는다.

- [ ] **Step 5: Add lazy `evaluate` command and run tests**

`cli_paper.py`의 evaluate branch 안에서만 `evaluation`과 `label_access`를 import한다.

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_evaluation_statistics_paper experiments.paper_public_baselines.tests.test_statistics -v`

Expected: PASS.

- [ ] **Step 6: Implement source-independent reaggregation**

`independent_reaggregate_paper.py`는 score source module과 SQLite를 import하지 않고 `score_label_rows.jsonl`, dataset/bootstrap JSONL, bootstrap config와 artifact map만 읽는다. row-level label/score에서 pooled AP/AUROC와 confusion을, dataset row에서 macro 지표를 다시 계산한다. `PCG64(20260812)` draw도 재생성해 bytes SHA와 interval을 확인하고 불일치 시 nonzero exit를 낸다. CLI는 `main(argv: Sequence[str] | None = None) -> int`와 `--run-root`만 노출한다.

- [ ] **Step 7: Commit**

```bash
git add paper/equipment_anomaly_detection
git commit -m "feat: evaluate frozen paper rerun scores"
```

### Task 12: Paper reporting, preregistration, and runbook

**Files:**
- Modify: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify: `paper/equipment_anomaly_detection/codemap_paper.lock`
- Create: `paper/equipment_anomaly_detection/reporting_paper.py`
- Create: `paper/equipment_anomaly_detection/docs_paper/preregistration_paper_ko.md`
- Create: `paper/equipment_anomaly_detection/docs_paper/runbook_paper_ko.md`
- Create: `paper/equipment_anomaly_detection/docs_paper/result_paper_ko.md`
- Modify: `paper/equipment_anomaly_detection/cli_paper.py`
- Test: `paper/equipment_anomaly_detection/tests_paper/test_reporting_cli_paper.py`

**Interfaces:**
- Consumes: immutable evaluation bundle.
- Produces: tables for proposal ablation, public decision comparison, AP/AUROC score types, bootstrap intervals, coverage/failures, and limitation text.

- [ ] **Step 1: Write RED tests for public names and mandatory limitations**

```python
class ReportingCliTests(unittest.TestCase):
    def test_main_table_has_no_internal_experiment_names(self) -> None:
        text = render_main_report(self.bundle)
        self.assertNotIn("Exp448", text)
        self.assertNotIn("Stable-26", text)
        self.assertIn("Proposed method", text)

    def test_report_discloses_score_adapter_and_retrospective_scope(self) -> None:
        text = render_main_report(self.bundle)
        self.assertIn("route-preserving continuous-evidence", text)
        self.assertIn("retrospective", text.casefold())
        self.assertIn("public benchmark", text.casefold())
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_reporting_cli_paper -v`

Expected: FAIL because reporting is absent.

- [ ] **Step 3: Implement create-only reports and verify-only replay**

보고서에는 다음 순서의 표를 만든다.

1. Proposed method four-way ablation
2. Matched-alert-budget binary decision comparison
3. Binary-policy AP/AUROC
4. Route-preserving AP/AUROC
5. Public native score and proposal component score appendix
6. Coverage, abstention, runtime and deep-method completion
7. Paired descriptive intervals and limitations

- [ ] **Step 4: Write exact operator commands and stop conditions**

runbook에는 root absent 확인, prepare, assess-mlx, score, freeze, parity, baseline bridge, evaluate, report, verify-only 순서를 적는다. parity 실패, deep partial, TEST-label pre-access와 source coverage failure의 중단 행동을 명시한다.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_reporting_cli_paper -v`

Expected: PASS.

```bash
git add paper/equipment_anomaly_detection paper/equipment_anomaly_detection/docs_paper/preregistration_paper_ko.md paper/equipment_anomaly_detection/docs_paper/runbook_paper_ko.md paper/equipment_anomaly_detection/docs_paper/result_paper_ko.md
git commit -m "docs: add clean paper rerun lifecycle"
```

### Task 13: End-to-end verification and official fresh execution

**Files:**
- Modify only if generated summary hashes need recording: `paper/equipment_anomaly_detection/docs_paper/result_paper_ko.md`
- No large runtime artifact is added to Git.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: complete proposal root, frozen classical root, zero to two frozen completed-deep roots, post-freeze evaluation and paper-ready result hashes.

- [ ] **Step 1: Run the full new package test suite**

Run: `python3 -m unittest discover -s paper/equipment_anomaly_detection/tests_paper -p 'test_*_paper.py' -v`

Expected: all tests PASS with zero failures.

- [ ] **Step 2: Run inherited regression suites**

Run: `python3 -m unittest discover -s experiments/paper_public_baselines/tests -v`

Run: `python3 -m unittest experiments.exp249_250_official_mini_multirocket.tests.test_scoring experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_scoring_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_stability_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_evaluation_antigravity -v`

Expected: all tests PASS.

- [ ] **Step 3: Create and verify the proposal smoke run**

먼저 `test ! -e /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812_smoke`가 exit 0인지 확인한다.

```bash
python3 paper/equipment_anomaly_detection/run_paper.py prepare --db-path /Users/mac/Documents/Dataset/univariate_ts.db --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812_smoke --limit 10
python3 paper/equipment_anomaly_detection/run_paper.py assess-acceleration --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812_smoke
python3 paper/equipment_anomaly_detection/run_paper.py score --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812_smoke --workers auto
python3 paper/equipment_anomaly_detection/run_paper.py freeze --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812_smoke
python3 paper/equipment_anomaly_detection/run_paper.py parity --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812_smoke
```

Expected: 10/10 shards frozen, no label access, exact smoke parity.

- [ ] **Step 4: Run and freeze the official 1,117-dataset proposal root**

먼저 `test ! -e /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812`가 exit 0인지 확인한다.

```bash
python3 paper/equipment_anomaly_detection/run_paper.py prepare --db-path /Users/mac/Documents/Dataset/univariate_ts.db --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812
python3 paper/equipment_anomaly_detection/run_paper.py assess-acceleration --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812
python3 paper/equipment_anomaly_detection/run_paper.py score --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812 --workers auto
python3 paper/equipment_anomaly_detection/run_paper.py freeze --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812
python3 paper/equipment_anomaly_detection/run_paper.py parity --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812
```

Expected: 1,117 shards, 195,900 TEST rows, policy counts `2218/2248/2244/2222`, exact coordinate parity, TEST-label access 0.

- [ ] **Step 5: Run fresh public baselines**

Use classical root `/Users/mac/Documents/Dataset/paper_public_baselines_classical_20260812`, Deep SVDD root `/Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_20260812`, and DROCC root `/Users/mac/Documents/Dataset/paper_public_baselines_drocc_20260812`. Classical methods run with 7 workers. Deep SVDD and DROCC run sequentially with 1 worker and the preregistered 50 epochs under one controller-enforced 48-hour deadline. Only roots completed and frozen inside that deadline are eligible for the main comparison; excluded methods are listed with their reason.

먼저 세 root와 budget report가 모두 없는지 확인한다.

```bash
test ! -e /Users/mac/Documents/Dataset/paper_public_baselines_classical_20260812
test ! -e /Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_20260812
test ! -e /Users/mac/Documents/Dataset/paper_public_baselines_drocc_20260812
test ! -e /Users/mac/Documents/Dataset/paper_public_baselines_deep_budget_20260812.json
```

```bash
python3 -m experiments.paper_public_baselines.cli prepare --db-path /Users/mac/Documents/Dataset/univariate_ts.db --run-root /Users/mac/Documents/Dataset/paper_public_baselines_classical_20260812 --methods knn_distance,lof_novelty,ocsvm_rbf,isolation_forest
python3 -m experiments.paper_public_baselines.cli assess-mlx --run-root /Users/mac/Documents/Dataset/paper_public_baselines_classical_20260812
PAPER_CPU_WORKERS=$(python3 -c 'import os; print(max(1, (os.cpu_count() or 1) - 1))')
python3 -m experiments.paper_public_baselines.cli score --run-root /Users/mac/Documents/Dataset/paper_public_baselines_classical_20260812 --methods knn_distance,lof_novelty,ocsvm_rbf,isolation_forest --workers "$PAPER_CPU_WORKERS"
python3 -m experiments.paper_public_baselines.cli freeze --run-root /Users/mac/Documents/Dataset/paper_public_baselines_classical_20260812
python3 -m experiments.paper_public_baselines.cli prepare --db-path /Users/mac/Documents/Dataset/univariate_ts.db --run-root /Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_20260812 --methods deep_svdd_1d
python3 -m experiments.paper_public_baselines.cli assess-mlx --run-root /Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_20260812
python3 -m experiments.paper_public_baselines.cli prepare --db-path /Users/mac/Documents/Dataset/univariate_ts.db --run-root /Users/mac/Documents/Dataset/paper_public_baselines_drocc_20260812 --methods drocc_1d
python3 -m experiments.paper_public_baselines.cli assess-mlx --run-root /Users/mac/Documents/Dataset/paper_public_baselines_drocc_20260812
python3 paper/equipment_anomaly_detection/run_paper.py run-deep-budget --deep-svdd-root /Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_20260812 --drocc-root /Users/mac/Documents/Dataset/paper_public_baselines_drocc_20260812 --budget-seconds 172800 --epochs 50 --report-path /Users/mac/Documents/Dataset/paper_public_baselines_deep_budget_20260812.json
```

- [ ] **Step 6: Freeze baseline policy scores, evaluate once, and report**

```bash
python3 paper/equipment_anomaly_detection/run_paper.py bridge-baselines --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812 --baseline-root /Users/mac/Documents/Dataset/paper_public_baselines_classical_20260812 --include-completed-deep-from /Users/mac/Documents/Dataset/paper_public_baselines_deep_budget_20260812.json
python3 paper/equipment_anomaly_detection/run_paper.py evaluate --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812 --baseline-policy-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812/baseline_policy_scores
python3 paper/equipment_anomaly_detection/run_paper.py report --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812
python3 paper/equipment_anomaly_detection/run_paper.py report --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812 --verify-only
```

Expected: one immutable evaluation, identical verify-only report hash, AP/AUROC score types and coverage explicitly separated.

- [ ] **Step 7: Independently reaggregate and record small hashes**

Run: `python3 -m paper.equipment_anomaly_detection.independent_reaggregate_paper --run-root /Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812`

Expected: every confusion matrix, AP/AUROC row, bootstrap draw SHA and report artifact map verifies from immutable result files only. Add only the final manifest hashes and headline tables to `paper/equipment_anomaly_detection/docs_paper/result_paper_ko.md`.

- [ ] **Step 8: Run fresh completion verification and commit the small report**

Run: `python3 -m unittest discover -s paper/equipment_anomaly_detection/tests_paper -p 'test_*_paper.py' -v`

Run: `git diff --check`

Expected: zero test failures and no whitespace errors.

```bash
git add paper/equipment_anomaly_detection/docs_paper/result_paper_ko.md
git commit -m "docs: report clean proposal paper rerun"
```
