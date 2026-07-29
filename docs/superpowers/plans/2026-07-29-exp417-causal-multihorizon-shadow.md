# Exp417 Causal Multi-Horizon Shadow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 정상 TRAIN만으로 작은 causal multi-horizon CNN을 학습하고, 엄격한 지속성·seed 합의·정상 OOF·검토량 gate를 통과한 label-free Exp417 Shadow prediction을 결정론적으로 생성할 수 있는 구현과 실행 준비 상태를 만든다.

**Architecture:** Exp417은 raw values-only 입력 동결, exact-length TRAIN preprocessing, identity-grouped OOF causal 학습, series-level multi-horizon score, TRAIN-only synthetic qualification, CPU/MPS/Numba backend 감사, provenance-bound checkpoint, exact-file atomic publication을 분리된 모듈로 구현한다. 기존 Hard/Standard/Priority/automatic-normal lane은 읽기 전용이며 prediction에서 제외만 하고 수정하지 않는다. 전체 1,117개 실행은 이 계획에 포함하지 않으며, 모든 테스트와 대표 label-free smoke가 통과한 뒤 사용자의 별도 명시적 승인을 받는다.

**Tech Stack:** Python 3.14, PyTorch 2.13.0, NumPy 2.3.5, Numba 0.66.0, SQLite read-only URI, `unittest`, canonical JSON/SHA256, atomic filesystem publication; canonical backend PyTorch CPU, optional MPS, no MLX

## Global Constraints

- Design source: `docs/superpowers/specs/2026-07-29-exp417-causal-multihorizon-shadow-design.md`.
- Frozen population: exactly 1,117 datasets.
- Frozen lane totals: Hard 2,218 / Standard 806 / Priority 31 / automatic-normal exact 35.
- Database SHA256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`.
- Exp291 prediction SHA256: `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848`.
- Exp291 manifest SHA256: `761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e`.
- Exp292 prediction SHA256: `f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b`.
- Exp292 manifest SHA256: `0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0`.
- Exp417 code must not import, open, hash, stat, query, or name the frozen TEST-label artifact. TEST SQL selects values and instance identity only; it never selects `label` or `labels_blob`.
- Normal TRAIN alone determines preprocessing, folds, fitting, early stopping, calibration, eligibility, thresholds, perturbations, backend selection, and workload gates.
- TEST labels, actual anomaly locations, historical TEST/family performance, dataset/family routing, tail replacement, top-N trimming, padding, truncation, and resampling are forbidden.
- Exp417 creates only `causal_multihorizon_shadow`; it does not modify or promote Hard, Standard, Priority, or automatic-normal lanes.
- Exp414 qualified no near-repeat rule. Exp417 applies only the frozen exact-normal guard and records near-repeat guard as `not_available`.
- Exact horizons: `1`, `ceil(0.02 * L)`, `ceil(0.10 * L)`; groups without three distinct usable horizons abstain.
- Model: 2 input channels, three causal convolution blocks, 32 hidden channels, kernel 5, dilations 1/2/4, three horizon heads, Huber loss, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch `min(32, n_train)`, max 100 epochs, patience 10.
- Model seeds: `41701`, `41702`, `41703`; perturbation seeds: `41711`, `41712`, `41713`.
- Length-group eligibility: `<50` abstain; `50..99` diagnostic only; `>=100` candidate eligible after all gates.
- Candidate: at least two horizon p-values `<=0.01`, agreeing horizon persistence `>=0.05`, and all three seeds must agree.
- Qualification: normal OOF candidate rate `<=0.01`, Wilson 95% upper `<=0.05`, seed agreement 100%, pooled primary synthetic sensitivity `>=0.70`, every primary type `>=0.50`, finite outputs, no backend fallback.
- Dataset workload failure: new Shadow rate `>0.05` causes whole-dataset abstention, never trimming.
- Global workload failure: new Shadow rate `>0.01` prevents official eight-file publication.
- Canonical backend is deterministic PyTorch CPU. MPS requires exact candidate identity, normalized max error `<=1e-4`, Spearman `>=0.999`, no fallback, and end-to-end speedup `>=1.1x`. Current environment reports MPS built but unavailable; CPU remains canonical unless a later clean benchmark passes.
- Numba may accelerate deterministic score aggregation only after NumPy parity. MLX is excluded.
- Official output is a new no-overwrite directory containing exactly eight files named in Task 10.
- Dataset-derived checkpoints, predictions, metrics, and reports remain local and are not pushed to GitHub.
- No full 1,117-dataset run occurs in this implementation plan. It requires a separate explicit user instruction.

---

## File Structure

- `docs/EXP417_PREREGISTRATION_KO.md`: frozen inputs, algorithm, gates, outputs, and prohibition boundary.
- `experiments/exp417_causal_multihorizon_shadow/__init__.py`: minimal public API.
- `experiments/exp417_causal_multihorizon_shadow/config.py`: immutable exact configuration.
- `experiments/exp417_causal_multihorizon_shadow/contracts.py`: validated immutable domain records.
- `experiments/exp417_causal_multihorizon_shadow/inputs.py`: hash-gated lanes and raw values-only SQLite loading.
- `experiments/exp417_causal_multihorizon_shadow/preprocessing.py`: repair, length grouping, TRAIN scaling, identity folds, horizons.
- `experiments/exp417_causal_multihorizon_shadow/model.py`: causal CNN and deterministic OOF/final training.
- `experiments/exp417_causal_multihorizon_shadow/scoring.py`: residual normalization, persistence, empirical p-values, seed consensus.
- `experiments/exp417_causal_multihorizon_shadow/qualification.py`: fixed perturbations, safety gates, lane exclusion, workload gates.
- `experiments/exp417_causal_multihorizon_shadow/backend.py`: CPU/MPS benchmark and NumPy/Numba parity.
- `experiments/exp417_causal_multihorizon_shadow/checkpoints.py`: exact provenance and atomic per-dataset records.
- `experiments/exp417_causal_multihorizon_shadow/runner.py`: dataset and full-population orchestration without labels.
- `experiments/exp417_causal_multihorizon_shadow/reporting.py`: deterministic eight-artifact payload rendering and revalidation.
- `experiments/exp417_causal_multihorizon_shadow/publication.py`: descriptor-pinned exact-file atomic publication.
- `experiments/exp417_causal_multihorizon_shadow/tests/`: synthetic/unit/integration tests only.
- `scratch/run_exp417_causal_multihorizon_shadow.py`: official all-population one-shot CLI.
- `scratch/smoke_exp417_causal_multihorizon_shadow.py`: explicitly non-official label-free readiness smoke.
- `docs/EXP417_RUNBOOK_KO.md`: environment, tests, smoke, resume, and separately approved full-run commands.

---

### Task 1: Frozen Governance, Configuration, and Contracts

**Files:**
- Create: `docs/EXP417_PREREGISTRATION_KO.md`
- Create: `experiments/exp417_causal_multihorizon_shadow/__init__.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/config.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/contracts.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/__init__.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_config.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_contracts.py`

**Interfaces:**
- Consumes: approved Exp417 design and the exact Global Constraints above.
- Produces: `Exp417FrozenConfig`, `FROZEN_CONFIG`, `RawSeries`, `FrozenLaneSnapshot`, `RawTrainDataset`, `RawTestDataset`, `FrozenRunContext`, `FrozenTrainContract`, `LengthGroup`, `ProcessedGroup`, `HorizonScore`, `SeedDecision`, `GroupQualification`, `ShadowCandidate`, `DatasetResult`, `BackendAudit`, `RunSummary`.

- [ ] **Step 1: Write strict config tests**

```python
from dataclasses import replace
import unittest

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG


class ConfigTests(unittest.TestCase):
    def test_exact_preregistered_values(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp417")
        self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
        self.assertEqual(FROZEN_CONFIG.model_seeds, (41701, 41702, 41703))
        self.assertEqual(FROZEN_CONFIG.horizon_fractions, (0.02, 0.10))
        self.assertEqual(FROZEN_CONFIG.candidate_alpha, 0.01)
        self.assertEqual(FROZEN_CONFIG.dataset_workload_rate, 0.05)
        self.assertEqual(FROZEN_CONFIG.global_workload_rate, 0.01)
        FROZEN_CONFIG.validate()

    def test_mutation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(FROZEN_CONFIG, candidate_alpha=0.02).validate()
```

- [ ] **Step 2: Write contract rejection and conservation tests**

```python
def test_raw_series_rejects_bool_index_and_nonfinite(self) -> None:
    with self.assertRaises(ValueError):
        RawSeries(instance_index=True, values=(0.0, 1.0))
    with self.assertRaises(ValueError):
        RawSeries(instance_index=1, values=(0.0, float("nan")))

def test_dataset_result_rejects_lane_or_candidate_overlap(self) -> None:
    lanes = frozen_lanes(hard={("A", 1)})
    with self.assertRaisesRegex(ValueError, "frozen lane"):
        dataset_result("A", candidates=(candidate("A", 1),), lanes=lanes)
```

Cover exact integer types, sorted unique coordinates, finite `[0,1]` rates,
three exact horizons, three exact seeds, candidate/p-value/persistence
consistency, group-to-dataset accounting, and immutable mappings.

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_config \
  experiments.exp417_causal_multihorizon_shadow.tests.test_contracts -v
```

Expected: import failure because the package is absent.

- [ ] **Step 4: Implement exact frozen config**

```python
@dataclass(frozen=True)
class Exp417FrozenConfig:
    experiment_id: str = "Exp417"
    expected_datasets: int = 1117
    expected_hard: int = 2218
    expected_standard: int = 806
    expected_priority: int = 31
    expected_automatic_normal: int = 35
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    exp291_prediction_sha256: str = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"
    exp291_manifest_sha256: str = "761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e"
    exp292_prediction_sha256: str = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"
    exp292_manifest_sha256: str = "0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0"
    model_seeds: tuple[int, ...] = (41701, 41702, 41703)
    perturbation_seeds: tuple[int, ...] = (41711, 41712, 41713)
    horizon_fractions: tuple[float, float] = (0.02, 0.10)
    candidate_alpha: float = 0.01
    persistence_fraction: float = 0.05
    min_train_model: int = 50
    min_train_candidate: int = 100
    oof_candidate_rate: float = 0.01
    oof_wilson_upper: float = 0.05
    pooled_synthetic_sensitivity: float = 0.70
    per_type_synthetic_sensitivity: float = 0.50
    dataset_workload_rate: float = 0.05
    global_workload_rate: float = 0.01
    mps_max_error: float = 1e-4
    mps_min_spearman: float = 0.999
    mps_min_speedup: float = 1.1

    def validate(self) -> None:
        expected = type(self)()
        for field in fields(self):
            actual, wanted = getattr(self, field.name), getattr(expected, field.name)
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError(f"Exp417 frozen config mismatch: {field.name}")
```

- [ ] **Step 5: Implement immutable contracts**

Every `__post_init__` must reject bool-as-int, negative indices, empty dataset
names, unsorted/duplicate coordinates, non-finite values, wrong tuple lengths,
candidate/frozen-lane overlap, and inconsistent group/dataset totals. Use
`MappingProxyType` for mappings and tuple copies for sequences.

- [ ] **Step 6: Write preregistration before any real values smoke**

`docs/EXP417_PREREGISTRATION_KO.md` must reproduce every Global Constraint,
the exact algorithm/gates, current environment (`torch 2.13.0`,
`numpy 2.3.5`, `numba 0.66.0`, MPS built but unavailable), the eight outputs,
the Exp418 separation, GitHub export prohibition, and the separate approval
required for the full run.

- [ ] **Step 7: Run GREEN and guards**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_config \
  experiments.exp417_causal_multihorizon_shadow.tests.test_contracts -v
rg -n "8458e678|73dff56e|f02259ae|TEST label|top-N|Exp418|별도.*승인" \
  docs/EXP417_PREREGISTRATION_KO.md
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add docs/EXP417_PREREGISTRATION_KO.md \
  experiments/exp417_causal_multihorizon_shadow
git commit -m "docs: preregister Exp417 causal shadow"
```

---

### Task 2: Frozen Lanes and Raw Values-Only Input Boundary

**Files:**
- Create: `experiments/exp417_causal_multihorizon_shadow/inputs.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_inputs.py`

**Interfaces:**
- Consumes: `Exp417FrozenConfig`, `RawSeries`, `FrozenLaneSnapshot`, `FrozenRunContext`, `RawTrainDataset`, `RawTestDataset`.
- Produces:
  - `freeze_lane_inputs(exp291_prediction: Path, exp291_manifest: Path, exp292_prediction: Path, exp292_manifest: Path, *, config=FROZEN_CONFIG) -> FrozenLaneSnapshot`
  - `list_dataset_names(database: Path, lanes: FrozenLaneSnapshot, *, config=FROZEN_CONFIG) -> tuple[str, ...]`
  - `freeze_run_context(database: Path, lanes: FrozenLaneSnapshot, *, config=FROZEN_CONFIG) -> FrozenRunContext`
  - `load_raw_train_values(database: Path, dataset: str, context: FrozenRunContext) -> RawTrainDataset`
  - `load_raw_test_values(database: Path, dataset: str, frozen_train_contract: FrozenTrainContract, context: FrozenRunContext) -> RawTestDataset`
  - `assert_label_free_sql_contract() -> None`
  - `verify_input_snapshots(...) -> None`

- [ ] **Step 1: Write hash, lane parity, and automatic-normal tests**

Synthetic Exp291/292 JSONL fixtures must include exact fields:

```python
{
    "dataset": "A",
    "n_test": 5,
    "automatic_hard_indices": [0],
    "automatic_normal_exact_indices": [1],
    "standard_review_indices": [2],
    "priority_review_indices": [3],
}
```

Test independent pre/post mutation of both predictions and both manifests,
Exp291/292 lane mismatch, total mismatch, duplicate/out-of-range coordinates,
and overlap between frozen lane types.

- [ ] **Step 2: Write values-only SQLite query tests**

Create a synthetic SQLite database and trace SQL. The TEST loader requires a
real `FrozenTrainContract`; passing `object()` must fail before opening the
database:

```python
def test_test_query_never_selects_labels(self) -> None:
    bundle, statements = load_test_with_sql_trace(
        self.database, "A", self.frozen_train_contract, self.context
    )
    test_selects = [sql.lower() for sql in statements if "split = 'test'" in sql.lower()]
    self.assertTrue(test_selects)
    for sql in test_selects:
        select_clause = sql.split("from", 1)[0]
        self.assertNotIn("label", select_clause)
        self.assertNotIn("labels_blob", select_clause)
```

Also assert read-only URI, TRAIN query requires `split='TRAIN' AND label='0'`,
raw byte length multiple of four, no padding/truncation/resampling, sorted
instance identity, and no label artifact path constant.

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_inputs -v
```

- [ ] **Step 4: Implement streaming SHA and duplicate-key JSON parsing**

Verify every artifact before parse, parse one JSONL row at a time, validate
manifest-declared SHA/row flags, rebuild all four lane maps, compare Exp291 and
Exp292 exactly, rehash after parse, and return a sorted immutable snapshot.

- [ ] **Step 5: Implement raw SQLite loading**

Use `sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)` and decode
`values_blob` with `np.frombuffer(blob, dtype="<f4")`. Keep TRAIN and TEST
loaders in separate functions and types. The TEST loader validates
`isinstance(frozen_train_contract, FrozenTrainContract)` before resolving or
opening the database. TEST rows select only instance ID/index and values. Do
not reuse `exp403_408.foundation.load_dataset_input` because it aligns and
per-series normalizes values contrary to Exp417.

- [ ] **Step 6: Add static label-isolation guard in the test module**

```python
def assert_package_has_no_label_path(package_root: Path) -> None:
    forbidden = ("frozen_test_labels", "labels_blob", "load_test_labels")
    for path in package_root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        if any(token in text for token in forbidden):
            raise ValueError(f"Exp417 label isolation violation: {path.name}")
```

Keep this helper in `test_inputs.py`, not the production package, so its own
forbidden literals do not violate the scan. The CLI preflight imports a
production `assert_label_free_sql_contract()` that parses fixed query constants
and verifies TEST selected columns are exactly instance identity plus values.
`label` remains allowed only in the normal TRAIN SQL predicate; tests inspect
selected columns separately.

- [ ] **Step 7: Run GREEN and regression**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_inputs -v
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow
git commit -m "feat: freeze Exp417 values-only inputs"
```

---

### Task 3: Exact-Length Preprocessing, Horizons, and Identity Folds

**Files:**
- Create: `experiments/exp417_causal_multihorizon_shadow/preprocessing.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `RawSeries`, `RawTrainDataset`, Exp414 `repair_train_blob`.
- Produces:
  - `Fold`
  - `build_horizons(length: int) -> tuple[int, int, int]`
  - `canonical_waveform_sha256(series: RawSeries) -> str`
  - `identity_grouped_folds(series: Sequence[RawSeries], *, folds: int = 5) -> tuple[Fold, ...]`
  - `preprocess_length_groups(dataset: RawTrainDataset, *, config=FROZEN_CONFIG) -> tuple[ProcessedGroup, ...]`

- [ ] **Step 1: Write exact horizon and abstention tests**

```python
def test_horizons_use_frozen_relative_offsets(self) -> None:
    self.assertEqual(build_horizons(100), (1, 2, 10))
    self.assertEqual(build_horizons(64), (1, 2, 7))

def test_duplicate_horizons_abstain(self) -> None:
    with self.assertRaisesRegex(ValueError, "distinct"):
        build_horizons(10)
```

Reject bool lengths and lengths that leave no valid long-horizon region.

- [ ] **Step 2: Write identity-fold isolation tests**

Assert byte-identical series share one validation fold, every series appears
in validation exactly once, folds are deterministic, and early-stop identity
subsets never cross their training subset.

- [ ] **Step 3: Write robust group-scaling tests**

Use two exact lengths and prove no resampling. Assert group-level TRAIN median
and MAD scaling preserves a shifted query series; per-series normalization
would make the test fail. Test non-finite TRAIN repair through the Exp414
function, all-missing abstention, residual non-finite rejection, and scale
`<=1e-12` abstention.

- [ ] **Step 4: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_preprocessing -v
```

- [ ] **Step 5: Implement horizons, identities, and folds**

Use `ceil` with exact integer validation. Assign identities to five folds by
sorted SHA256 then greedily place each group into the currently smallest fold
with fold index as tie-breaker. Early-stop validation uses identity hash modulo
10 and requires at least five identities on both sides.

- [ ] **Step 6: Implement preprocessing**

Group by exact tuple length, repair normal TRAIN blobs without length changes,
calculate raw and first-difference group medians/MAD scales from TRAIN only,
store immutable float32 two-channel arrays, and record diagnostic/candidate
eligibility from the 50/100 boundaries.

- [ ] **Step 7: Run GREEN**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_preprocessing -v
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow
git commit -m "feat: preprocess Exp417 exact-length groups"
```

---

### Task 4: Causal CNN and Deterministic OOF Training

**Files:**
- Create: `experiments/exp417_causal_multihorizon_shadow/model.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_model.py`

**Interfaces:**
- Consumes: `ProcessedGroup`, identity folds, frozen architecture/training config.
- Produces:
  - `class CausalMultiHorizonCNN(torch.nn.Module)`
  - `class TrainedSeedModel`
  - `class OOFModelBundle`
  - `set_deterministic_seed(seed: int, device: torch.device) -> None`
  - `train_oof_models(group: ProcessedGroup, *, device: torch.device, config=FROZEN_CONFIG) -> OOFModelBundle`
  - `fit_final_seed_models(group: ProcessedGroup, *, device: torch.device, config=FROZEN_CONFIG) -> tuple[TrainedSeedModel, ...]`
  - `predict_residuals(model: TrainedSeedModel, channels: np.ndarray) -> Mapping[int, np.ndarray]`

- [ ] **Step 1: Write architecture and future-leakage tests**

```python
def test_future_mutation_cannot_change_earlier_prediction(self) -> None:
    model = frozen_test_model()
    original = torch.zeros(1, 2, 128)
    changed = original.clone()
    changed[:, :, 96:] = 1000.0
    before = model(original)
    after = model(changed)
    for horizon in before:
        torch.testing.assert_close(before[horizon][..., :80], after[horizon][..., :80])
```

Also assert exactly 3 blocks, 32 channels, kernel 5, dilations 1/2/4, three
heads, and output alignment excludes unavailable targets.

- [ ] **Step 2: Write deterministic training tests**

On a small synthetic normal waveform group, two CPU fits with the same seed
must produce identical canonical model hashes formed from sorted parameter
names, dtype, shape, and contiguous tensor bytes; losses, early-stop epoch,
and residual arrays must also match. Do not compare raw `torch.save` bytes,
which may contain container metadata. Assert identical identities never cross
OOF fit and validation.

- [ ] **Step 3: Write failure tests**

Reject wrong channel count, non-finite tensors/losses/parameters, unexpected
device fallback, validation identities fewer than five, and an early-stopping
epoch beyond 100.

- [ ] **Step 4: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_model -v
```

- [ ] **Step 5: Implement causal model**

Use left-only padding inside a `CausalConv1d` wrapper:

```python
class CausalConv1d(nn.Conv1d):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        left = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(torch.nn.functional.pad(values, (left, 0)))
```

Each block is causal convolution, GELU, and residual projection when needed.
Heads map 32 channels to the raw-value target for each horizon.

- [ ] **Step 6: Implement deterministic OOF/final training**

Set Python/NumPy/Torch seeds, deterministic algorithms, fixed single-process
DataLoader order, AdamW/Huber parameters from config, TRAIN-only identity
early stopping, finite checks each epoch, and CPU state serialization.

- [ ] **Step 7: Run GREEN**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_model -v
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow
git commit -m "feat: train Exp417 causal OOF models"
```

---

### Task 5: Multi-Horizon Scoring, Persistence, and Seed Consensus

**Files:**
- Create: `experiments/exp417_causal_multihorizon_shadow/scoring.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_scoring.py`

**Interfaces:**
- Consumes: OOF/test residual mappings and frozen score config.
- Produces:
  - `longest_true_run_fraction(mask: np.ndarray) -> float`
  - `empirical_p_value(oof_scores: Sequence[float], query: float) -> float`
  - `score_horizon(oof_residuals: Sequence[np.ndarray], query_residual: np.ndarray) -> HorizonScore`
  - `seed_decision(scores: Mapping[int, HorizonScore], *, config=FROZEN_CONFIG) -> SeedDecision`
  - `consensus_candidate(decisions: Sequence[SeedDecision]) -> bool`

- [ ] **Step 1: Write hand-calculated score tests**

```python
def test_empirical_p_value_has_plus_one_correction(self) -> None:
    self.assertEqual(empirical_p_value([1.0, 2.0, 3.0], 2.5), 0.5)

def test_persistence_is_longest_consecutive_fraction(self) -> None:
    mask = np.array([0, 1, 1, 0, 1], dtype=bool)
    self.assertEqual(longest_true_run_fraction(mask), 2 / 5)
```

Use a fixed residual fixture to hand-calculate OOF median/MAD, OOF 99th
percentile, query Q95, persistence, `S=Q+2R`, and empirical p-value.

- [ ] **Step 2: Write two-of-three and all-seed tests**

Assert two horizons with `p<=0.01` but persistence below 0.05 fail, two valid
horizons pass one seed, and one disagreeing seed makes final consensus false.

- [ ] **Step 3: Write type/finiteness tests**

Reject bool counts, empty OOF, wrong horizon keys, duplicate seeds, non-finite
residuals, zero robust scale, p-values outside `[0,1]`, and mismatched residual
length.

- [ ] **Step 4: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_scoring -v
```

- [ ] **Step 5: Implement pure NumPy reference scoring**

Use deterministic NumPy percentile settings (`method="linear"`), exact
series-level empirical p-values, and immutable score records. Do not accept
labels or lane objects in any scoring signature.

- [ ] **Step 6: Run GREEN**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_scoring -v
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow
git commit -m "feat: score Exp417 persistent horizon errors"
```

---

### Task 6: Fixed Synthetic Safety and Workload Qualification

**Files:**
- Create: `experiments/exp417_causal_multihorizon_shadow/qualification.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_qualification.py`

**Interfaces:**
- Consumes: normal OOF series/scores, frozen lanes, per-seed TEST decisions.
- Produces:
  - `build_fixed_perturbations(series: np.ndarray, robust_scale: float, *, seed: int) -> Mapping[str, np.ndarray]`
  - `wilson_upper(successes: int, trials: int, confidence: float = 0.95) -> float`
  - `qualify_group(...) -> GroupQualification`
  - `freeze_train_contract(dataset: str, groups: Sequence[GroupQualification], backend: BackendAudit, *, config=FROZEN_CONFIG) -> FrozenTrainContract`
  - `exclude_frozen_lanes(...) -> tuple[ShadowCandidate, ...]`
  - `apply_dataset_workload(...) -> DatasetResult`
  - `apply_global_workload(results: Sequence[DatasetResult], total_test: int) -> RunSummary`

- [ ] **Step 1: Write exact perturbation tests**

For a length-100 centered ramp, assert onset 40, duration 60, level shift
`+3*scale`, drift endpoint `+3*scale`, amplitude `*1.5`, period warp `1.25`,
secondary spike length 1 and `+5*scale`, and deterministic seeds.

- [ ] **Step 2: Write qualification boundary tests**

Test exact pass/fail boundaries:

- OOF rate 0.01 passes and next representable larger rate fails;
- Wilson upper 0.05 passes and larger fails;
- pooled 0.70 and per-type 0.50 pass exactly;
- one seed disagreement fails;
- any non-finite or fallback fails.

- [ ] **Step 3: Write lane/workload tests**

Assert all four frozen lane types are excluded without mutation. A dataset at
exactly 5% passes; above 5% abstains with zero retained candidates. Global
exactly 1% passes; above 1% sets `publication_allowed=False` and retains no
official payload. Prove no sorting/top-N branch exists by permuting candidate
scores and obtaining the same all-or-none result.

- [ ] **Step 4: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_qualification -v
```

- [ ] **Step 5: Implement perturbations and gates**

Use TRAIN robust scale only. Period warp uses deterministic linear
interpolation inside the valid region and preserves output length; this is a
TRAIN synthetic perturbation, not TEST preprocessing. Implement Wilson upper
with `statistics.NormalDist().inv_cdf(0.975)`.

- [ ] **Step 6: Run GREEN**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_qualification -v
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow
git commit -m "feat: qualify Exp417 Shadow safety gates"
```

---

### Task 7: NumPy/Numba Parity and CPU/MPS Benchmark

**Files:**
- Create: `experiments/exp417_causal_multihorizon_shadow/backend.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_backend.py`

**Interfaces:**
- Consumes: reference scoring functions, trained synthetic model/group fixtures.
- Produces:
  - `persistence_numpy(mask: np.ndarray) -> float`
  - `persistence_numba(mask: np.ndarray) -> float`
  - `benchmark_scoring_backend(...) -> BackendAudit`
  - `benchmark_torch_devices(...) -> BackendAudit`
  - `select_canonical_backend(audits: Sequence[BackendAudit]) -> str`

- [ ] **Step 1: Write NumPy/Numba parity tests**

Cover empty, all-false, all-true, alternating, random seeded, and non-contiguous
boolean arrays. Require exact persistence identity and candidate identity.

- [ ] **Step 2: Write mocked CPU/MPS gate tests**

Inject benchmark results to test every boundary independently: exact candidate
identity, error `1e-4`, Spearman `0.999`, speedup `1.1`, finite outputs,
fallback false. One failed field must select CPU and record the exact reason.

- [ ] **Step 3: Write actual environment probe test**

On this host assert PyTorch reports MPS built but unavailable and the audit
returns `canonical_backend="cpu"` with `fallback_reason="mps_unavailable"`.
If availability changes, run the frozen smoke rather than hard-failing the
test; the test then validates the real gate result.

- [ ] **Step 4: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_backend -v
```

- [ ] **Step 5: Implement benchmark logic**

Include transfers and `torch.mps.synchronize()` in MPS timing. Benchmark one
device-heavy process at a time. Select up to nine TRAIN-only strata by
normal-count tertile × length tertile with lexicographic tie-breaking.

- [ ] **Step 6: Run GREEN**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_backend -v
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow
git commit -m "feat: gate Exp417 accelerated backends"
```

---

### Task 8: Provenance-Bound Atomic Checkpoints

**Files:**
- Create: `experiments/exp417_causal_multihorizon_shadow/checkpoints.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_checkpoints.py`

**Interfaces:**
- Consumes: frozen input identities, config, runtime versions, `DatasetResult`, `BackendAudit`.
- Produces:
  - `Exp417CheckpointProvenance`
  - `Exp417CheckpointStore(root: Path, provenance: Exp417CheckpointProvenance, expected_datasets: Sequence[str])`
  - methods `write_dataset`, `load_completed`, `progress`, `require_complete`.

- [ ] **Step 1: Write provenance mutation tests**

Create a complete synthetic checkpoint then mutate every provenance field
independently: commit, DB, each lane artifact, config, preprocessing, folds,
model, seeds, perturbations, package versions, backend. Reopen must fail before
loading a dataset record.

- [ ] **Step 2: Write record and atomicity tests**

Reject duplicate dataset writes, out-of-universe names, wrong schema/types,
non-finite metrics, candidate/lane overlap, hash mismatch, symlink root/file,
truncation, extra files, and missing dataset accounting. Inject write/fsync
failure and prove no completed record appears.

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints -v
```

- [ ] **Step 4: Implement checkpoint store**

Create root files `checkpoint_provenance.json` and `progress.json`, plus one
exclusive `datasets/<sha256(dataset)>.json` record per dataset. Root creation
uses a new directory, mode `0o700`, canonical JSON, file fsync, and directory
fsync. Each dataset record contains schema version, dataset name, raw/repaired
TRAIN identity, length-group identities, canonical model hashes, score
calibration hashes, qualification, candidate coordinates, backend audit,
elapsed time, and a SHA256 over the payload without its digest field. Loading
opens files with `O_NOFOLLOW`, rejects any unrecognized inventory entry, and
reconstructs all Exp417 contracts before returning a record.

- [ ] **Step 5: Run GREEN**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints -v
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
git diff --check
```

- [ ] **Step 6: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow
git commit -m "feat: checkpoint Exp417 causal Shadow runs"
```

---

### Task 9: Dataset and Population Runner

**Files:**
- Create: `experiments/exp417_causal_multihorizon_shadow/runner.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_runner.py`

**Interfaces:**
- Consumes: all Task 2-8 public interfaces.
- Produces:
  - `run_dataset(dataset: str, context: FrozenRunContext) -> DatasetResult`
  - `resume_or_run_population(context: FrozenRunContext, checkpoints: Exp417CheckpointStore) -> RunSummary`
  - `build_prediction_rows(summary: RunSummary) -> tuple[Mapping[str, object], ...]`

- [ ] **Step 1: Write orchestration-order test**

Patch boundary functions and assert:

```python
self.assertEqual(events, [
    "verify_inputs", "load_train_values", "preprocess", "train_oof",
    "qualify_train", "freeze_contract", "load_test_values", "score_test",
    "exclude_lanes", "dataset_workload", "checkpoint",
])
```

Prediction failure before `freeze_contract` must prove TEST values were not
loaded. No event may contain label access.

- [ ] **Step 2: Write dataset state tests**

Cover `<50`, `50..99`, eligible/pass, failed synthetic, failed OOF, failed
backend, excessive dataset workload, no candidate, multiple length groups,
all-missing repair, resume, and unexpected exception. Every path returns or
checkpoints exactly one explicit status.

- [ ] **Step 3: Write population conservation tests**

Require sorted exact 1,117 accounting in production config, no duplicate
coordinates, full lane exclusion, global all-or-none workload gate, and
prediction rows with exact false governance flags and no metric/label keys.

- [ ] **Step 4: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_runner -v
```

- [ ] **Step 5: Implement runner**

Keep label-free TRAIN qualification separate from TEST scoring. Use sequential
dataset execution as the canonical deterministic path. CPU may use at most
seven dataset workers only after a separate benchmark proves deterministic
identity and speedup; the initial implementation remains sequential.

- [ ] **Step 6: Run GREEN**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_runner -v
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow
git commit -m "feat: orchestrate Exp417 label-free prediction"
```

---

### Task 10: Deterministic Reporting and Atomic Eight-File Publication

**Files:**
- Create: `experiments/exp417_causal_multihorizon_shadow/reporting.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/publication.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_reporting.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_publication.py`

**Interfaces:**
- Consumes: complete `RunSummary`, frozen context, checkpoint provenance.
- Produces:
  - `render_exp417_payloads(...) -> Mapping[str, bytes]`
  - `verify_exp417_payloads(payloads: Mapping[str, bytes], *, config=FROZEN_CONFIG) -> None`
  - `publish_exp417(payloads: Mapping[str, bytes], output_dir: Path, *, pre_publish_verify: Callable[[], None]) -> Mapping[str, str]`

- [ ] **Step 1: Write exact rendering tests**

Assert exact seven non-manifest payload names, canonical JSON, sorted JSONL,
fixed CSV headers/order, empty fields for undefined diagnostics, no
label/evaluation keys, exact counts, frozen-lane exclusion, and Korean report
governance. A failed global workload summary must make rendering official
prediction payloads impossible.

- [ ] **Step 2: Write independent tamper tests**

Mutate candidate score/p-value/persistence, seed consensus, dataset status,
qualification rate, workload totals, backend audit, config hash, source hash,
governance flags, CSV order, and report count. Revalidation must reconstruct
contracts and fail.

- [ ] **Step 3: Write atomic publication attack tests**

Exercise all of these exact publication invariants:

- descriptor-pinned directory verification;
- `O_NOFOLLOW` file opens and fstat identity;
- exact eight files after manifest;
- mutation/symlink/root substitution/output race;
- atomic no-replace;
- post-rename content/inode revalidation;
- provenance bound to verified frozen config and report;
- parent fsync rollback and explicit cleanup failure.

- [ ] **Step 4: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_reporting \
  experiments.exp417_causal_multihorizon_shadow.tests.test_publication -v
```

- [ ] **Step 5: Implement payload rendering**

The exact official files are:

```python
OFFICIAL_FILES = {
    "exp417_shadow_predictions.jsonl",
    "exp417_dataset_qualification.csv",
    "exp417_oof_normal_metrics.csv",
    "exp417_synthetic_safety.csv",
    "exp417_backend_benchmark.json",
    "exp417_frozen_config.json",
    "EXP417_RESULT_REPORT_KO.md",
    "MANIFEST.json",
}
```

Use the verified summary as the single source for counts and governance.
Serialize JSON with sorted keys, compact separators, and `allow_nan=False`.

- [ ] **Step 6: Implement fail-closed publication**

Implement this exact sequence:

1. reject an existing output or missing parent;
2. create a unique sibling staging directory with mode `0o700`;
3. open and pin its directory descriptor using `O_DIRECTORY|O_NOFOLLOW`;
4. write seven payloads with exclusive dir-fd-relative opens, flush, and fsync;
5. render `MANIFEST.json` from the verified config, context, and payload hashes;
6. verify exact eight-entry inventory and every file through one
   `O_NOFOLLOW` descriptor with pre-open stat/fstat/post-read identity checks;
7. invoke `pre_publish_verify` and repeat semantic/byte/hash/inode checks;
8. fsync the staging directory;
9. atomically rename with macOS `renamex_np(RENAME_EXCL)` or Linux
   `renameat2(RENAME_NOREPLACE)`; fail closed on unsupported platforms;
10. open the output directory, require the captured staging inode, and repeat
    exact inventory/content/inode/semantic verification;
11. fsync the parent;
12. on failure, atomically quarantine only the owned inode, preserve foreign
    replacements, surface cleanup errors with the original error, and leave no
    owned final output.

- [ ] **Step 7: Run GREEN and regressions**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_reporting \
  experiments.exp417_causal_multihorizon_shadow.tests.test_publication -v
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp414_missing_aware_near_repeat/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp416_exp404_hard_test_confusion_audit/tests -p "test_*.py"
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow
git commit -m "feat: publish Exp417 Shadow artifacts atomically"
```

---

### Task 11: Official CLI, Non-Official Smoke, Runbook, and Readiness Gate

**Files:**
- Create: `scratch/run_exp417_causal_multihorizon_shadow.py`
- Create: `scratch/smoke_exp417_causal_multihorizon_shadow.py`
- Create: `experiments/exp417_causal_multihorizon_shadow/tests/test_cli.py`
- Create: `docs/EXP417_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: complete Exp417 package.
- Produces:
  - official `main(argv: list[str] | None = None) -> int`
  - smoke `main(argv: list[str] | None = None) -> int`
  - `class SmokeContext`
  - clean-commit readiness evidence only; no full run.

- [ ] **Step 1: Write strict official CLI tests**

The official CLI accepts exactly:

```text
--database PATH
--exp291-prediction PATH
--exp291-manifest PATH
--exp292-prediction PATH
--exp292-manifest PATH
--checkpoint-dir NEW_OR_MATCHING_PATH
--output-dir NEW_PATH
```

Use `ArgumentParser(allow_abbrev=False)`. Reject dataset subsets, maximum
datasets, family, alpha, horizon, threshold, seed, budget, backend override,
label path, existing output, dirty Git, and missing parent.

- [ ] **Step 2: Write separate smoke CLI tests**

The smoke CLI accepts `--smoke-datasets 3` but always sets:

- `official_prediction=false`;
- `publication_eligible=false`;
- `contains_test_labels=false`;
- a temporary or explicitly named smoke output;
- no reuse as official checkpoint.

Dataset selection uses TRAIN-only normal-count/length strata and lexicographic
tie-breaking. It cannot accept explicit dataset names.

- [ ] **Step 3: Write call-order/failure tests**

Assert official order: clean Git → package label guard → frozen hashes/lanes →
checkpoint → runner → snapshot revalidation → render/verify → publish.
Failure at any stage prevents later calls and leaves no output.

- [ ] **Step 4: Run RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_cli -v
```

- [ ] **Step 5: Implement both CLIs**

Bootstrap repository root for direct execution. Official CLI has no subset or
tuning mode. Smoke uses a separate `SmokeContext` type that cannot be passed
to official publication or checkpoint constructors.

- [ ] **Step 6: Write Korean runbook**

Document:

- exact frozen SHA commands;
- current environment and MPS unavailability;
- full unit/regression commands;
- label-free smoke command;
- checkpoint inspection and resume rules;
- exact official full-run command under a heading
  `사용자의 별도 명시적 승인 후에만 실행`;
- no-monitoring note: do not create a token-consuming poller;
- local-only artifact and GitHub export prohibition;
- Exp418 separation;
- no output overwrite/rerun rule.

- [ ] **Step 7: Run full synthetic verification before commit**

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp414_missing_aware_near_repeat/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp416_exp404_hard_test_confusion_audit/tests -p "test_*.py"
.venv/bin/python -m compileall -q \
  experiments/exp417_causal_multihorizon_shadow \
  scratch/run_exp417_causal_multihorizon_shadow.py \
  scratch/smoke_exp417_causal_multihorizon_shadow.py
git diff --check
```

- [ ] **Step 8: Commit CLI and runbook before any real values smoke**

```bash
git add scratch/run_exp417_causal_multihorizon_shadow.py \
  scratch/smoke_exp417_causal_multihorizon_shadow.py \
  experiments/exp417_causal_multihorizon_shadow/tests/test_cli.py \
  docs/EXP417_RUNBOOK_KO.md
git commit -m "feat: prepare Exp417 label-free readiness"
```

- [ ] **Step 9: Verify clean commit and real frozen non-label inputs**

```bash
git status --short --branch
git rev-parse HEAD
shasum -a 256 \
  "/Users/mac/Documents/Dataset/univariate_ts.db" \
  "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl" \
  "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json" \
  "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl" \
  "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json"
```

Do not hash, stat, or open any frozen TEST-label artifact.

- [ ] **Step 10: Run one non-official label-free readiness smoke**

Run the exact smoke command from the runbook on three deterministically selected
TRAIN strata. Verify it cannot publish an official prediction, cannot reuse an
official checkpoint, records CPU canonical because MPS is unavailable, and
does not change any frozen input.

- [ ] **Step 11: Review and stop before the full run**

Run task review and broad branch review. Report:

- test counts and regression status;
- smoke dataset selection derived only from TRAIN metadata;
- eligibility/abstention status without TEST metrics;
- backend audit;
- estimated full-run time and disk use from the smoke;
- exact next full-run command.

Stop and request the explicit instruction:

`Exp417 1,117개 전체 label-free 실행 승인`

Do not launch the full run as part of this plan.
