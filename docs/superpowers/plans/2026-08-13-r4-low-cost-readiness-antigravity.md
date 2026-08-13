# r4 Low-Cost Readiness Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a label-free, fail-closed command that verifies the exact 1,117-dataset nonfinite inventory and reproduces the frozen MiniRocket scores for only the 19 affected datasets before an official r4 run is allowed.

**Architecture:** Add one focused readiness module to the active Exp448 package that reuses Exp443's immutable source snapshot and database pinning authority, then imports and executes the current standalone paper repair/alignment/MiniRocket scorer for the 19 affected datasets. The module remains outside the paper namespace because that reviewed namespace forbids `antigravity` filenames while the repository requires them for every new file. It emits deterministic canonical JSON to stdout and progress to stderr; it never creates a run root or reads TEST labels. Unit tests inject tiny frozen snapshots and scorer outputs, while one opt-in bounded test exercises the real 19-dataset capable environment.

**Tech Stack:** Python 3.14.6, NumPy 2.3.5, aeon 1.5.0, scikit-learn 1.8.0, canonical JSON/SHA-256, `unittest`, SQLite read-only pinning.

## Global Constraints

- Grok CLI implements with `--model grok-4.6 --reasoning-effort high` in an isolated Git worktree.
- Codex performs the independent diff review, focused verification, and integration.
- Active experiment is Exp448. Inspect `docs/codemap/codemap_exp448_antigravity.json` before edits; update the complete Exp448 triplet when the new module boundary is added. The paper triplet stays byte-unchanged because paper files are read-only dependencies.
- Every new filename contains `antigravity`.
- Do not modify the frozen detector, preprocessing formula, thresholds, seeds, folds, route rules, database, r3 root, or any proposed r4 root.
- Do not read TEST labels or evaluation artifacts. TEST access is limited to value blobs or row identities already permitted by the frozen label-free boundaries.
- Do not create an official r4 root or launch the 1,117-dataset proposal run.
- Canonical readiness uses CPU. The values-only scan is sequential; official aeon MiniRocket remains `n_jobs=1`. MLX, MPS, and PyTorch are recorded as non-canonical because rewriting the frozen transformer changes the comparison implementation.
- The only preregistered MiniRocket candidate alpha is `0.005`; compare it for both kernels and both seeds.

---

## File structure

- Create `experiments/exp448_exp447_text_label_boundary_repair_antigravity/paper_r4_readiness_antigravity.py`: immutable result contracts, inventory verification, paper-versus-archive MiniRocket parity, canonical CLI, and stderr progress.
- Create `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_readiness_antigravity.py`: tiny fail-closed unit tests and one opt-in real 19-dataset capable gate.
- Leave `paper/equipment_anomaly_detection/codemap_paper.{html,json,lock}` unchanged because no paper module changes.
- Modify `docs/codemap/codemap_exp448_antigravity.{html,json,lock}` together: Exp448 repository-level callers, effects, tests, and result flow.

### Task 1: Implement the values-only inventory contract

**Files:**
- Create: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/paper_r4_readiness_antigravity.py`
- Create: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_readiness_antigravity.py`

**Interfaces:**
- Consumes: `verify_frozen_sources_antigravity() -> FrozenSourceSnapshotAntigravity`, `pin_database_snapshot_antigravity(...)`, `load_raw_dataset_antigravity(...)`, and `repair_linear_edge_nearest_paper(np.ndarray) -> np.ndarray`.
- Produces: `InventoryGateAntigravity`, `ReadinessFailureAntigravity`, and `verify_inventory_gate_antigravity(...) -> InventoryGateAntigravity`.

- [ ] **Step 1: Write failing immutable-contract tests**

Add tests that construct a minimal snapshot and assert exact accepted values, deep immutability, and rejection of booleans or altered counts:

```python
def test_inventory_gate_requires_exact_frozen_counts(self):
    valid = InventoryGateAntigravity(
        dataset_count=1117,
        affected_datasets=EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY,
        affected_rows=164,
        nonfinite_scalars=983,
        all_missing_rows=0,
        affected_datasets_sha256=EXPECTED_AFFECTED_SHA256,
        label_access_count=0,
        repaired_row_count=164,
    )
    self.assertEqual(valid.affected_rows, 164)
    with self.assertRaises(ValueError):
        replace(valid, all_missing_rows=1)
```

Use a fixed 19-name tuple in the fixture; do not make the test depend on the real database.

- [ ] **Step 2: Run the exact test and record RED**

Run:

```bash
python3 -m unittest \
  experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_readiness_antigravity.R4ReadinessContractTests -v
```

Expected: import failure because `paper_r4_readiness_antigravity.py` does not exist.

- [ ] **Step 3: Implement strict inventory types and validation**

Implement these constants and contracts:

```python
READINESS_SCHEMA_ANTIGRAVITY = "paper-r4-readiness-antigravity.v1"
EXPECTED_DATASETS_ANTIGRAVITY = 1117
EXPECTED_AFFECTED_ROWS_ANTIGRAVITY = 164
EXPECTED_NONFINITE_SCALARS_ANTIGRAVITY = 983
EXPECTED_ALL_MISSING_ANTIGRAVITY = 0
EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY = (
    "DodgerLoopDay_normal_1", "DodgerLoopDay_normal_2",
    "DodgerLoopDay_normal_4", "DodgerLoopDay_normal_5",
    "DodgerLoopDay_normal_6", "DodgerLoopDay_normal_7",
    "DodgerLoopGame_normal_1", "DodgerLoopGame_normal_2",
    "DodgerLoopWeekend_normal_1", "DodgerLoopWeekend_normal_2",
    "MelbournePedestrian_normal_1", "MelbournePedestrian_normal_2",
    "MelbournePedestrian_normal_3", "MelbournePedestrian_normal_4",
    "MelbournePedestrian_normal_5", "MelbournePedestrian_normal_6",
    "MelbournePedestrian_normal_7", "MelbournePedestrian_normal_8",
    "MelbournePedestrian_normal_9",
)
EXPECTED_AFFECTED_SHA256_ANTIGRAVITY = (
    "ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9"
)
FROZEN_ALPHAS_ANTIGRAVITY = (0.005,)

class ReadinessFailureAntigravity(RuntimeError):
    def __init__(self, failures: tuple[str, ...]): ...

@dataclass(frozen=True, slots=True)
class InventoryGateAntigravity:
    dataset_count: int
    affected_datasets: tuple[str, ...]
    affected_rows: int
    nonfinite_scalars: int
    all_missing_rows: int
    affected_datasets_sha256: str
    label_access_count: int
    repaired_row_count: int
```

`InventoryGateAntigravity.__post_init__` must require sorted unique 19 affected datasets, exact scalar counts, exact affected SHA, zero labels, and exactly 164 repaired affected rows. It must reject `bool` for integer fields.

- [ ] **Step 4: Implement inventory verification with dependency injection**

Use the existing Exp443 source verifier for the full database scan and source-hash audit. Then pin the database once in a temporary `attempt_antigravity` directory, load only the 19 affected datasets, call `repair_linear_edge_nearest_paper` only for TRAIN rows with nonfinite coordinates, and require finite little-endian float32 output preserving original length:

```python
def verify_inventory_gate_antigravity(
    *,
    verify_sources=verify_frozen_sources_antigravity,
    pin_database=pin_database_snapshot_antigravity,
    load_raw=load_raw_dataset_antigravity,
    repair=repair_linear_edge_nearest_paper,
    progress: Callable[[str], None] | None = None,
) -> tuple[InventoryGateAntigravity, object]:
    ...
```

Return the verified snapshot as the second value so the parity task cannot rescan or substitute a different authority. Before returning, require the snapshot values `(1117, 19, 164, 983, 0, label_access_count=0)` and the affected dataset SHA. Close the pin in `finally` with durable terminal evidence. Do not inspect a TEST label column.

- [ ] **Step 5: Add fail-before-score tests**

Inject snapshots with a changed count, an all-missing row, wrong SHA, or nonzero label access. Patch the parity scorer to raise if called and prove all inventory failures occur first. Inject one repair returning NaN and require `ReadinessFailureAntigravity` containing the dataset and TRAIN index.

- [ ] **Step 6: Run Task 1 tests and commit**

Run the contract and inventory test classes only. Expected: PASS in under five seconds with no real database access.

Commit:

```bash
git add experiments/exp448_exp447_text_label_boundary_repair_antigravity/paper_r4_readiness_antigravity.py \
  experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_readiness_antigravity.py
git commit -m "feat: add r4 values-only readiness inventory"
```

### Task 2: Add paper MiniRocket parity and deterministic CLI output

**Files:**
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/paper_r4_readiness_antigravity.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_readiness_antigravity.py`

**Interfaces:**
- Consumes: the exact snapshot returned by Task 1, `prepare_dataset_antigravity`, paper repair/alignment/z-normalization, `score_configuration_paper`, `upper_tail_pvalues`, and frozen archive arrays in `snapshot.cache_inventory`.
- Produces: `ParityEvidenceAntigravity`, `R4ReadinessResultAntigravity`, `run_r4_readiness_antigravity(...) -> R4ReadinessResultAntigravity`, `canonical_result_bytes_antigravity(...) -> bytes`, and `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing parity tests**

Use two tiny affected datasets, two kernels, and two seeds. Inject `ConfigurationScoresPaper` values and archive arrays. Assert 8 comparisons, maximum difference at most `1e-6`, finite outputs, and exact candidate equality at alpha `0.005`.

Add separate failures for:

```python
paper_score = archive_score + 1.000001e-6  # score tolerance failure
paper_candidates = frozenset({1})          # archive candidate is empty
```

Both must produce `status="not_ready"`; neither may create a filesystem path.

- [ ] **Step 2: Run parity tests and record RED**

Run only `R4ReadinessParityTests`. Expected: missing result/parity interfaces.

- [ ] **Step 3: Implement paper-path matrix and score comparison**

For each affected dataset in sorted snapshot order:

1. load one raw dataset through the already pinned label-free connection;
2. prepare Exp443 Policy A as the independent reference matrix;
3. repair every paper TRAIN and TEST row with `repair_linear_edge_nearest_paper`;
4. align to the frozen TRAIN-derived target length and z-normalize using paper helpers;
5. require paper matrices to be finite float32 and byte-identical to the Policy A prepared matrices;
6. call paper `score_configuration_paper` for `(1024, 4096) × (20260721, 20261721)`;
7. compare TRAIN OOF and TEST arrays to the matching immutable Exp249 cache arrays;
8. recompute empirical upper-tail candidates at `alpha=0.005` on both sides and require identical coordinates.

Collect 76 evidence records. Continue through all 19 datasets after a dataset-local parity failure so `not_ready` contains the complete bounded failure list. Do not continue if Gate A fails.

- [ ] **Step 4: Implement deterministic result serialization**

Define:

```python
@dataclass(frozen=True, slots=True)
class ParityEvidenceAntigravity:
    dataset: str
    requested_kernels: int
    seed: int
    train_shape: tuple[int, ...]
    test_shape: tuple[int, ...]
    max_abs_difference: float
    candidate_equal: bool

@dataclass(frozen=True, slots=True)
class R4ReadinessResultAntigravity:
    status: str  # ready | not_ready
    inventory: InventoryGateAntigravity | None
    parity: tuple[ParityEvidenceAntigravity, ...]
    failures: tuple[str, ...]
    database_sha256: str
    code_closure_sha256: str
    environment: Mapping[str, object]
```

The canonical payload must contain no timestamp, PID, duration, temporary path, or unordered mapping. Add a self-excluding `result_sha256` computed from canonical JSON bytes. Bind the code closure to SHA-256 values for the readiness module, paper preprocessing/MiniRocket/calibration, and the exact Exp443 config/database/sources/preprocessing/MiniRocket/control modules.

- [ ] **Step 5: Implement CLI and progress**

`python -m experiments.exp448_exp447_text_label_boundary_repair_antigravity.paper_r4_readiness_antigravity` must:

- write progress such as `inventory start`, `inventory ready`, and `parity 7/19 <dataset>` to stderr with flushing;
- write exactly one canonical JSON object to stdout;
- return exit code `0` only for `ready`, and `2` for `not_ready`;
- accept no database path, run root, output root, threshold, seed, worker, or model override.

- [ ] **Step 6: Add deterministic and boundary tests**

Run the same injected ready fixture twice and assert identical stdout bytes. Patch label-capable imports and filesystem writers to raise. Assert the CLI neither imports `label_access_paper` nor calls `mkdir`, `open` for write, `os.link`, or artifact writers.

- [ ] **Step 7: Run Task 2 tests and commit**

Run the new test module only. Expected: all non-capable tests pass in under ten seconds.

Commit:

```bash
git add experiments/exp448_exp447_text_label_boundary_repair_antigravity/paper_r4_readiness_antigravity.py \
  experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_readiness_antigravity.py
git commit -m "feat: verify bounded r4 MiniRocket readiness"
```

### Task 3: Synchronize maps and run the bounded capable gate

**Files:**
- Modify together: `docs/codemap/codemap_exp448_antigravity.html`
- Modify together: `docs/codemap/codemap_exp448_antigravity.json`
- Modify together: `docs/codemap/codemap_exp448_antigravity.lock`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_readiness_antigravity.py`

**Interfaces:**
- Consumes: the completed readiness command from Task 2 and the pinned capable interpreter.
- Produces: exact code-map coverage and an opt-in real gate named `test_real_19_dataset_r4_readiness_antigravity`.

- [ ] **Step 1: Add the opt-in real test**

Guard the test with `PAPER_R4_READINESS_REAL_ANTIGRAVITY=1`. It must call the production readiness function and require:

```python
self.assertEqual(result.status, "ready")
self.assertEqual(result.inventory.dataset_count, 1117)
self.assertEqual(len(result.inventory.affected_datasets), 19)
self.assertEqual(result.inventory.affected_rows, 164)
self.assertEqual(result.inventory.nonfinite_scalars, 983)
self.assertEqual(result.inventory.all_missing_rows, 0)
self.assertEqual(len(result.parity), 76)
self.assertLessEqual(max(item.max_abs_difference for item in result.parity), 1e-6)
self.assertTrue(all(item.candidate_equal for item in result.parity))
```

Also require zero label access and confirm that the immutable r3 path hash/tree and the proposed r4 absence are unchanged before and after the bounded run.

- [ ] **Step 2: Update the complete Exp448 code-map triplet**

The new module's map entry must state:

- callers: module CLI and `tests/test_paper_r4_readiness_antigravity.py`;
- effects: values-only 1,117 inventory gate, 19-dataset paper MiniRocket parity, canonical readiness output, no official run creation;
- tests: the new test module;
- source authorities: Exp443 config/database/sources/preprocessing/MiniRocket/control plus paper preprocessing/MiniRocket/calibration.

Update every changed/new module hash. Ensure embedded HTML JSON is semantically identical to the JSON file. Regenerate all three files in each triplet together, never one member alone.

- [ ] **Step 3: Run non-capable verification**

Run:

```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_readiness_antigravity -v
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper -v
git diff --check
```

Expected: all pass; the real test skips without the opt-in environment variable.

- [ ] **Step 4: Run one bounded capable readiness execution**

Run only after the non-capable suite passes:

```bash
PAPER_R4_READINESS_REAL_ANTIGRAVITY=1 \
  /Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python \
  -m unittest \
  experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_readiness_antigravity.R4ReadinessRealTests.test_real_19_dataset_r4_readiness_antigravity \
  -v
```

Expected: exactly one real bounded test passes, progress identifies all 19 datasets, readiness is `ready`, 76 comparisons pass, and no r4 root appears.

- [ ] **Step 5: Verify Grok scope and commit**

Confirm the Grok branch contains only the two new files and the two synchronized triplets. Confirm no database, r3, r4, public baseline, proposal runner, or detector source was modified.

Commit:

```bash
git add experiments/exp448_exp447_text_label_boundary_repair_antigravity/paper_r4_readiness_antigravity.py \
  experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_readiness_antigravity.py \
  docs/codemap/codemap_exp448_antigravity.html \
  docs/codemap/codemap_exp448_antigravity.json \
  docs/codemap/codemap_exp448_antigravity.lock
git commit -m "test: gate r4 with bounded readiness evidence"
```

## Self-review

- Spec coverage: full inventory, exact 19/164/983/0 authority, paper repair, 76 MiniRocket comparisons, candidate equality, deterministic output, no labels, no r4 creation, acceleration assessment, and Grok/Codex isolation each have an explicit task.
- Scope: one production module and one test module; no detector, runner, database, r3, r4, baseline, or evaluation change.
- Type consistency: Task 1 returns `(InventoryGateAntigravity, snapshot)`; Task 2 consumes that same snapshot and returns `R4ReadinessResultAntigravity`; Task 3 invokes only the Task 2 public readiness function.
- Ambiguity resolution: “every frozen alpha” means the sole preregistered MiniRocket alpha `0.005`; the 76 comparisons are `19 datasets × 2 kernels × 2 seeds`.
- Placeholder scan: no deferred implementation decisions or unspecified tests remain.
