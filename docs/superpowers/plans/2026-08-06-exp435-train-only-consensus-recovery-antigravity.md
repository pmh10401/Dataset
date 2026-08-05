# Exp435 Train-Only Consensus Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a deterministic review-only detector that requires representative-normal distance plus gradient or ACF agreement, without using TEST labels before prediction freeze.

**Architecture:** A new Exp435 namespace package reuses the verified Exp425 database snapshot and Exp417 frozen-lane boundary, but owns its feature, scoring, MLX decision, checkpoint, publication, and retrospective-evaluation contracts. Prediction and evaluation are separate command paths; only the evaluation module may load TEST labels, and only after canonical prediction and manifest hashes are verified.

**Tech Stack:** Python 3, NumPy, SQLite read-only snapshots, optional MLX/Metal behind parity gates, `concurrent.futures.ProcessPoolExecutor`, canonical JSON/JSONL SHA256 artifacts, `unittest`.

## Global Constraints

- Every newly created repository file name must contain `antigravity`.
- Official population is the exact frozen Exp291/292 set of 1,117 datasets; the two DB-only extras are excluded.
- Prediction may use normal TRAIN, label-free TEST values, fixed metadata, and frozen lanes only.
- TEST labels and anomaly positions remain inaccessible until prediction and manifest SHA256 values are frozen.
- Candidate rule is `prototype AND (gradient OR ACF)` outside all four frozen lanes.
- Fixed scoring values are `k = 3`, `alpha = 0.01`, `q = 0.99`, `quantile method='higher'`, float32, and strict `score > threshold`.
- Global new-candidate count is at most 100; 101 fails closed without top-N trimming.
- Task 0 frozen-value summary: candidate budget `100개`, rescued-dataset minimum `10개`, added-false-positive maximum `30개`, paired bootstrap `20,000` replicates with seed `43520260806`.
- Results remain `review_only=true` and `operational_alert_eligible=false`.
- MLX is adopted only with finite outputs, score tolerance `1e-5`, exact candidate identity, and at least 1.1x stage speedup including transfer and synchronization.
- CPU fallback uses at most seven dataset workers and one inner numerical thread per worker.
- Success requires at least 10 rescued datasets, no more than 30 added false positives, and positive micro and macro F1 deltas whose paired 95% bootstrap lower bounds are both above zero.
- Before changing an existing module, read `docs/codemap/codemap.json` and record its callers, downstream effects, and tests. If stale, regenerate all three code-map files first.
- Use TDD for every behavior change and commit each independently reviewable task.

---

## Planned File Structure

### New prediction package

- `experiments/exp435_train_only_consensus_recovery_antigravity/config_antigravity.py`: immutable scientific and runtime constants.
- `experiments/exp435_train_only_consensus_recovery_antigravity/contracts_antigravity.py`: validated immutable records shared across tasks.
- `experiments/exp435_train_only_consensus_recovery_antigravity/inputs_antigravity.py`: official population and label-free TRAIN/TEST-value boundary.
- `experiments/exp435_train_only_consensus_recovery_antigravity/features_antigravity.py`: length grouping, TRAIN-only normalization, gradients, ACF, and LOO calibration.
- `experiments/exp435_train_only_consensus_recovery_antigravity/scoring_antigravity.py`: CPU scoring, consensus, lane exclusion, and global budget.
- `experiments/exp435_train_only_consensus_recovery_antigravity/mlx_antigravity.py`: optional dense-distance implementation and adoption benchmark.
- `experiments/exp435_train_only_consensus_recovery_antigravity/checkpoint_antigravity.py`: canonical run context and dataset-record persistence.
- `experiments/exp435_train_only_consensus_recovery_antigravity/progress_antigravity.py`: atomic progress, event, text log, and PID lifecycle.
- `experiments/exp435_train_only_consensus_recovery_antigravity/runtime_antigravity.py`: deterministic dataset scheduling and resume.
- `experiments/exp435_train_only_consensus_recovery_antigravity/reporting_antigravity.py`: independent publication replay and SHA256 map.
- `experiments/exp435_train_only_consensus_recovery_antigravity/evaluation_antigravity.py`: post-freeze TEST-label loading, paired metrics, bootstrap, and success decision.
- `experiments/exp435_train_only_consensus_recovery_antigravity/cli_antigravity.py`: fixed-path `readiness/new/resume/inspect/evaluate` boundary.
- `scratch/run_exp435_train_only_consensus_recovery_antigravity.py`: minimal executable wrapper.

### New tests

Every production file above has one matching file under
`experiments/exp435_train_only_consensus_recovery_antigravity/tests/`, named
`test_<component>_antigravity.py`. Do not add `__init__.py`; the repository already uses
namespace packages, and a new `__init__.py` would violate the filename rule.

### Existing governance files modified near completion

- `RESEARCH_STATE.json`
- `EXPERIMENT_CATALOG.json`
- `FAILURE_LEDGER.json` or `CHAMPION_LEDGER.json`, selected only by the frozen evaluation result
- `docs/codemap/codemap.html`
- `docs/codemap/codemap.json`
- `docs/codemap/codemap.lock`

---

### Task 0: Commit the approved planning baseline

**Files:**
- Create: `docs/superpowers/specs/2026-08-06-exp435-train-only-consensus-recovery-antigravity-design.md`
- Create: `docs/superpowers/plans/2026-08-06-exp435-train-only-consensus-recovery-antigravity.md`
- Create: `docs/EXP435_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP435_RUNBOOK_antigravity_KO.md`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `CHAMPION_LEDGER.json`
- Modify together: `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, `docs/codemap/codemap.lock`

**Interfaces:**
- Produces: reviewed Exp435 science constants, exact implementation tasks, fixed artifact paths, and a resumable planning state.
- Consumes: approved design decisions and the completed Exp426 attempt-2 evidence.

- [ ] **Step 1: Validate planning JSON, code-map identity, and whitespace**

```bash
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap.json >/dev/null
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 2: Confirm the frozen design values appear in all three documents**

```bash
rg -n "k = 3|alpha.*0.01|100개|10개|30개|20,000|43520260806" \
  docs/EXP435_PREREGISTRATION_antigravity_KO.md \
  docs/EXP435_RUNBOOK_antigravity_KO.md \
  docs/superpowers/plans/2026-08-06-exp435-train-only-consensus-recovery-antigravity.md
```

Expected: each fixed value has at least one matching line in the preregistration and plan; the
runbook contains the operator-facing budget, rescue/false-positive, and bootstrap values.

- [ ] **Step 3: Commit the approved baseline**

```bash
git add CHAMPION_LEDGER.json EXPERIMENT_CATALOG.json RESEARCH_STATE.json \
  docs/EXP435_PREREGISTRATION_antigravity_KO.md \
  docs/EXP435_RUNBOOK_antigravity_KO.md \
  docs/superpowers/plans/2026-08-06-exp435-train-only-consensus-recovery-antigravity.md \
  docs/superpowers/specs/2026-08-06-exp435-train-only-consensus-recovery-antigravity-design.md \
  docs/codemap
git commit -m "docs(exp435): approve consensus recovery research plan"
```

---

### Task 1: Freeze configuration and shared contracts

**Files:**
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/config_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/contracts_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_config_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_contracts_antigravity.py`

**Interfaces:**
- Produces: `Exp435FrozenConfig.validate() -> None`, `FROZEN_CONFIG`, `PreparedTrainGroup`, `ChannelCalibration`, `CalibratedTrainGroup`, `InstanceScoreEvidence`, `Exp435Candidate`, `Exp435DatasetResult`, `BackendDecision`, and `Exp435RunSummary`.
- Consumes: Exp417 `FrozenLaneSnapshot`, `RawSeries`, `RawTrainDataset`, and `RawTestDataset` only as typed boundary values.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_frozen_config_has_preregistered_values(self):
    self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp435")
    self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
    self.assertEqual(FROZEN_CONFIG.neighbor_count, 3)
    self.assertEqual(FROZEN_CONFIG.candidate_alpha, 0.01)
    self.assertEqual(FROZEN_CONFIG.train_quantile, 0.99)
    self.assertEqual(FROZEN_CONFIG.global_candidate_budget, 100)
    self.assertEqual(FROZEN_CONFIG.cpu_workers, 7)
    FROZEN_CONFIG.validate()

def test_mutated_config_fails_closed(self):
    with self.assertRaisesRegex(ValueError, "Exp435 frozen config mismatch"):
        replace(FROZEN_CONFIG, neighbor_count=2).validate()
```

- [ ] **Step 2: Run the configuration test and verify import failure**

Run:

```bash
python3 -m unittest experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_config_antigravity -v
```

Expected: FAIL because `config_antigravity` does not exist.

- [ ] **Step 3: Implement the immutable configuration**

```python
@dataclass(frozen=True)
class Exp435FrozenConfig:
    experiment_id: str = "Exp435"
    expected_datasets: int = 1117
    neighbor_count: int = 3
    candidate_alpha: float = 0.01
    train_quantile: float = 0.99
    quantile_method: str = "higher"
    min_train_rows: int = 4
    min_series_length: int = 2
    min_acf_length: int = 16
    max_acf_lag: int = 32
    numeric_floor: float = 1e-12
    score_tolerance: float = 1e-5
    global_candidate_budget: int = 100
    cpu_workers: int = 7
    mlx_min_speedup: float = 1.1
    bootstrap_repetitions: int = 20_000
    bootstrap_seed: int = 43_520_260_806
    rescued_dataset_minimum: int = 10
    added_false_positive_maximum: int = 30
```

Add all DB/lane/reference hashes from `docs/EXP435_PREREGISTRATION_antigravity_KO.md` and implement `validate()` by comparing every dataclass field with `type(self)()` using exact type and value equality.

- [ ] **Step 4: Write failing contract-invariant tests**

```python
def test_candidate_requires_prototype_and_one_secondary_channel(self):
    evidence = InstanceScoreEvidence.fixture(
        prototype_abnormal=True, gradient_abnormal=False, acf_abnormal=True
    )
    self.assertTrue(evidence.consensus_candidate)
    with self.assertRaisesRegex(ValueError, "consensus"):
        replace(evidence, consensus_candidate=False)

def test_dataset_result_rejects_duplicate_coordinates(self):
    candidate = Exp435Candidate("Demo", 3, 64, evidence_digest="a" * 64)
    with self.assertRaisesRegex(ValueError, "duplicate"):
        Exp435DatasetResult.fixture(candidates=(candidate, candidate))
```

- [ ] **Step 5: Implement focused immutable contracts**

Use frozen dataclasses. `InstanceScoreEvidence.__post_init__` recomputes
`prototype_abnormal and (gradient_abnormal or acf_abnormal)` and rejects inconsistent stored
decisions. `Exp435DatasetResult` validates sorted unique instance indices, dataset equality,
TEST bounds, finite scores, and no candidate on an abstained/failed result. Store NumPy arrays
only in `PreparedTrainGroup`/`CalibratedTrainGroup`; published records contain tuples and scalar
values only.

- [ ] **Step 6: Run both tests**

```bash
python3 -m unittest \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_config_antigravity \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_contracts_antigravity -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add experiments/exp435_train_only_consensus_recovery_antigravity
git commit -m "feat(exp435): freeze consensus recovery contracts"
```

---

### Task 2: Build the official label-free input boundary

**Files:**
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/inputs_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_inputs_antigravity.py`

**Interfaces:**
- Consumes: `open_database_snapshot(...)` from `experiments.exp425_selective_recovery.database`, `freeze_lane_inputs(...)` from Exp417, and `FROZEN_CONFIG`.
- Produces: `open_exp435_input_session(...) -> Iterator[Exp435InputSession]`, `Exp435InputSession.context`, `load_train(dataset) -> RawTrainDataset`, and `load_test_values(dataset, train_contract) -> RawTestDataset`.

- [ ] **Step 1: Record code-map impact before reusing existing modules**

Read `docs/codemap/codemap.json` nodes `exp425_database`, `exp417_inputs`, and
`sqlite_store`. In the task report record:

```text
Callers: Exp426 runtime/publication and Exp425 source/runtime boundaries.
Downstream: frozen population, TRAIN/TEST-value loaders, database integrity lifecycle.
Tests: Exp425 test_database.py, Exp417 test_inputs.py, Exp426 runtime/CLI tests.
```

Do not modify those existing modules.

- [ ] **Step 2: Write failing population and label-firewall tests**

```python
def test_session_uses_frozen_1117_population_not_all_db_names(self):
    with self.fixture_session(database_names=("A", "B", "DBOnlyExtra"), frozen=("A", "B")) as session:
        self.assertEqual(session.context.datasets, ("A", "B"))

def test_prediction_sql_never_selects_test_labels(self):
    source = Path(inputs_antigravity.__file__).read_text()
    self.assertNotIn("labels_blob", source)
    self.assertNotRegex(source, r"TEST.+label")
```

- [ ] **Step 3: Run the test and verify failure**

```bash
python3 -m unittest experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_inputs_antigravity -v
```

Expected: FAIL because the input session is absent.

- [ ] **Step 4: Implement the descriptor-bound session**

```python
@dataclass
class Exp435InputSession:
    database: Exp425DatabaseSnapshot
    context: FrozenRunContext
    label_access_count: int = 0

    def load_train(self, dataset: str) -> RawTrainDataset:
        return self.database.load_train(dataset, self.context)

    def load_test_values(
        self, dataset: str, contract: FrozenTrainContract
    ) -> RawTestDataset:
        return self.database.load_test(dataset, contract, self.context)
```

`open_exp435_input_session` hash-gates Exp291/292, opens one Exp425 snapshot, freezes the
Exp417 context at the reviewed commit, requires exactly 1,117 sorted names, and closes the
snapshot on every exit path. It must not import `evaluation_antigravity`.

- [ ] **Step 5: Add tampering and lifecycle tests**

Test DB hash mismatch, lane population mismatch, population count 1,116/1,118, extra DB-only
names staying excluded, close-on-exception, and label-access count remaining zero after TRAIN
and TEST-value reads.

- [ ] **Step 6: Run new and inherited input tests**

```bash
python3 -m unittest experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_inputs_antigravity -v
python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -v
python3 -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_inputs -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add experiments/exp435_train_only_consensus_recovery_antigravity
git commit -m "feat(exp435): enforce label-free input boundary"
```

---

### Task 3: Implement TRAIN-only features and CPU calibration

**Files:**
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/features_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_features_antigravity.py`

**Interfaces:**
- Consumes: `RawTrainDataset`, `RawSeries`, and `FROZEN_CONFIG`.
- Produces: `prepare_train_groups(train) -> tuple[PreparedTrainGroup, ...]`, `fit_calibrated_group(group) -> CalibratedTrainGroup`, `acf_vector(values, max_lag, floor) -> NDArray[np.float32] | None`, and `score_channel_cpu(query, references, k) -> float`.

- [ ] **Step 1: Write exact numerical tests first**

```python
def test_repeated_normal_rows_remain_as_neighbors(self):
    group = prepared_group([[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [2, 3, 2, 3]])
    fitted = fit_calibrated_group(group)
    self.assertEqual(fitted.prototype.reference_count, 4)
    self.assertEqual(fitted.prototype.loo_scores[0], 0.0)

def test_higher_quantile_and_strict_comparison(self):
    calibration = channel_calibration((0.0, 0.1, 0.2, 0.3))
    self.assertEqual(calibration.threshold, 0.3)
    self.assertFalse(calibration.is_abnormal(0.3))
    self.assertTrue(calibration.is_abnormal(np.nextafter(np.float32(0.3), np.float32(1.0))))
```

- [ ] **Step 2: Run the feature test and verify failure**

```bash
python3 -m unittest experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_features_antigravity -v
```

Expected: FAIL because feature functions do not exist.

- [ ] **Step 3: Implement length grouping and shared robust scaling**

Group rows by exact `len(values)`. Reject non-finite TRAIN groups. For amplitude and gradient,
compute:

```python
center = np.float32(np.median(matrix))
scale = np.float32(1.4826) * np.median(np.abs(matrix - center))
standardized = np.asarray((matrix - center) / scale, dtype=np.float32)
```

If row count is below 4, length below 2, scale non-finite, or scale at most `1e-12`, return a
validated unavailable channel with the exact preregistered reason.

- [ ] **Step 4: Implement ACF and k-neighbor RMS scoring**

```python
def score_channel_cpu(query, references, k):
    delta = references - query[None, :]
    distances = np.sqrt(np.mean(delta * delta, axis=1, dtype=np.float32))
    nearest = np.partition(distances, k - 1)[:k]
    return float(np.median(nearest))
```

ACF subtracts the row mean, computes lagged dot products divided by the zero-lag energy, and
returns lags `1..min(32, length//4)`. Return `None` for length below 16 or denominator at most
`1e-12`.

- [ ] **Step 5: Implement LOO calibration**

For every TRAIN row, call the same scorer with that row removed and all duplicate-valued other
rows retained. Require four finite LOO scores and compute:

```python
threshold = float(np.quantile(scores, 0.99, method="higher"))
```

Store the ordered scores, threshold, transform parameters, reference matrix SHA256, and reason
for unavailable optional channels.

- [ ] **Step 6: Add edge-case tests**

Cover row counts 3/4, length 1/2/15/16, zero MAD, non-finite TRAIN, unavailable gradient with
available ACF, unavailable ACF with available gradient, and row-order-independent canonical
hashes.

- [ ] **Step 7: Run feature and contract tests**

```bash
python3 -m unittest \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_features_antigravity \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_contracts_antigravity -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add experiments/exp435_train_only_consensus_recovery_antigravity
git commit -m "feat(exp435): add train-only consensus features"
```

---

### Task 4: Implement CPU dataset scoring and fail-closed candidate budget

**Files:**
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/scoring_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_scoring_antigravity.py`

**Interfaces:**
- Consumes: calibrated groups, label-free `RawTestDataset`, `FrozenLaneSnapshot`, and CPU channel scorer.
- Produces: `score_test_instance_cpu(...) -> InstanceScoreEvidence`, `score_dataset_cpu(...) -> Exp435DatasetResult`, and `validate_global_candidates(results, budget=100) -> tuple[Exp435Candidate, ...]`.

- [ ] **Step 1: Write failing consensus and boundary tests**

```python
def test_prototype_is_mandatory(self):
    evidence = score_fixture(prototype=False, gradient=True, acf=True)
    self.assertFalse(evidence.consensus_candidate)

def test_one_secondary_channel_is_enough(self):
    self.assertTrue(score_fixture(prototype=True, gradient=True, acf=False).consensus_candidate)
    self.assertTrue(score_fixture(prototype=True, gradient=False, acf=True).consensus_candidate)

def test_frozen_lane_coordinate_is_never_published(self):
    result = self.score(candidate_index=7, hard={"Demo": (7,)})
    self.assertEqual(result.candidates, ())
    self.assertEqual(result.evidence[0].exclusion_reason, "frozen_lane")
```

- [ ] **Step 2: Run the scoring test and verify failure**

```bash
python3 -m unittest experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_scoring_antigravity -v
```

Expected: FAIL because scoring functions are absent.

- [ ] **Step 3: Implement exact-length routing and evidence retention**

For each TEST row, select only the calibrated group with equal length. Missing length,
unavailable prototype, or unavailable gradient+ACF produces explicit abstention evidence. Score
available channels, compare with strict `>`, compute consensus, then call
`lanes.contains(dataset, instance_index)` before producing a candidate.

- [ ] **Step 4: Implement the global budget gate**

```python
def validate_global_candidates(results, budget=100):
    candidates = tuple(sorted(
        (candidate for result in results for candidate in result.candidates),
        key=lambda item: (item.dataset, item.instance_index),
    ))
    if len(candidates) > budget:
        raise CandidateBudgetExceeded(observed=len(candidates), budget=budget)
    if len({(c.dataset, c.instance_index) for c in candidates}) != len(candidates):
        raise ValueError("Exp435 candidates contain a duplicate coordinate")
    return candidates
```

Do not provide a sorting-by-score or slicing function.

- [ ] **Step 5: Add 100/101 and instance-boundary tests**

Verify exactly 100 candidates succeeds, 101 raises before publication, no top-100 artifact is
returned, mixed lengths use separate references, and concatenating two TEST rows produces a
different shape that the API rejects.

- [ ] **Step 6: Run scoring tests**

```bash
python3 -m unittest experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_scoring_antigravity -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add experiments/exp435_train_only_consensus_recovery_antigravity
git commit -m "feat(exp435): enforce conservative candidate consensus"
```

---

### Task 5: Add the MLX parity and speed adoption gate

**Files:**
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/mlx_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_mlx_antigravity.py`

**Interfaces:**
- Consumes: float32 query/reference batches and CPU scorer.
- Produces: `score_channel_mlx(...) -> NDArray[np.float32]`, `benchmark_mlx_adoption(fixtures, clock=perf_counter) -> BackendDecision`, and `score_with_backend(decision, ...) -> float`.

- [ ] **Step 1: Write gate tests with a fake MLX engine**

```python
def test_adopts_only_exact_candidate_identity_and_speed(self):
    decision = benchmark_mlx_adoption(
        fixtures=self.fixtures,
        engine=FakeMlxEngine(error=1e-6, candidate_flip=False, speedup=1.2),
    )
    self.assertTrue(decision.adopted)
    self.assertEqual(decision.backend, "mlx")

def test_candidate_flip_forces_cpu_even_when_fast(self):
    decision = benchmark_mlx_adoption(
        fixtures=self.fixtures,
        engine=FakeMlxEngine(error=0.0, candidate_flip=True, speedup=10.0),
    )
    self.assertFalse(decision.adopted)
    self.assertEqual(decision.reason, "candidate_identity_mismatch")
```

- [ ] **Step 2: Run the MLX test and verify failure**

```bash
python3 -m unittest experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_mlx_antigravity -v
```

Expected: FAIL because the MLX gate does not exist.

- [ ] **Step 3: Implement optional MLX import and dense distances**

Import `mlx.core` inside the MLX engine constructor only. Convert the exact float32 batches,
compute RMS distances and k-neighbor median without changing feature definitions, call
`mx.eval`, synchronize, and return NumPy float32 values. Absence of MLX/Metal returns a CPU
decision rather than raising during normal readiness.

- [ ] **Step 4: Implement preregistered benchmarking**

Measure batch sizes `(16, 32, 64, 128, 256)` with two warm-ups and five timed runs. Include
transfer and synchronization. Require finite scores, max absolute and normalized errors at most
`1e-5`, exact candidate coordinates, and speedup at least 1.1. Choose the fastest passing batch
by median time; stable tie-breaker is smaller batch size.

- [ ] **Step 5: Add fallback tests**

Cover unavailable MLX, non-finite output, error `1.0001e-5`, speedup 1.099, all batches failing,
and deterministic fastest-batch tie-breaking. Assert versions/device/timings/parity/reason appear
in `BackendDecision.to_dict()`.

- [ ] **Step 6: Run MLX and scoring tests**

```bash
python3 -m unittest \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_mlx_antigravity \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_scoring_antigravity -v
```

Expected: PASS on systems with or without MLX.

- [ ] **Step 7: Commit Task 5**

```bash
git add experiments/exp435_train_only_consensus_recovery_antigravity
git commit -m "feat(exp435): gate MLX scoring by parity and speed"
```

---

### Task 6: Add checkpoint, progress, and deterministic runtime

**Files:**
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/checkpoint_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/progress_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/runtime_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_checkpoint_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_progress_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_runtime_antigravity.py`

**Interfaces:**
- Consumes: `Exp435DatasetResult`, backend decision, verified run context, and label-free input session.
- Produces: `Exp435CheckpointStore`, `ProgressWriter`, `run_exp435_new(...) -> Exp435RunSummary`, `resume_exp435(...) -> Exp435RunSummary`, and `inspect_exp435(...) -> Exp435RunSummary`.

- [ ] **Step 1: Record code-map answers for reused persistence modules**

Read nodes `artifact_utils`, `exp426_checkpoint`, and `exp426_runtime`. Record their callers,
downstream publication effects, and matching Exp426 tests. Reuse canonical byte helpers, not
Exp426 schemas.

- [ ] **Step 2: Write checkpoint crash-window tests**

```python
def test_record_commit_survives_missing_event_and_progress(self):
    store.commit(self.result)
    reopened = Exp435CheckpointStore.open_existing(self.root, self.context)
    recovered = reopened.recover_runtime_accounting()
    self.assertEqual(recovered.completed, 1)
    self.assertEqual(recovered.missing_events, ("Demo",))

def test_context_mismatch_refuses_resume(self):
    with self.assertRaisesRegex(ValueError, "context fingerprint"):
        Exp435CheckpointStore.open_existing(self.root, altered_context)
```

- [ ] **Step 3: Implement no-clobber canonical checkpoint records**

Create `run_context_antigravity.json`, one canonical record per dataset, a record-set digest,
and inspect/recovery methods. Use atomic temp-file replacement and fsync from the existing
artifact helper. Validate dataset order, context hash, record hash, candidate bounds, and no
unknown files before resume.

- [ ] **Step 4: Write progress/PID lifecycle tests**

Assert atomic JSON contains PID, heartbeat, current dataset and metadata, stage/backend,
completed/total, success/abstention/error, candidate count, throughput, elapsed, ETA,
checkpoint, and latest error. Test normal completion, user interruption, exception failure, and
inactive PID cleanup.

- [ ] **Step 5: Implement progress and logs**

`ProgressWriter` writes `progress_antigravity.json` atomically, appends and fsyncs
`events_antigravity.jsonl`, appends `parent_antigravity.log`, and owns
`run_antigravity.pid`. Heartbeat interval is 30 seconds. Never store TEST values or labels.

- [ ] **Step 6: Write serial/parallel identity tests**

Run a fixed multi-dataset fixture with one worker and seven workers. Require identical ordered
dataset-record SHA256 and prediction coordinates. Simulate out-of-order completion and verify
commit order remains dataset-name order.

- [ ] **Step 7: Implement the runtime scheduler**

CPU uses `ProcessPoolExecutor(max_workers=7)` and a bounded prefetch queue of 14. Set numerical
thread environment variables to `1` before worker imports. Parent alone owns DB snapshot,
checkpoint, logs, and ordered commits. MLX backend uses one process and no outer scoring pool.
On candidate count 101, persist the exact failure event and stop before publication.

- [ ] **Step 8: Run runtime tests and Exp426 recovery regression**

```bash
python3 -m unittest discover \
  -s experiments/exp435_train_only_consensus_recovery_antigravity/tests \
  -p 'test_*antigravity.py'
python3 -m unittest \
  experiments.exp426_verified_baseline.tests.test_checkpoint_antigravity \
  experiments.exp426_verified_baseline.tests.test_runtime_antigravity -v
```

Expected: PASS.

- [ ] **Step 9: Commit Task 6**

```bash
git add experiments/exp435_train_only_consensus_recovery_antigravity
git commit -m "feat(exp435): add resumable deterministic runtime"
```

---

### Task 7: Publish frozen predictions and wire the fixed CLI

**Files:**
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/reporting_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/cli_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_reporting_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_cli_antigravity.py`
- Create: `scratch/run_exp435_train_only_consensus_recovery_antigravity.py`

**Interfaces:**
- Consumes: completed checkpoint, terminal runtime logs, fixed paths, and backend decision.
- Produces: `publish_exp435(...) -> FrozenPublication`, `inspect_publication(...) -> FrozenPublication`, `run_readiness()`, `run_new()`, `run_resume()`, `run_inspect()`, and CLI `main(argv) -> int`.

- [ ] **Step 1: Write publication replay tests**

```python
def test_publication_replays_checkpoint_without_labels(self):
    publication = publish_exp435(self.completed_store, self.output)
    replay = inspect_publication(self.output, self.completed_store)
    self.assertEqual(publication.prediction_sha256, replay.prediction_sha256)
    self.assertFalse(publication.manifest["test_labels_opened"])
    self.assertTrue(publication.manifest["review_only"])

def test_budget_failure_never_creates_publication(self):
    with self.assertRaises(CandidateBudgetExceeded):
        publish_exp435(self.store_with_101_candidates, self.output)
    self.assertFalse((self.output / "publication_antigravity").exists())
```

- [ ] **Step 2: Run reporting tests and verify failure**

```bash
python3 -m unittest experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_reporting_antigravity -v
```

Expected: FAIL because publication code is absent.

- [ ] **Step 3: Implement atomic independent publication**

Create a staging directory, replay all records, validate 1,117 conservation and candidate
budget, then publish:

```text
prediction_antigravity.jsonl
dataset_summary_antigravity.jsonl
score_evidence_antigravity.jsonl
abstention_ledger_antigravity.jsonl
backend_decision_antigravity.json
manifest_antigravity.json
status_antigravity.json
artifact_sha256_antigravity.json
```

Manifest records `review_only=true`, `operational_alert_eligible=false`,
`contains_test_labels=false`, `test_labels_opened=false`, config/input/code hashes, backend
evidence, and scientific file hashes. Exclude the SHA map's self-hash and state that exclusion.

- [ ] **Step 4: Write fixed-path CLI and readiness tests**

Test only five modes are accepted, alternate output paths are rejected, readiness binds clean
HEAD and absent roots, `new` rechecks the same HEAD, completed `inspect` opens neither DB nor
workers, and `evaluate` is delegated without importing its module in prediction modes.

- [ ] **Step 5: Implement CLI and wrapper**

The wrapper is only:

```python
from experiments.exp435_train_only_consensus_recovery_antigravity.cli_antigravity import main

if __name__ == "__main__":
    raise SystemExit(main())
```

CLI paths are constants copied from the preregistration. Return codes are 0 success, 1 valid
incomplete/interrupted state, and 2 invalid command/path/integrity state.

- [ ] **Step 6: Run reporting/CLI and inherited publication tests**

```bash
python3 -m unittest \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_reporting_antigravity \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_cli_antigravity -v
python3 -m unittest experiments.exp426_verified_baseline.tests.test_reporting_antigravity -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 7**

```bash
git add experiments/exp435_train_only_consensus_recovery_antigravity scratch/run_exp435_train_only_consensus_recovery_antigravity.py
git commit -m "feat(exp435): freeze review-only prediction artifacts"
```

---

### Task 8: Add hash-gated retrospective evaluation

**Files:**
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_evaluation_antigravity.py`

**Interfaces:**
- Consumes: independently verified frozen publication, Exp291 Hard coordinates, DB TEST labels, and `FROZEN_CONFIG`.
- Produces: `evaluate_frozen_prediction(...) -> Exp435Evaluation`, `paired_dataset_bootstrap(...) -> BootstrapResult`, and `decide_exp435_success(...) -> SuccessDecision`.

- [ ] **Step 1: Write the label-order guard test**

```python
def test_hash_failure_occurs_before_label_loader_call(self):
    calls = []
    with self.assertRaisesRegex(ValueError, "prediction SHA256"):
        evaluate_frozen_prediction(
            publication=self.tampered_publication,
            label_loader=lambda: calls.append("opened"),
        )
    self.assertEqual(calls, [])
```

- [ ] **Step 2: Write deterministic metric and bootstrap tests**

Use a three-dataset hand-computed fixture. Assert exact TP/FP/FN/TN, rescued names, strong
regression names, micro/macro deltas, 20,000 repeat count, seed `43520260806`, and identical
bootstrap bytes on two runs.

- [ ] **Step 3: Run evaluation tests and verify failure**

```bash
python3 -m unittest experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_evaluation_antigravity -v
```

Expected: FAIL because evaluation code is absent.

- [ ] **Step 4: Implement the only TEST-label loader**

Keep label SQL private in `evaluation_antigravity.py`. It loads TEST instance labels only after
`inspect_publication` validates prediction, manifest, artifact map, population, code/input
hashes, and complete-review-only status. No prediction module imports this function.

- [ ] **Step 5: Implement paired confusion metrics and bootstrap**

Compute Hard and Hard+Exp435 confusion per dataset and globally. Resample dataset indices with
`np.random.default_rng(43520260806)` for 20,000 paired replicates. Recompute aggregate micro F1
and mean per-dataset F1 in every replicate. Use percentile 2.5 and 97.5 bounds. Define F1 as
0.0 when its denominator is zero.

- [ ] **Step 6: Implement the all-gates decision**

```python
passed = (
    evaluation.dataset_count == 1117
    and evaluation.candidate_count <= 100
    and len(evaluation.rescued_datasets) >= 10
    and evaluation.added_fp <= 30
    and evaluation.micro_f1_delta > 0
    and evaluation.micro_f1_ci.lower > 0
    and evaluation.macro_f1_delta > 0
    and evaluation.macro_f1_ci.lower > 0
    and evaluation.prediction_label_access_count == 0
)
```

Write canonical evaluation JSON, per-dataset JSONL, rescued/regressed JSONL, confusion JSON,
bootstrap JSON, decision JSON, and SHA256 map under `evaluation_antigravity` without modifying
prediction files.

- [ ] **Step 7: Add failure-gate tests**

Vary one condition at a time: rescued 9, FP 31, micro lower 0, macro lower 0, candidate 101,
population 1,116, and prediction label-access count 1. Every case must fail with a stable reason.

- [ ] **Step 8: Run evaluation and full Exp435 unit tests**

```bash
python3 -m unittest discover \
  -s experiments/exp435_train_only_consensus_recovery_antigravity/tests \
  -p 'test_*antigravity.py'
```

Expected: PASS.

- [ ] **Step 9: Commit Task 8**

```bash
git add experiments/exp435_train_only_consensus_recovery_antigravity
git commit -m "feat(exp435): add post-freeze paired evaluation"
```

---

### Task 9: Synchronize governance, documentation, and code map

**Files:**
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `docs/EXP435_PREREGISTRATION_antigravity_KO.md`
- Modify: `docs/EXP435_RUNBOOK_antigravity_KO.md`
- Modify: `docs/codemap/codemap.html`
- Modify: `docs/codemap/codemap.json`
- Modify: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: final code interfaces and tests from Tasks 1-8.
- Produces: exact resume state, catalog lineage, runnable commands, and an interactive map with the Exp435 prediction/evaluation boundary.

- [ ] **Step 1: Update research state and catalog before execution**

Set `active_experiment` to `Exp435`, phase to `implementation_complete_readiness_pending`, PID
to null, and active document paths to the approved design, this plan, preregistration, and
runbook. Record Exp426 attempt 2 as complete `1117/983/134/0` and add Exp435 as
`implementation_complete_prediction_not_started` with `test_labels_opened=false`.

- [ ] **Step 2: Reconcile documentation with implemented CLI help**

Run:

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py --help
```

Compare all five commands, paths, state names, file names, constants, return codes, and resume
rules with both Exp435 documents. Change documentation only to correct implementation drift;
do not change frozen science constants.

- [ ] **Step 3: Regenerate all three code-map files together**

Add no more than the existing 20 primary nodes by grouping Exp435 files into prediction,
persistence/publication, and evaluation nodes. Show prediction reading TRAIN/TEST values but
not labels, and evaluation reading labels only after publication verification. Attach source
path and symbol evidence to every new edge and flow.

- [ ] **Step 4: Validate the code map**

Verify JSON parse, node paths/evidence symbols, edge and flow references, HTML embedded JSON
identity, current commit/dirty state, and deterministic top-level fingerprints. Every
relationship without evidence must explicitly be `unknown`; do not infer edges.

- [ ] **Step 5: Run all targeted and inherited tests**

```bash
python3 -m unittest discover \
  -s experiments/exp435_train_only_consensus_recovery_antigravity/tests \
  -p 'test_*antigravity.py'
python3 -m unittest discover \
  -s experiments/exp426_verified_baseline/tests \
  -p 'test_*antigravity.py'
python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py'
python3 -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_inputs \
  experiments.exp417_causal_multihorizon_shadow.tests.test_preprocessing \
  experiments.exp417_causal_multihorizon_shadow.tests.test_contracts -v
git diff --check
```

Expected: all PASS and no whitespace errors.

- [ ] **Step 6: Commit Task 9**

```bash
git add RESEARCH_STATE.json EXPERIMENT_CATALOG.json \
  docs/EXP435_PREREGISTRATION_antigravity_KO.md \
  docs/EXP435_RUNBOOK_antigravity_KO.md docs/codemap
git commit -m "docs(exp435): freeze execution governance and code map"
```

---

### Task 10: Independent review, readiness, full prediction, and evaluation

**Files:**
- Modify after result: `RESEARCH_STATE.json`
- Modify after result: `EXPERIMENT_CATALOG.json`
- Modify after failure only: `FAILURE_LEDGER.json`
- Modify after success only: `CHAMPION_LEDGER.json`
- External artifacts only: `/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806`

**Interfaces:**
- Consumes: clean reviewed implementation commit and fixed CLI.
- Produces: frozen label-free prediction, retrospective evaluation, and an immutable pass/fail lineage record.

- [ ] **Step 1: Request fresh read-only scientific and engineering review**

Reviewer checks label firewall, exact formulas, repeated-normal handling, no concatenation,
budget fail-close, MLX gates, recovery semantics, publication replay, evaluation ordering, and
all success criteria. Resolve findings with new tests and commits; rerun Task 9 regressions.

- [ ] **Step 2: Create run parent and run readiness**

```bash
mkdir /Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py readiness
```

Require `state=ready`, exact input hashes/population, label access 0, passing backend decision,
and absent output/checkpoint roots.

- [ ] **Step 3: Start the full label-free prediction**

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py new
```

Do not poll repeatedly. The process writes 30-second heartbeat, dataset metadata, throughput,
ETA, and checkpoint itself. If interrupted, use the exact `resume` command from the runbook.

- [ ] **Step 4: Inspect and freeze prediction hashes**

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py inspect
```

Require 1,117 completed, error 0, candidates at most 100, inactive PID, label access 0, and
replayed hashes. Record prediction and manifest SHA256 in `RESEARCH_STATE.json` and
`EXPERIMENT_CATALOG.json`, commit that freeze, and do not modify prediction files afterward.

- [ ] **Step 5: Run retrospective evaluation only after the freeze commit**

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py evaluate
```

Verify evaluation artifact SHA256 and rerun evaluation read-only to require identical bytes.

- [ ] **Step 6: Record the preregistered decision**

If every gate passes, append Exp435 to `CHAMPION_LEDGER.json` as a review-only candidate. If
any gate fails, append the exact results and lesson to `FAILURE_LEDGER.json`. In both cases,
update research state, catalog, exact artifact paths, prediction/evaluation hashes, and next
action. Never edit Exp435 constants based on its TEST result.

- [ ] **Step 7: Final verification and result commit**

```bash
python3 -m unittest discover \
  -s experiments/exp435_train_only_consensus_recovery_antigravity/tests \
  -p 'test_*antigravity.py'
git diff --check
git status --short --branch
```

Commit only governance/result metadata; the large artifacts remain under the fixed local
Dataset root and are referenced by SHA256.
