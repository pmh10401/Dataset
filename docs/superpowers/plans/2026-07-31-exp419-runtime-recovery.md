# Exp419 Runtime Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover Exp419 with safe local progress and deterministic seed-level CPU concurrency without changing Exp417 scientific outputs.

**Architecture:** Every CNN fit remains single-threaded. New Exp417 helpers prepare all seed initial states serially; Exp419 consumes those immutable states in three concurrent seed tasks and restores the existing canonical evidence order. A local atomic JSON status record is visibility-only and excluded from checkpoint, prediction, and publication evidence.

**Tech Stack:** Python 3.14, unittest, PyTorch CPU, NumPy, macOS spawn, canonical JSON.

## Global Constraints

- Do not change Exp417 preprocessing, CNN, horizons, loss, frozen seeds, early stop, qualification, thresholds, lanes, or publication.
- No TEST labels, anomaly locations, past TEST results, human reviews, raw waveforms, scores, predictions, or metrics enter status, scheduling, selection, or smoke timing.
- Each fit remains one PyTorch/BLAS thread. MLX and MPS remain disabled.
- Outer workers remain `(1, 2, 4, 6)`. Inner seed workers are exactly `{1: 1, 2: 3, 4: 3, 6: 2}`; their product is at most 12 CPU slots.
- Status and all data-derived artifacts are local only. No monitor/polling/cron/cloud/TencentDB data export.
- Fail closed unless serial and parallel paths have exactly equal model hashes, loss histories, residual bytes, and candidate indices.

---

### Task 1: Atomic worker status

**Files:**
- Create: `experiments/exp419_deterministic_cpu_parallel/worker_status.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/test_worker_status.py`

**Interfaces:**
- `WorkerStatus(shard_index, worker_pid, expected_datasets, completed_datasets, phase, dataset, dataset_index, seed, fold, error)`.
- `write_worker_status(root: Path, status: WorkerStatus) -> Path` writes `exp419_worker_status.json` with a temporary sibling then `os.replace`.

- [ ] **Step 1: Write the failing test**

```python
def test_status_writer_publishes_safe_dataset_progress(self):
    status = WorkerStatus(0, 123, 2, 0, "dataset_starting", "alpha", 0, None, None, "")
    payload = json.loads(write_worker_status(self.root, status).read_text())
    self.assertEqual(payload["dataset"], "alpha")
    self.assertEqual(payload["completed_datasets"], 0)

def test_status_rejects_prediction_payload(self):
    with self.assertRaisesRegex(ValueError, "status field"):
        WorkerStatus.from_payload({"phase": "oof", "prediction": [1]})
```

- [ ] **Step 2: Verify red**

Run `/Users/mac/Documents/Time Series/.venv/bin/python -m unittest experiments.exp419_deterministic_cpu_parallel.tests.test_worker_status -v`.

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the writer**

```python
def write_worker_status(root: Path, status: WorkerStatus) -> Path:
    target = Path(root) / "exp419_worker_status.json"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json_bytes(status.to_payload()) + b"\n")
    os.replace(temporary, target)
    return target
```

Permit only literal phases `worker_starting`, `context`, `dataset_starting`, `train_loading`, `preprocessing`, `oof`, `final_fit`, `qualification`, `test_scoring`, `checkpoint`, `worker_completed`, and `worker_failed`. Validate all fields and do not suppress I/O errors.

- [ ] **Step 4: Verify green and commit**

Run the focused test again; it must PASS. Commit `worker_status.py` and its test with message `feat(exp419): add atomic worker status`.

### Task 2: Prepared deterministic seed fitting

**Files:**
- Modify: `experiments/exp417_causal_multihorizon_shadow/model.py`
- Modify: `experiments/exp417_causal_multihorizon_shadow/tests/test_model.py`

**Interfaces:**
- `prepare_seed_initial_states(horizons, device) -> Mapping[int, FrozenState]`.
- `SeedOOFTraining(seed: int, models: tuple[TrainedSeedModel, ...], oof_residuals: Mapping[int, Mapping[int, np.ndarray]])` for one seed's five fits and validation residuals.
- `fit_seed_oof_from_state(group, seed, initial_state, device) -> SeedOOFTraining` for one seed.
- `fit_seed_final_from_state(group, seed, initial_state, device) -> TrainedSeedModel` for one seed.

- [ ] **Step 1: Write the failing byte-parity test**

```python
def test_prepared_seed_oof_matches_legacy_evidence(self):
    legacy = train_oof_models(self.group, device=CPU)
    states = prepare_seed_initial_states(build_horizons(self.group.group.length), CPU)
    actual = fit_seed_oof_from_state(self.group, 41701, states[41701], CPU)
    self.assertEqual(
        tuple(model.model_sha256 for model in legacy.models if model.seed == 41701),
        tuple(model.model_sha256 for model in actual.models),
    )
    for index, residuals in actual.oof_residuals.items():
        for horizon, residual in residuals.items():
            np.testing.assert_array_equal(residual, legacy.oof_residuals[41701][index][horizon])
```

- [ ] **Step 2: Verify red**

Run `/Users/mac/Documents/Time Series/.venv/bin/python -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_model.DeterministicTrainingTests.test_prepared_seed_oof_matches_legacy_evidence -v`.

Expected: FAIL because the prepared-state API is absent.

- [ ] **Step 3: Implement state preparation and fitting**

```python
def _initial_state(seed, horizons, device):
    set_deterministic_seed(seed, device)
    network = CausalMultiHorizonCNN(horizons).to(device)
    return _serialize_state(network.state_dict())

def prepare_seed_initial_states(horizons, device):
    return MappingProxyType({seed: _initial_state(seed, horizons, device) for seed in FROZEN_CONFIG.model_seeds})
```

Implement private stop-epoch/refit helpers by materializing a new model from `initial_state` before every fit. Preserve batch order, optimizer, loss, validation, error checks, and serialization. Keep the current public sequential APIs unchanged as the reference implementation.

- [ ] **Step 4: Verify green and commit**

Run `/Users/mac/Documents/Time Series/.venv/bin/python -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_model -v`; it must PASS. Commit the model and test files with message `refactor(exp417): expose prepared deterministic seed fits`.

### Task 3: Seed-parallel Exp419 training

**Files:**
- Create: `experiments/exp419_deterministic_cpu_parallel/seed_parallel.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/test_seed_parallel.py`
- Modify: `experiments/exp417_causal_multihorizon_shadow/runner.py`

**Interfaces:**
- `train_models_seed_parallel(processed, backend, seed_workers, on_stage=None) -> TrainModelEvidence`.
- `RunnerRuntime.production_with_train_models(resources, train_models) -> RunnerRuntime` differs from production only in the TRAIN-model callable.

- [ ] **Step 1: Write the failing evidence-equivalence test**

```python
def test_three_seed_tasks_reassemble_exact_serial_evidence(self):
    sequential = _production_train_models((synthetic_group(),), passing_cpu_audit())
    parallel = train_models_seed_parallel((synthetic_group(),), passing_cpu_audit(), 3)
    self.assertEqual(parallel.train_identity_sha256, sequential.train_identity_sha256)
    self.assertEqual(
        tuple(model.model_sha256 for model in parallel.oof_bundles[128].models),
        tuple(model.model_sha256 for model in sequential.oof_bundles[128].models),
    )
```

- [ ] **Step 2: Verify red**

Run `/Users/mac/Documents/Time Series/.venv/bin/python -m unittest experiments.exp419_deterministic_cpu_parallel.tests.test_seed_parallel -v`.

Expected: FAIL because seed-parallel training is absent.

- [ ] **Step 3: Implement bounded concurrent seed tasks**

```python
with ThreadPoolExecutor(max_workers=seed_workers) as pool:
    futures = {seed: pool.submit(_fit_one_seed, group, seed, initial_states[seed], device) for seed in FROZEN_CONFIG.model_seeds}
    by_seed = {seed: futures[seed].result() for seed in FROZEN_CONFIG.model_seeds}
```

Each task uses only prepared states and returns `SeedOOFTraining` plus its optional final model. Reassemble its five-model sequences into a complete `OOFModelBundle` in original fold-major/seed-major order, then reassemble final models in frozen seed order. On any task error cancel outstanding tasks and re-raise before returning `TrainModelEvidence`. Reuse all existing production TEST barrier/scoring/publication callables.

- [ ] **Step 4: Verify green and commit**

Run `/Users/mac/Documents/Time Series/.venv/bin/python -m unittest experiments.exp419_deterministic_cpu_parallel.tests.test_seed_parallel experiments.exp417_causal_multihorizon_shadow.tests.test_runner -v`; it must PASS. Commit with message `feat(exp419): parallelize independent seed fits`.

### Task 4: Layout, status wiring, bounded smoke, and verification

**Files:**
- Modify: `experiments/exp419_deterministic_cpu_parallel/config.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/contracts.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/workers.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_config.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_workers.py`
- Modify: `scratch/benchmark_exp419_deterministic_cpu_parallel.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_cli.py`
- Modify: `docs/EXP419_PREREGISTRATION_KO.md`, `docs/EXP419_RUNBOOK_KO.md`, and `experiments/exp419_deterministic_cpu_parallel/SOURCE_MANIFEST.json`

- [ ] **Step 1: Write failing layout/status/smoke tests**

```python
def test_four_outer_workers_have_three_seed_tasks_within_budget(self):
    self.assertEqual(seed_workers_for_outer_workers(4), 3)
    self.assertLessEqual(4 * seed_workers_for_outer_workers(4), 12)

def test_bounded_runtime_smoke_uses_two_train_batches_without_test_loader(self):
    measurement = run_bounded_train_runtime_smoke(self.frozen_inputs, 4)
    self.assertEqual(measurement.worker_count, 4)
    self.assertFalse(self.test_loader_called)
```

- [ ] **Step 2: Verify red**

Run the focused Exp419 config, worker, and CLI tests. They must FAIL because the layout/status integration and bounded smoke are absent.

- [ ] **Step 3: Implement frozen layout and bounded smoke**

```python
def seed_workers_for_outer_workers(worker_count):
    for outer, inner in FROZEN_CONFIG.seed_workers_by_outer_worker:
        if worker_count == outer:
            return inner
    raise ValueError("outer worker count is not frozen")
```

Extend `WorkerSpec` with the plan's outer worker count. Write safe statuses before context reconstruction, immediately before a dataset, before TRAIN load/preprocessing/OOF/final-fit/qualification/TEST-values-only scoring/checkpoint, after durable checkpoint evidence, and at terminal success/failure. The runtime smoke retains the existing 19 TRAIN-metadata selection but executes exactly two frozen TRAIN batches per length group; record batch limit, layout, peak RSS, source manifest digest, input proof digest, and no-TEST assertion. Update both Korean documents to retire the old four full-pipeline trials and document the status file. Regenerate the source manifest with the reviewed builder.

- [ ] **Step 4: Full verification and commit**

Run:

```bash
/Users/mac/Documents/Time Series/.venv/bin/python -m py_compile experiments/exp417_causal_multihorizon_shadow/*.py experiments/exp419_deterministic_cpu_parallel/*.py scratch/benchmark_exp419_deterministic_cpu_parallel.py scratch/run_exp419_deterministic_cpu_parallel.py
/Users/mac/Documents/Time Series/.venv/bin/python -m unittest discover -s experiments/exp417_causal_multihorizon_shadow/tests -p 'test_*.py'
/Users/mac/Documents/Time Series/.venv/bin/python -m unittest discover -s experiments/exp419_deterministic_cpu_parallel/tests -p 'test_*.py'
/Users/mac/Documents/Time Series/.venv/bin/python - <<'PY'
from scratch.benchmark_exp419_deterministic_cpu_parallel import assert_exp419_label_isolation
print(assert_exp419_label_isolation())
PY
```

All commands must PASS. Do not create a benchmark root or official output root. Commit the changed Exp419 runtime, tests, docs, and manifest with message `fix(exp419): recover deterministic runtime`.
