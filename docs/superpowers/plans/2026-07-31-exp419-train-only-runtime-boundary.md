# Exp419 TRAIN-only Runtime Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Exp419's runtime benchmark truly TRAIN-only and remove repeated
worker context/database verification without changing Exp417 scientific output.

**Architecture:** Add a process-local verified read-only SQLite session and
connection-based loaders in Exp417 inputs. Exp419 passes a separately frozen
TRAIN-only benchmark context to smoke workers, while official workers receive
the parent-frozen full context. Both paths emit safe local runtime logs and
fail closed on provenance or database identity mismatch.

**Tech Stack:** Python 3.14, SQLite read-only descriptors, PyTorch CPU,
Numba-only audited score aggregation, `unittest`, macOS `spawn`.

## Global Constraints

- Do not change the model, preprocessing, horizons, seed values, loss, early
  stopping, OOF calibration, scoring, threshold, qualification, lanes, or
  publication schema.
- Benchmark workers may access only normal TRAIN metadata and raw TRAIN values.
- Official workers may access raw TEST values only after the full parent context
  is frozen; neither path may access TEST labels, anomaly locations, scores,
  predictions, evaluation metrics, or human-review data for decisions.
- Keep MLX and MPS excluded. Keep one PyTorch/BLAS thread per CNN fit and the
  frozen outer/inner layouts `1→1`, `2→3`, `4→3`, `6→2`.
- Verify the database descriptor SHA-256 before use and stable inode metadata
  after use. A mismatch is fail-closed.
- Runtime logs are local, append-only, safe-field-only, and observation-only;
  never create polling, monitors, cron jobs, or keepalives.
- Regenerate `SOURCE_MANIFEST.json` explicitly after reviewed runtime-source
  changes. Do not reuse any old Exp417/Exp419 checkpoint, benchmark, or output
  directory.

---

### Task 1: Add reusable verified database sessions

**Files:**
- Modify: `experiments/exp417_causal_multihorizon_shadow/inputs.py:226-275,495-580`
- Test: `experiments/exp417_causal_multihorizon_shadow/tests/test_inputs.py`

**Interfaces:**
- Produces `verified_read_only_connection(database: Path, expected_sha256: str)`;
  it yields one SQLite connection, verifies source bytes before opening, and
  verifies descriptor identity on close.
- Produces `load_raw_train_values_from_connection(connection, dataset, *,
  database_sha256, datasets)` and
  `load_raw_test_values_from_connection(connection, dataset, context)`.
- Existing `load_raw_train_values` and `load_raw_test_values` remain public and
  delegate to the new functions through one-use sessions, preserving Exp417
  behavior for existing callers.

- [ ] **Step 1: Write the failing session-reuse test**

```python
def test_verified_connection_loads_two_train_datasets_without_reopening_source():
    with verified_read_only_connection(self.database, self.database_sha) as conn:
        first = load_raw_train_values_from_connection(
            conn, "alpha", database_sha256=self.database_sha,
            datasets=("alpha", "beta"),
        )
        second = load_raw_train_values_from_connection(
            conn, "beta", database_sha256=self.database_sha,
            datasets=("alpha", "beta"),
        )
    self.assertEqual(first.dataset, "alpha")
    self.assertEqual(second.dataset, "beta")
```

The production change this catches is accidentally opening/re-hashing the
database once per dataset instead of once per worker.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_inputs.VerifiedConnectionTests.test_verified_connection_loads_two_train_datasets_without_reopening_source
```

Expected: FAIL because the session and connection loaders do not exist.

- [ ] **Step 3: Write the failing mutation test**

```python
def test_verified_connection_rejects_descriptor_change_on_close():
    with self.assertRaisesRegex(ValueError, "inode changed"):
        with verified_read_only_connection(self.database, self.database_sha) as conn:
            load_raw_train_values_from_connection(
                conn, "alpha", database_sha256=self.database_sha,
                datasets=("alpha",),
            )
            self.replace_database_bytes_with_same_schema()
```

The production change this catches is accepting a source inode replacement
after a query has begun.

- [ ] **Step 4: Implement the minimal shared session and loaders**

```python
@contextmanager
def verified_read_only_connection(database: Path, expected_sha256: str):
    descriptor = _open_verified_database_descriptor(database, expected_sha256)
    before = os.fstat(descriptor)
    connection = _open_read_only(descriptor)
    try:
        yield connection
        if _stable_inode_metadata(before) != _stable_inode_metadata(os.fstat(descriptor)):
            raise ValueError("Exp417 database inode changed during query")
    finally:
        connection.close()
        os.close(descriptor)
```

Factor the existing `_load_values` decoding path so the connection-based
loaders execute the unchanged frozen SQL and `_decode_raw_series`; do not copy
or alter query text.

- [ ] **Step 5: Run focused GREEN and the full Exp417 input suite**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_inputs
```

Expected: PASS, with existing label-isolation tests unchanged.

- [ ] **Step 6: Commit the isolated database session change**

```bash
git add experiments/exp417_causal_multihorizon_shadow/inputs.py \
        experiments/exp417_causal_multihorizon_shadow/tests/test_inputs.py
git commit -m "feat(exp419): reuse verified worker database sessions"
```

### Task 2: Separate benchmark and official context bootstraps

**Files:**
- Modify: `experiments/exp419_deterministic_cpu_parallel/contracts.py:345-430`
- Modify: `experiments/exp419_deterministic_cpu_parallel/workers.py:53-110,210-295,336-550,963-1048`
- Modify: `scratch/benchmark_exp419_deterministic_cpu_parallel.py:812-838`
- Modify: `scratch/run_exp419_deterministic_cpu_parallel.py`
- Test: `experiments/exp419_deterministic_cpu_parallel/tests/test_workers.py`
- Test: `experiments/exp419_deterministic_cpu_parallel/tests/test_cli.py`

**Interfaces:**
- Produces `BenchmarkTrainContext(database_sha256, datasets, lanes,
  code_commit, runtime_versions, config)` with canonical payload and SHA-256;
  it has no TEST-related field.
- Produces `BenchmarkTrainBootstrap(resources, context,
  expected_train_context_sha256)` for bounded smoke workers.
- Changes `WorkerBootstrap` to include the parent-frozen exact
  `FrozenRunContext` and its expected digest for official workers.

- [ ] **Step 1: Write the failing benchmark-boundary test**

```python
def test_bounded_smoke_uses_train_context_without_full_context_freezer(self):
    with patch.object(workers, "_freeze_context", side_effect=AssertionError("full context")):
        completion = workers.run_bounded_train_smoke_worker(
            self.train_only_spec, self.train_only_bootstrap,
        )
    self.assertEqual(completion.status, "completed")
```

The production change this catches is reintroducing the Exp417 full context
freezer, which queries TEST identity metadata, into the benchmark path.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_workers.BoundedSmokeTests.test_bounded_smoke_uses_train_context_without_full_context_freezer
```

Expected: FAIL because the current smoke worker invokes `_freeze_context`.

- [ ] **Step 3: Write the failing official-bootstrap test**

```python
def test_official_worker_accepts_parent_frozen_context_without_reconstruction(self):
    with patch.object(workers, "_freeze_context", side_effect=AssertionError("rebuild")):
        completion = workers.run_worker(self.official_spec, self.official_bootstrap)
    self.assertEqual(completion.status, "completed")
```

The production change this catches is rebuilding a complete population context
in every spawned official worker.

- [ ] **Step 4: Implement minimal immutable contexts and serialization**

```python
@dataclass(frozen=True)
class BenchmarkTrainBootstrap:
    resources: RunnerResources
    context: BenchmarkTrainContext
    expected_context_sha256: str

def freeze_benchmark_inputs(...):
    lanes = freeze_lane_inputs(...)
    datasets = list_dataset_names(args.database, lanes)
    context = BenchmarkTrainContext(...)
    return FrozenBenchmarkInputs(..., bootstrap=BenchmarkTrainBootstrap(...))
```

Official construction freezes `FrozenRunContext` once in the parent, validates
it against provenance, serializes its canonical payload, and workers validate
the passed object/digest rather than invoking `_freeze_context`.

- [ ] **Step 5: Use one verified connection per worker**

```python
with verified_read_only_connection(resources.database, context.database_sha256) as connection:
    for dataset in spec.shard.datasets:
        train = load_raw_train_values_from_connection(
            connection, dataset, database_sha256=context.database_sha256,
            datasets=context.datasets,
        )
        # bounded smoke: no TEST call
        # official worker: unchanged TEST loader after train contract freeze
```

Keep the existing dataset order, model-thread pinning, checkpoint sequence,
and fail-closed digest checks intact.

- [ ] **Step 6: Run focused GREEN tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_workers \
  experiments.exp419_deterministic_cpu_parallel.tests.test_cli
```

Expected: PASS; benchmark fixtures report no TEST access and official fixtures
preserve their existing canonical output digest.

- [ ] **Step 7: Commit the bootstrap-boundary change**

```bash
git add experiments/exp419_deterministic_cpu_parallel/contracts.py \
        experiments/exp419_deterministic_cpu_parallel/workers.py \
        scratch/benchmark_exp419_deterministic_cpu_parallel.py \
        scratch/run_exp419_deterministic_cpu_parallel.py \
        experiments/exp419_deterministic_cpu_parallel/tests/test_workers.py \
        experiments/exp419_deterministic_cpu_parallel/tests/test_cli.py
git commit -m "fix(exp419): isolate train-only worker contexts"
```

### Task 3: Add safe local progress logs

**Files:**
- Modify: `experiments/exp419_deterministic_cpu_parallel/contracts.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/workers.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/coordinator.py`
- Test: `experiments/exp419_deterministic_cpu_parallel/tests/test_worker_status.py`
- Test: `experiments/exp419_deterministic_cpu_parallel/tests/test_workers.py`

**Interfaces:**
- Produces `RuntimeLogRecord` with only `run_id`, `timestamp`, `pid`,
  `shard_index`, `current_dataset`, `phase`, optional `seed`, optional `fold`,
  `completed_datasets`, `elapsed_seconds`, `status`, and `error`.
- Produces one local append-only JSONL log per worker inside the existing
  checkpoint-sibling status root, named `worker-<shard>.runtime.jsonl`.

- [ ] **Step 1: Write the failing safe-log schema test**

```python
def test_runtime_log_rejects_test_prediction_label_and_metric_fields(self):
    with self.assertRaisesRegex(ValueError, "forbidden"):
        RuntimeLogRecord.from_mapping({
            "run_id": "r", "timestamp": "2026-07-31T00:00:00Z",
            "pid": 1, "shard_index": 0, "phase": "train",
            "test_values": [1.0],
        })
```

The production change this catches is accidentally putting private model or
TEST material in an operator-visible progress log.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_worker_status.RuntimeLogTests.test_runtime_log_rejects_test_prediction_label_and_metric_fields
```

Expected: FAIL because the validated log record does not exist.

- [ ] **Step 3: Implement atomic safe-log append**

```python
def append_runtime_log(status_root: Path, record: RuntimeLogRecord) -> Path:
    path = status_root / f"worker-{record.shard_index}.runtime.jsonl"
    with path.open("ab", buffering=0) as stream:
        stream.write(record.canonical_json_line())
        stream.flush()
        os.fsync(stream.fileno())
    return path
```

Emit a record at worker start, dataset phase changes, completion, and terminal
failure. Reuse the existing status reporter's safe values; do not read logs
from scheduling, resume, merge, publication, or model code.

- [ ] **Step 4: Run focused GREEN tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_worker_status \
  experiments.exp419_deterministic_cpu_parallel.tests.test_workers
```

Expected: PASS with JSON status and JSONL logs agreeing on safe progress fields.

- [ ] **Step 5: Commit the progress-log change**

```bash
git add experiments/exp419_deterministic_cpu_parallel/contracts.py \
        experiments/exp419_deterministic_cpu_parallel/workers.py \
        experiments/exp419_deterministic_cpu_parallel/coordinator.py \
        experiments/exp419_deterministic_cpu_parallel/tests/test_worker_status.py \
        experiments/exp419_deterministic_cpu_parallel/tests/test_workers.py
git commit -m "feat(exp419): persist safe worker runtime logs"
```

### Task 4: Freeze the corrected runtime contract and verify equivalence

**Files:**
- Modify: `docs/EXP419_PREREGISTRATION_KO.md`
- Modify: `docs/EXP419_RUNBOOK_KO.md`
- Modify: `experiments/exp419_deterministic_cpu_parallel/SOURCE_MANIFEST.json`
- Test: `experiments/exp419_deterministic_cpu_parallel/tests/test_cli.py`
- Test: `experiments/exp419_deterministic_cpu_parallel/tests/test_seed_parallel.py`
- Test: `experiments/exp417_causal_multihorizon_shadow/tests/test_runner.py`

**Interfaces:**
- Documents the parent-frozen official context, benchmark TRAIN-only context,
  once-per-worker DB verification, and safe JSONL log locations.
- Keeps the benchmark evidence schema bound to the corrected source manifest
  digest and `test_values_loaded == false`.

- [ ] **Step 1: Write the failing parity test**

```python
def test_session_backed_and_legacy_loader_paths_keep_exact_seed_outputs(self):
    legacy = self.run_fixture(loader_mode="legacy")
    session = self.run_fixture(loader_mode="session")
    self.assertEqual(session.model_hashes, legacy.model_hashes)
    self.assertEqual(session.oof_residual_bytes, legacy.oof_residual_bytes)
    self.assertEqual(session.candidate_coordinates, legacy.candidate_coordinates)
```

The production change this catches is changing scientific output while
optimizing only I/O and worker plumbing.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_seed_parallel.SessionParityTests.test_session_backed_and_legacy_loader_paths_keep_exact_seed_outputs
```

Expected: FAIL because the session-backed runtime path is not yet wired.

- [ ] **Step 3: Update preregistration/runbook and regenerate the manifest**

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from pathlib import Path
from scratch.benchmark_exp419_deterministic_cpu_parallel import propose_source_manifest
root = Path.cwd()
propose_source_manifest(
    root,
    output_path=root / "experiments/exp419_deterministic_cpu_parallel/SOURCE_MANIFEST.json",
)
PY
```

Document no TEST identity/metadata construction in the bounded smoke, the
one-time parent full-context freeze for official work, per-worker safe logs,
and the requirement for a new approval only after the corrected benchmark.

- [ ] **Step 4: Run full regression and static validation**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover \
  -s experiments/exp419_deterministic_cpu_parallel/tests -p 'test_*.py'
PYTHONPATH=. .venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p 'test_*.py'
PYTHONPATH=. .venv/bin/python -m unittest discover \
  -s experiments/exp414_missing_aware_near_repeat/tests -p 'test_*.py'
PYTHONPATH=. .venv/bin/python -m unittest discover \
  -s experiments/exp416_exp404_hard_test_confusion_audit/tests -p 'test_*.py'
.venv/bin/python -m py_compile \
  experiments/exp419_deterministic_cpu_parallel/*.py \
  scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  scratch/run_exp419_deterministic_cpu_parallel.py
PYTHONPATH=. .venv/bin/python - <<'PY'
from scratch.benchmark_exp419_deterministic_cpu_parallel import assert_exp419_label_isolation
print(assert_exp419_label_isolation())
PY
git diff --check
```

Expected: all suites pass, source manifest verifies, and no official root exists.

- [ ] **Step 5: Commit the frozen runtime contract**

```bash
git add docs/EXP419_PREREGISTRATION_KO.md docs/EXP419_RUNBOOK_KO.md \
        experiments/exp419_deterministic_cpu_parallel/SOURCE_MANIFEST.json \
        experiments/exp419_deterministic_cpu_parallel/tests/test_cli.py \
        experiments/exp419_deterministic_cpu_parallel/tests/test_seed_parallel.py \
        experiments/exp417_causal_multihorizon_shadow/tests/test_runner.py
git commit -m "docs(exp419): freeze train-only runtime evidence"
```

### Task 5: Run the corrected non-official benchmark and hand off

**Files:**
- Read: `/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced/EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json`
- Read: `<benchmark-root>.status/worker-*.runtime.jsonl`
- Read: `<benchmark-root>.status/exp419_worker_status.json`

**Interfaces:**
- Consumes the clean committed runtime, frozen source manifest, and five
  existing local input files.
- Produces only the no-clobber benchmark evidence plus safe local status/log
  sidecars; it produces no prediction, checkpoint, score, candidate, metric,
  or TEST-value artifact.

- [ ] **Step 1: Run the full benchmark preflight**

Run the exact preflight in `docs/EXP419_RUNBOOK_KO.md` and verify the balanced
benchmark root, official root, and official checkpoint root do not exist.

- [ ] **Step 2: Start the 19-dataset two-batch TRAIN-only benchmark with durable logs**

Run the exact benchmark command in `docs/EXP419_RUNBOOK_KO.md`, redirecting
the parent process output to a dated local `exp419_benchmark_<timestamp>.log`
and writing its PID and exit code beside that log. Do not create an automatic
monitor.

- [ ] **Step 3: Verify benchmark evidence and safe logs after completion**

Require worker candidates `(1, 2, 4, 6)`, 19 datasets per trial,
`test_values_loaded == false`, fixed seed-worker layout, finite outputs, no
fallback, aggregate RSS at most `27,487,790,694`, and a qualified 4- or
6-worker speedup of at least `2.0x`. Check every JSONL/status record contains
only the allowed fields.

- [ ] **Step 4: Report the benchmark result and request fresh official approval**

If every gate passes, report the selected worker count and log paths, then ask
for the exact new phrase:

```text
Exp419 1,117개 전체 label-free 병렬 실행 승인
```

If any gate fails, preserve the evidence, do not open an official root, and
report the fail-closed reason.
