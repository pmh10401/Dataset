# Exp419 Deterministic CPU Parallel Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, deterministic CPU process coordinator that runs the unchanged Exp417 dataset computation with 1, 2, 4, or 6 isolated workers and publishes an Exp419 bundle only after exact sequential-parallel parity and at least `2.0x` speedup.

**Architecture:** Exp419 imports the frozen Exp417 scientific pipeline and adds only TRAIN-only cost planning, deterministic shard assignment, spawned worker isolation, worker-local checkpoints, single-writer validated merge, runtime evidence, and Exp419 reporting. Workers never share model state, SQLite connections, random-number generators, or checkpoint roots; the parent alone merges validated records and publishes.

**Tech Stack:** Python 3.14, standard-library `multiprocessing` with `spawn`, PyTorch CPU, NumPy, Numba, SQLite, `unittest`, canonical JSON/SHA-256, existing Exp417 atomic checkpoint/publication primitives.

**2026-07-30 pre-official correction:** checkpoint-universe fixes are frozen
in reviewed commits `f54ac4d` and `b706393`; the user-approved balanced
benchmark design is commit `4ea4334`. The terminated diagnostic root
`/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark`
has status `143`, no benchmark JSON, no qualified trial, and no opened
official root. It remains immutable and may never be accepted, resumed,
merged, or reused. The corrected exact root is
`/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced`.

## Global Constraints

- The population is exactly 1,117 datasets from the frozen database SHA-256 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`.
- Exp417 preprocessing, model architecture, horizons, seeds, scoring, synthetic qualification, lane exclusion, and workload gates remain unchanged.
- TEST labels, anomaly locations, historical TEST/family performance, and human-review results are unreachable from Exp419.
- Existing Hard 2,218, Standard 806, Priority 31, and automatic-normal exact 35 coordinates remain read-only.
- MLX and MPS are excluded; canonical model execution is deterministic PyTorch CPU.
- Worker candidates are exactly `(1, 2, 4, 6)` with multiprocessing start method `spawn`.
- Every process uses one PyTorch intra-op thread, one inter-op thread, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1`.
- The selected runtime must be an eligible 4- or 6-worker trial, be exactly
  equivalent to the one-worker reference, use at most `27,487,790,694`
  resident bytes, and be at least `2.0x` faster. The mandatory 2-worker trial
  is comparison evidence only and cannot qualify the official run.
- The original 15 benchmark anchors remain included. Remaining TRAIN-only
  costs are added in `(-cost, dataset)` order until both 4- and 6-worker proxy
  headrooms reach `2.5x`; timing never selects the benchmark population.
- Existing Exp417 checkpoint records are local immutable evidence and are never imported into Exp419.
- The full 1,117-dataset run is forbidden until the user says exactly `Exp419 1,117개 전체 label-free 병렬 실행 승인`.

---

## File Structure

- `docs/EXP419_PREREGISTRATION_KO.md`: frozen scientific inheritance, runtime gates, artifacts, and prohibition boundary.
- `docs/EXP419_RUNBOOK_KO.md`: environment, tests, benchmark, resume, failure recovery, and separately approved full-run command.
- `experiments/exp419_deterministic_cpu_parallel/__init__.py`: narrow public API.
- `experiments/exp419_deterministic_cpu_parallel/config.py`: immutable runtime configuration.
- `experiments/exp419_deterministic_cpu_parallel/contracts.py`: validated cost, shard, worker, benchmark, and run records.
- `experiments/exp419_deterministic_cpu_parallel/planner.py`: TRAIN-only cost proxy, benchmark selection, and deterministic shard plan.
- `experiments/exp419_deterministic_cpu_parallel/workers.py`: thread pinning, spawned worker bootstrap, and worker-local sequential execution.
- `experiments/exp419_deterministic_cpu_parallel/coordinator.py`: resume, worker lifecycle, canonical merge, and global summary.
- `experiments/exp419_deterministic_cpu_parallel/reporting.py`: exact ten-file rendering and revalidation.
- `experiments/exp419_deterministic_cpu_parallel/publication.py`: Exp419 no-clobber atomic bundle wrapper.
- `experiments/exp419_deterministic_cpu_parallel/tests/`: synthetic unit and integration tests only.
- `scratch/benchmark_exp419_deterministic_cpu_parallel.py`: non-official 1/2/4/6-worker benchmark.
- `scratch/run_exp419_deterministic_cpu_parallel.py`: separately approved official all-population CLI.
- `experiments/exp417_causal_multihorizon_shadow/runner.py`: add one public checkpointed single-dataset boundary without changing computation.
- `experiments/exp417_causal_multihorizon_shadow/checkpoints.py`: add validated record import for a single-writer canonical merge.
- `experiments/exp417_causal_multihorizon_shadow/publication.py`: expose the already tested atomic exact-bundle core while retaining the Exp417 wrapper.

---

### Task 1: Freeze Exp419 Runtime Contracts and Preregistration

**Files:**
- Create: `docs/EXP419_PREREGISTRATION_KO.md`
- Create: `experiments/exp419_deterministic_cpu_parallel/__init__.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/config.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/contracts.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/__init__.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/test_config.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/test_contracts.py`

**Interfaces:**
- Produces: `Exp419FrozenConfig`, `FROZEN_CONFIG`, `TrainLengthGroupMeta`, `DatasetCost`, `ShardAssignment`, `ShardPlan`, `WorkerSpec`, `WorkerBootstrap`, `WorkerCompletion`, `ParallelRoots`, `ParallelRun`, `RuntimeTrial`, and `RuntimeBenchmark`.
- Consumes: immutable values copied from the approved Exp419 design and `experiments.exp417_causal_multihorizon_shadow.config.FROZEN_CONFIG`.

- [ ] **Step 1: Write failing immutable-config tests**

```python
from dataclasses import replace
import unittest

from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG


class Exp419ConfigTests(unittest.TestCase):
    def test_exact_runtime_contract_is_frozen(self):
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp419")
        self.assertEqual(FROZEN_CONFIG.worker_candidates, (1, 2, 4, 6))
        self.assertEqual(FROZEN_CONFIG.start_method, "spawn")
        self.assertEqual(FROZEN_CONFIG.speedup_gate, 2.0)
        self.assertEqual(FROZEN_CONFIG.memory_limit_bytes, 27_487_790_694)
        self.assertEqual(FROZEN_CONFIG.torch_threads, 1)

    def test_modified_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "worker_candidates"):
            replace(FROZEN_CONFIG, worker_candidates=(1, 8)).validate()
```

- [ ] **Step 2: Run the config test and verify RED**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_config -v
```

Expected: import failure because the Exp419 package does not exist.

- [ ] **Step 3: Implement the exact frozen config**

```python
@dataclass(frozen=True)
class Exp419FrozenConfig:
    experiment_id: str = "Exp419"
    expected_datasets: int = 1117
    worker_candidates: tuple[int, ...] = (1, 2, 4, 6)
    start_method: str = "spawn"
    speedup_gate: float = 2.0
    memory_limit_bytes: int = 27_487_790_694
    torch_threads: int = 1
    benchmark_quantiles: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
```

`validate()` must compare field names, exact types, and exact values against a new default instance, matching the Exp417 frozen-config pattern.

- [ ] **Step 4: Write failing contract validation tests**

Test that:

```python
DatasetCost("a", 10, "preprocessing_only")
ShardPlan(2, (
    ShardAssignment(0, ("a",), 10),
    ShardAssignment(1, ("b",), 20),
), ("a", "b"))
```

is valid, while duplicate datasets, missing datasets, out-of-order shard indices, negative/non-integer cost, unknown strata, non-finite timing, and non-lowercase SHA-256 digests raise `ValueError`.

- [ ] **Step 5: Run contract tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_contracts -v
```

Expected: import or missing-type failure.

- [ ] **Step 6: Implement minimal validated frozen dataclasses**

Use `tuple` and `MappingProxyType` in `__post_init__`, reject `bool` where an integer is required, reject non-finite floats, and expose `ShardPlan.sha256` as canonical JSON over worker count, assignments, costs, and population.

Use these exact fields:

```python
TrainLengthGroupMeta(dataset, length, series_count)
DatasetCost(dataset, cost, stratum)
ShardAssignment(shard_index, datasets, total_cost)
ShardPlan(worker_count, assignments, population, sha256=init_false)
WorkerSpec(shard, checkpoint_root)
WorkerBootstrap(resources, provenance, expected_context_sha256)
WorkerCompletion(
    shard_index, pid, completed_datasets, record_digests,
    elapsed_seconds, peak_rss_bytes, thread_environment, status, error
)
ParallelRoots(root, worker_roots, canonical_root)
ParallelRun(
    plan, completions, record_digests, summary,
    publication_allowed, status
)
RuntimeTrial(
    worker_count, elapsed_seconds, peak_rss_bytes, parity,
    finite, fallback, completed_datasets, evidence_sha256
)
RuntimeBenchmark(
    trials, selected_worker_count, status, benchmark_datasets, sha256=init_false
)
```

`stratum` is exactly one of `preprocessing_only`, `diagnostic_oof`, or
`candidate_eligible`; worker/run/benchmark status values are closed literal
sets declared next to their dataclasses.

- [ ] **Step 7: Write the Korean preregistration**

Copy the exact constraints, cost formula, benchmark strata, parity gates, memory gate, artifact list, local-only rule, Exp420 audit separation, and exact full-run approval phrase from the approved design. State that Exp419 inherits rather than retunes Exp417.

- [ ] **Step 8: Run Task 1 tests and commit**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_config \
  experiments.exp419_deterministic_cpu_parallel.tests.test_contracts -v
git add docs/EXP419_PREREGISTRATION_KO.md \
  experiments/exp419_deterministic_cpu_parallel
git commit -m "feat(exp419): freeze parallel runtime contract"
```

---

### Task 2: Build the TRAIN-Only Cost and Shard Planner

**Files:**
- Create: `experiments/exp419_deterministic_cpu_parallel/planner.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/test_planner.py`

**Interfaces:**
- Consumes: `TrainLengthGroupMeta(dataset: str, length: int, series_count: int)`.
- Produces:
  - `estimate_dataset_cost(dataset: str, groups: Sequence[TrainLengthGroupMeta]) -> DatasetCost`
  - `select_benchmark_datasets(costs: Sequence[DatasetCost]) -> tuple[str, ...]`
  - `build_shard_plan(costs: Sequence[DatasetCost], worker_count: int) -> ShardPlan`

- [ ] **Step 1: Write failing exact-cost tests**

```python
def test_cost_proxy_uses_frozen_group_weights(self):
    groups = (
        TrainLengthGroupMeta("a", 10, 49),
        TrainLengthGroupMeta("a", 20, 50),
        TrainLengthGroupMeta("a", 30, 100),
    )
    result = estimate_dataset_cost("a", groups)
    self.assertEqual(result.cost, 49 * 10 + 50 * 20 * 30 + 100 * 30 * 81)
    self.assertEqual(result.stratum, "candidate_eligible")
```

Also test preprocessing-only and diagnostic-OOF strata, duplicate lengths,
cross-dataset groups, zero length/count, and rejection when the summed cost
would exceed signed 64-bit maximum `9_223_372_036_854_775_807`.

- [ ] **Step 2: Run the cost tests and verify RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_planner.PlannerTests.test_cost_proxy_uses_frozen_group_weights -v
```

Expected: `estimate_dataset_cost` is missing.

- [ ] **Step 3: Implement the minimal cost function**

Use exactly:

```python
weight = 1 if count < 50 else 30 if count < 100 else 81
cost += count * length * weight
```

Derive the stratum from the largest group count without reading TEST metadata.

- [ ] **Step 4: Write failing deterministic benchmark-selection tests**

Create at least seven records in each stratum. Assert selection uses indices
`floor(q * (n - 1))` for `q=(0,.25,.5,.75,1)`, removes duplicates without
replacement, and returns globally sorted dataset names. Assert that changing a
TEST-like field is impossible because `DatasetCost` has no such field.

- [ ] **Step 5: Implement benchmark selection and verify GREEN**

Sort each stratum by `(cost, dataset)`, select the frozen positions, remove
duplicates while retaining quantile order, then return the selected dataset
names in canonical lexical order.

- [ ] **Step 6: Write failing largest-first shard tests**

Assert:

- every dataset appears once;
- descending cost with dataset tie-break is used;
- the currently lightest shard wins, then lowest shard index;
- worker counts outside `(1,2,4,6)` fail;
- repeated calls produce the same `ShardPlan.sha256`.

- [ ] **Step 7: Implement shard planning and run all planner tests**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_planner -v
```

Expected: all planner tests pass.

- [ ] **Step 8: Commit**

```bash
git add experiments/exp419_deterministic_cpu_parallel/planner.py \
  experiments/exp419_deterministic_cpu_parallel/tests/test_planner.py
git commit -m "feat(exp419): plan deterministic CPU shards"
```

---

### Task 3: Expose Safe Single-Dataset Execution and Validated Checkpoint Import

**Files:**
- Modify: `experiments/exp417_causal_multihorizon_shadow/runner.py`
- Modify: `experiments/exp417_causal_multihorizon_shadow/checkpoints.py`
- Modify: `experiments/exp417_causal_multihorizon_shadow/tests/test_runner.py`
- Modify: `experiments/exp417_causal_multihorizon_shadow/tests/test_checkpoints.py`

**Interfaces:**
- Produces:
  - `run_dataset_checkpointed(dataset: str, context: FrozenRunContext, checkpoints: Exp417CheckpointStore, *, runtime: RunnerRuntime | None = None) -> DatasetResult`
  - `Exp417CheckpointStore.import_validated_dataset(source: Exp417CheckpointStore, dataset: str) -> str`
- `import_validated_dataset` returns the imported canonical record SHA-256.
- Existing `run_dataset`, `resume_or_run_population`, `write_dataset`, and Exp417 official behavior remain unchanged.

- [ ] **Step 1: Write a failing checkpointed-runner test**

Use the existing synthetic runtime and store helpers. Assert one call writes
exactly one valid record, a second call refuses the duplicate, and a dataset
outside `context.datasets` is rejected.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_runner.RunnerPopulationTests.test_checkpointed_single_dataset_uses_transactional_store -v
```

Expected: missing `run_dataset_checkpointed`.

- [ ] **Step 3: Add the minimal public wrapper**

The implementation validates the concrete store and calls the existing
`_run_dataset(dataset, context, checkpoints, runtime or
RunnerRuntime.production())`. It contains no model, score, qualification, or
error-policy logic of its own.

- [ ] **Step 4: Write failing validated-import tests**

Create a source store whose expected population is `("a",)` and a target store
whose expected population is `("a","b")`, with identical provenance. Write
`a` to the source and assert:

```python
digest = target.import_validated_dataset(source, "a")
self.assertEqual(digest, target.completed_record_digests()["a"])
```

Add tests that reject provenance mismatch, wrong shard inventory, missing
record, duplicate target record, corrupt source JSON, non-regular source
entries, and source/target path aliasing.

- [ ] **Step 5: Run import tests and verify RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints.Exp417CheckpointTests.test_imports_one_validated_shard_record -v
```

Expected: missing `import_validated_dataset`.

- [ ] **Step 6: Implement descriptor-pinned single-writer import**

Under target and source root locks:

1. verify both path identities and exact provenance equality;
2. load and fully reconstruct the source record through existing validation;
3. require `dataset` in both expected inventories and exactly once in source;
4. reject an existing target record;
5. copy the canonical source bytes to a target temporary regular file;
6. fsync, publish no-clobber, update target progress atomically, and fsync the
   directory;
7. return the validated record SHA-256.

Reuse existing private validation and rollback primitives rather than adding a
second JSON codec.

- [ ] **Step 7: Run complete Exp417 checkpoint and runner regressions**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints \
  experiments.exp417_causal_multihorizon_shadow.tests.test_runner -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add experiments/exp417_causal_multihorizon_shadow/runner.py \
  experiments/exp417_causal_multihorizon_shadow/checkpoints.py \
  experiments/exp417_causal_multihorizon_shadow/tests/test_runner.py \
  experiments/exp417_causal_multihorizon_shadow/tests/test_checkpoints.py
git commit -m "feat(exp417): expose validated dataset checkpoint transfer"
```

---

### Task 4: Implement Isolated Spawned Workers

**Files:**
- Create: `experiments/exp419_deterministic_cpu_parallel/workers.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/test_workers.py`

**Interfaces:**
- Consumes:
  - `WorkerSpec(shard: ShardAssignment, checkpoint_root: Path)`
  - `WorkerBootstrap(resources: RunnerResources, provenance: Exp417CheckpointProvenance, expected_context_sha256: str)`
- Produces:
  - `pin_worker_threads() -> Mapping[str, str]`
  - `run_worker(spec: WorkerSpec, bootstrap: WorkerBootstrap) -> WorkerCompletion`
  - `spawn_workers(specs: Sequence[WorkerSpec], bootstrap: WorkerBootstrap) -> tuple[WorkerCompletion, ...]`

- [ ] **Step 1: Write failing thread-pinning tests**

Patch PyTorch thread setters, call `pin_worker_threads()`, and assert exact
calls with `1` plus exact environment values for OMP, OpenBLAS, and vecLib.
Assert a pre-existing conflicting value raises instead of being overwritten.

- [ ] **Step 2: Verify RED and implement thread pinning**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_workers.WorkerTests.test_pins_exact_thread_contract -v
```

Set the environment before importing or initializing model work. Return an
immutable snapshot that is recorded in `WorkerCompletion`.

- [ ] **Step 3: Write failing isolated-worker tests**

Using injected synthetic bootstrap functions, assert:

- a worker creates only its own checkpoint root;
- datasets execute in assigned order;
- a dataset outside the shard is refused;
- worker completion records exact completed names and record digests;
- an exception produces non-zero terminal evidence and no success status.

- [ ] **Step 4: Implement `run_worker` with the existing Exp417 boundary**

The worker reconstructs the frozen context, checks its digest, creates an
`Exp417CheckpointStore` whose expected datasets equal the shard datasets, and
calls `run_dataset_checkpointed` sequentially. Record `resource.getrusage`
maximum RSS and monotonic elapsed seconds.

- [ ] **Step 5: Write a real `spawn` integration test**

Run two tiny top-level synthetic worker functions through
`multiprocessing.get_context("spawn")`. Assert different PIDs, disjoint roots,
canonical completion order, and propagated failure exit codes. Do not use a
fork-only closure or lambda.

- [ ] **Step 6: Implement `spawn_workers` and verify GREEN**

Use explicit `Process` and one result pipe per worker. Close unused pipe ends,
join every process, reject missing/multiple messages, and return completions
ordered by shard index. Do not add a polling loop or background monitor.

- [ ] **Step 7: Run worker tests and commit**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_workers -v
git add experiments/exp419_deterministic_cpu_parallel/workers.py \
  experiments/exp419_deterministic_cpu_parallel/tests/test_workers.py
git commit -m "feat(exp419): run isolated CPU worker shards"
```

---

### Task 5: Coordinate Resume, Merge, and Runtime Selection

**Files:**
- Create: `experiments/exp419_deterministic_cpu_parallel/coordinator.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/test_coordinator.py`

**Interfaces:**
- Produces:
  - `resume_or_run_parallel(context: FrozenRunContext, plan: ShardPlan, roots: ParallelRoots, bootstrap: WorkerBootstrap) -> ParallelRun`
  - `merge_worker_checkpoints(context: FrozenRunContext, plan: ShardPlan, worker_stores: Mapping[int, Exp417CheckpointStore], canonical: Exp417CheckpointStore) -> Mapping[str, str]`
  - `select_runtime(trials: Sequence[RuntimeTrial]) -> RuntimeBenchmark`

- [ ] **Step 1: Write failing canonical-merge tests**

Build two synthetic shard stores that finish in reverse order. Assert the
canonical store imports records in `context.datasets` order and its digest
mapping is independent of worker completion order.

Add separate tests for missing, duplicate, extra, wrong-shard, wrong-plan,
wrong-provenance, corrupt, and terminal-failure records. Each must leave the
canonical store incomplete and return no publishable `ParallelRun`.

- [ ] **Step 2: Verify RED and implement minimal merge**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_coordinator.CoordinatorTests.test_merge_is_canonical_when_workers_finish_out_of_order -v
```

For each name in `context.datasets`, look up its exact preregistered shard and
call `canonical.import_validated_dataset(source, name)`. Never scan for an
alternative copy in another shard.

- [ ] **Step 3: Write failing resume tests**

Prepopulate selected shard records, reopen all stores, and assert only missing
records are scheduled in their original shard. A completed record in the wrong
shard must fail rather than migrate. A complete canonical store must skip
workers and only reconstruct the global summary.

- [ ] **Step 4: Implement resume and failure propagation**

Freeze the shard-plan digest in every root. Revalidate worker completions
against on-disk record digests before merge. Any non-zero worker exit, terminal
dataset result, or aggregate RSS above the frozen limit returns
`publication_allowed=False`.

- [ ] **Step 5: Write failing runtime-selection tests**

Create exact trials for 1, 2, 4, and 6 workers. Assert selection chooses the
fastest trial satisfying exact parity, memory, finite metrics, no fallback,
and `speedup>=2.0`. Assert no passing multi-worker trial returns
`no_qualified_parallel_runtime`.

- [ ] **Step 6: Implement runtime selection and run coordinator tests**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_coordinator -v
```

- [ ] **Step 7: Commit**

```bash
git add experiments/exp419_deterministic_cpu_parallel/coordinator.py \
  experiments/exp419_deterministic_cpu_parallel/tests/test_coordinator.py
git commit -m "feat(exp419): merge and qualify parallel checkpoints"
```

---

### Task 6: Render and Atomically Publish the Exact Exp419 Bundle

**Files:**
- Modify: `experiments/exp417_causal_multihorizon_shadow/publication.py`
- Modify: `experiments/exp417_causal_multihorizon_shadow/tests/test_publication.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/reporting.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/publication.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/test_reporting.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/test_publication.py`

**Interfaces:**
- Produces:
  - `publish_exact_bundle(payloads: Mapping[str, bytes], output_dir: Path, *, exact_names: tuple[str, ...], pre_publish_verify: Callable[[], None]) -> Mapping[str, str]`
  - `render_exp419_payloads(run: ParallelRun, benchmark: RuntimeBenchmark, plan: ShardPlan, trusted_checkpoint: TrustedCheckpointProvenance) -> Mapping[str, bytes]`
  - `verify_exp419_payloads(payloads: Mapping[str, bytes], *, trusted_checkpoint: TrustedCheckpointProvenance) -> None`
  - `publish_exp419(...) -> Mapping[str, str]`

- [ ] **Step 1: Write a failing generic-publication regression**

Add a test proving `publish_exact_bundle` publishes a caller-supplied exact
two-file set atomically, rejects missing/extra names, preserves no-clobber, and
runs the verification callback immediately before exchange. Retain all
existing Exp417 adversarial filesystem tests.

- [ ] **Step 2: Verify RED and extract the existing atomic core**

Keep `publish_exp417` as a wrapper with its unchanged exact file list and
manifest behavior. Move no scientific/reporting logic into the generic core.

- [ ] **Step 3: Run the full Exp417 publication suite**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_publication -v
```

- [ ] **Step 4: Write failing Exp419 rendering tests**

Assert this exact tuple:

```python
(
    "exp419_shadow_predictions.jsonl",
    "exp419_dataset_qualification.csv",
    "exp419_oof_normal_metrics.csv",
    "exp419_synthetic_safety.csv",
    "exp419_backend_benchmark.json",
    "exp419_parallel_runtime_benchmark.json",
    "exp419_frozen_config.json",
    "exp419_shard_plan.json",
    "EXP419_RESULT_REPORT_KO.md",
    "MANIFEST.json",
)
```

Parse every payload and assert:

- prediction rows contain no forbidden evaluation or label fields;
- Exp419 config embeds the exact Exp417 scientific config and Exp419 runtime;
- shard-plan bytes reproduce `ShardPlan.sha256`;
- runtime benchmark names the selected worker count and all four trials;
- report states Shadow-only, label-free, no lane promotion;
- manifest binds every other payload, trusted canonical checkpoint, Git
  commit, runtime, shard records, and benchmark digest.

- [ ] **Step 5: Implement rendering by composing verified Exp417 evidence**

Use Exp417 scientific renderers for predictions and scientific CSV evidence,
rename only through an explicit mapping, wrap the frozen config with Exp419
runtime identity, and generate a new Exp419 report and manifest. Do not
string-replace serialized Exp417 content.

- [ ] **Step 6: Write adversarial Exp419 verification tests**

Mutate each payload in turn; add a candidate, remove a dataset, change a model
hash, alter worker count, swap a shard, inject `precision`, and alter manifest
hashes. Every mutation must make `verify_exp419_payloads` fail.

- [ ] **Step 7: Implement verification and publication wrapper**

`publish_exp419` requires the exact ten-file set, a complete canonical
checkpoint, and a fresh input/Git revalidation callback before delegating to
`publish_exact_bundle`.

- [ ] **Step 8: Run Task 6 tests and commit**

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_publication \
  experiments.exp419_deterministic_cpu_parallel.tests.test_reporting \
  experiments.exp419_deterministic_cpu_parallel.tests.test_publication -v
git add experiments/exp417_causal_multihorizon_shadow/publication.py \
  experiments/exp417_causal_multihorizon_shadow/tests/test_publication.py \
  experiments/exp419_deterministic_cpu_parallel
git commit -m "feat(exp419): publish verified parallel artifacts"
```

---

### Task 7: Add Benchmark, Official CLI, Label Guards, and Runbook

**2026-07-30 human ruling:** the user approved replacing generic AST
inventory/taint proof with `SOURCE_MANIFEST.json`. Freeze the exact eight
package runtime modules plus both scratch CLIs. Verification must use
descriptor-pinned no-follow regular-file reads, canonical sorted repo-relative
paths, and exact SHA-256, rejecting missing/extra/duplicate/traversal,
symlink, and non-regular inputs. Keep only a narrow direct AST denial for
explicit label loaders, frozen TEST-label artifact names, and label-column
projection. Bind the manifest digest into every trial and benchmark evidence,
and require the current post-clean-Git digest in the official verifier.
Manifest proposal is an explicit offline development action only.

**Files:**
- Create: `scratch/benchmark_exp419_deterministic_cpu_parallel.py`
- Create: `scratch/run_exp419_deterministic_cpu_parallel.py`
- Create: `experiments/exp419_deterministic_cpu_parallel/tests/test_cli.py`
- Create: `docs/EXP419_RUNBOOK_KO.md`

**Interfaces:**
- Benchmark CLI produces local `EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json`.
- Official CLI accepts database, four frozen lane artifacts, benchmark
  evidence, checkpoint root, and new output root.
- Official CLI has no worker override; it reads the selected count from the
  verified frozen benchmark.

- [ ] **Step 1: Write failing CLI-order tests**

Inject stage callables and assert exact fail-stop order:

```text
clean Git -> label guard -> freeze inputs -> verify benchmark ->
open roots -> run/resume -> revalidate -> render -> verify -> publish
```

Assert a failure at each boundary prevents every later stage.

- [ ] **Step 2: Write failing approval and argument tests**

Assert:

- output must be new and disjoint from every checkpoint/shard root;
- checkpoint may be new or exact-provenance resume;
- benchmark must contain all four trials and a selected passing worker count;
- no `--workers`, label path, threshold, seed, horizon, or budget argument
  exists;
- dirty Git state prevents benchmark and official execution.

- [ ] **Step 3: Implement the two CLIs minimally**

The benchmark selects the original anchors through
`select_benchmark_datasets`, then freezes the exact balanced proof through
`select_balanced_benchmark_datasets`. It runs each worker candidate with fresh
disjoint roots, verifies parity, and writes one local no-clobber JSON file.
For the frozen inputs this is exactly 19 datasets: the original 15 plus
`StarLightCurves_normal_2`, `HandOutlines_normal_1`,
`StarLightCurves_normal_1`, and `HandOutlines_normal_0`. The original total
cost is `388,627,551` with a `1.0996059372x` proxy ceiling; the corrected
total is `888,487,623`, and both 4- and 6-worker proxy headrooms are
`2.5139397937x`. The measured gate remains `2.0x`.

The official CLI verifies the benchmark digest and runs only the frozen
selected worker count. It recomputes the exact proof from frozen TRAIN
metadata and rejects the terminated diagnostic root or any descendant before
opening official roots. It contains the exact approval phrase in help and
runbook text but does not infer approval from environment state, prior
blanket approval, a file, or benchmark success.

- [ ] **Step 4: Add frozen-source manifest and label-isolation tests**

Freeze the Exp419 package and both scripts by exact path and SHA-256. Reject
manifest/source tamper, missing/extra/duplicate/traversal paths, symlinks,
non-regular files, forbidden label-loader imports/calls/attributes,
`label`/`labels_blob` SQL selection, and frozen label artifact names.

- [ ] **Step 5: Write the runbook**

Include:

- exact environment/version checks;
- original Exp417 stop evidence and immutable 94-checkpoint location;
- full unit/regression commands;
- label-free benchmark command;
- benchmark acceptance inspection;
- exact anchor/addition, proxy gate, total-cost, 2/4/6 load/headroom, and proof
  digest inspection;
- invalid-root retention and non-reuse plus the corrected exact root;
- resume and worker-failure procedure;
- exact official approval phrase and command;
- no automatic polling or monitoring;
- local-only artifact and checkpoint rule.

- [ ] **Step 6: Run CLI tests and syntax checks**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_cli -v
.venv/bin/python -m py_compile \
  scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  scratch/run_exp419_deterministic_cpu_parallel.py
```

- [ ] **Step 7: Commit**

```bash
git add scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  scratch/run_exp419_deterministic_cpu_parallel.py \
  experiments/exp419_deterministic_cpu_parallel/tests/test_cli.py \
  docs/EXP419_RUNBOOK_KO.md
git commit -m "feat(exp419): prepare deterministic parallel execution"
```

---

### Task 8: Full Regression, Balanced Parity Benchmark, and Readiness Report

**Files:**
- Modify only if evidence exposes a tested defect in Task 1--7 files.
- Retain local diagnostic-only:
  `/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark/`
- Create local only:
  `/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced/`

**Interfaces:**
- Consumes the completed Exp419 package and frozen private inputs.
- Produces a local immutable benchmark artifact and readiness decision; it
  does not start the 1,117-dataset run.

- [ ] **Step 1: Run all Exp417 and Exp419 tests**

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp419_deterministic_cpu_parallel/tests -p "test_*.py"
```

Expected: both commands exit 0 with zero failures and errors.

- [ ] **Step 2: Run syntax and label-isolation scans**

```bash
.venv/bin/python -m py_compile \
  experiments/exp419_deterministic_cpu_parallel/*.py \
  scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  scratch/run_exp419_deterministic_cpu_parallel.py
rg -n "labels_blob|TEST_LABEL|load.*label|SELECT.*label" \
  experiments/exp419_deterministic_cpu_parallel \
  scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  scratch/run_exp419_deterministic_cpu_parallel.py
```

Expected: compilation exits 0; the source scan has no prohibited access path.

- [ ] **Step 3: Verify frozen inputs and run the private benchmark once**

Use the exact command recorded in `docs/EXP419_RUNBOOK_KO.md`, writing only to:

```text
/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced/
```

Do not touch or resume the terminated diagnostic root. Do not run competing
MPS/MLX work and do not create an automatic monitor, polling loop, cron, or
keepalive.

- [ ] **Step 4: Verify benchmark evidence**

Check:

- all four worker trials exist;
- the proof contains the exact original 15 anchors plus the four frozen
  additions, total cost `888,487,623`, proxy gate `2.5`, and exact 2/4/6
  loads/headrooms;
- one-worker matches the Exp417 scientific reference;
- selected parallel trial has exact model, calibration, qualification, and
  candidate identity;
- speedup is at least `2.0x`;
- aggregate peak RSS is at most `27,487,790,694`;
- no fallback, missing, duplicate, extra, or terminal dataset exists;
- benchmark manifest and files match their SHA-256 digests.
- no `trial_roots` entry names the terminated diagnostic root or a descendant.

- [ ] **Step 5: Run final clean-tree verification**

```bash
git status --short --branch
git diff --check
git log -1 --oneline
```

If Task 8 required a tested code correction, rerun Steps 1--4 and commit only
that correction. Otherwise do not create an evidence-only Git commit because
private benchmark results must remain local.

- [ ] **Step 6: Report readiness and stop**

Report the exact selection proof, selected worker count, one-worker time,
selected time, speedup,
peak memory, parity outcome, test counts, commit, benchmark SHA-256, and local
checkpoint/output paths. Do not start the full run. Request the exact approval:

`Exp419 1,117개 전체 label-free 병렬 실행 승인`
