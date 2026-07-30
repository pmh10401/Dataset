# Exp419 Deterministic CPU Parallel Runtime Design

Date: 2026-07-30

## 1. Purpose

Exp419 accelerates the frozen Exp417 causal multi-horizon Shadow workflow by
running independent datasets in isolated CPU worker processes. It is an
execution-equivalence experiment, not a new score source.

Exp419 must preserve the Exp417 scientific result for every dataset:

- the same 1,117-dataset population and source database;
- the same normal-TRAIN-only preprocessing and identity-grouped folds;
- the same causal CNN, horizons, losses, seeds, and early stopping;
- the same OOF calibration, synthetic qualification, exact-normal guard,
  frozen-lane exclusion, and workload gates;
- no TEST labels, anomaly locations, historical TEST/family performance, or
  human-review results;
- no changes to Hard, Standard, Priority, or automatic-normal lanes.

The stopped Exp417 sequential attempt and its 94 validated dataset
checkpoints remain immutable local evidence. Exp419 does not import, rewrite,
or claim those records as Exp419 results.

## 2. Scope

The first Exp419 implementation changes only population orchestration and
checkpoint publication. It does not batch model inference, change PyTorch
operators, change thread counts implicitly, use MPS, or introduce a new model
implementation.

MLX remains excluded by the user's explicit project decision. MPS remains
excluded because the frozen Exp417 benchmark was numerically acceptable but
only `0.6709759390840965x` as fast as CPU, below the `1.1x` adoption gate.
Numba remains limited to the already audited deterministic score aggregation
path.

Batch inference and within-dataset seed/fold parallelism are deferred. They
may be considered only under a later experiment number if dataset-level
parallelism does not provide sufficient speed.

## 3. Experiment Identity and Output Boundary

Exp419 receives its own preregistration, immutable runtime configuration,
checkpoint provenance, output directory, manifest, and result report. It does
not publish files under an Exp417 output directory.

The Exp419 manifest binds:

- the exact Exp417 frozen scientific configuration digest;
- the Exp417 model, preprocessing, scoring, qualification, and lane inputs;
- the Exp419 scheduler, worker, merge, and publication source digests;
- Git commit, database identity, population identity, runtime versions,
  worker count, multiprocessing start method, and thread environment;
- the deterministic shard plan and every worker checkpoint digest;
- the sequential-versus-parallel parity audit and speed benchmark.

Exp420, not Exp418, is the reserved retrospective TEST-label audit for a
successfully frozen Exp419 prediction. Exp418 remains reserved for an Exp417
prediction if one is ever completed under the original Exp417 contract.

## 4. Architecture

### 4.1 Parent coordinator

One parent process owns the run state. It:

1. verifies clean Git state, database identity, population, frozen lanes,
   label isolation, Exp417 scientific configuration, and Exp419 runtime
   configuration;
2. derives a deterministic TRAIN-only cost estimate for each incomplete
   dataset;
3. assigns datasets to a fixed number of worker shards;
4. launches isolated worker processes using the macOS `spawn` start method;
5. waits for workers and records exit state without polling automation;
6. reopens and validates every worker checkpoint;
7. merges validated records in canonical dataset order;
8. applies the unchanged global workload gate;
9. revalidates all frozen inputs and atomically publishes the Exp419 bundle.

The parent never lets workers write to the canonical merged checkpoint or
official output directory.

### 4.2 Deterministic shard planner

The planner uses only normal TRAIN metadata available before TEST values are
opened. For an exact-length group with `n` series of length `L`, its integer
cost proxy is `n * L * w`, where `w=1` for `n<50`, `w=30` for `50<=n<100`,
and `w=81` for `n>=100`. Dataset cost is the sum of its group costs. These
weights represent preprocessing-only, OOF selection/refit, and
OOF/final/synthetic work respectively; they never change model computation.
The planner must not use TEST labels, anomaly positions, TEST scores,
historical family performance, or human review.

Datasets are ordered by descending estimated cost with the dataset name as
the stable tie-breaker. Each dataset is assigned to the currently lightest
shard, with the shard index as the stable tie-breaker. This largest-first
greedy assignment is deterministic and balances long datasets without
changing their scientific computation.

The complete shard plan is hashed and stored before workers start. A worker
may process only its assigned dataset names in their recorded order.

### 4.3 Worker isolation

Each worker:

- independently reconstructs and verifies the same frozen run context;
- opens the database read-only;
- uses the unchanged Exp417 production runtime for each assigned dataset;
- writes only to its own no-clobber checkpoint root;
- records its process identity, assigned shard, dataset start/end times,
  terminal state, and peak resident memory;
- exits non-zero on an unexplained error, duplicate dataset, provenance
  mismatch, non-finite evidence, or checkpoint write failure.

PyTorch model seeds remain isolated because a dataset is processed wholly
inside one spawned process. No dataset is split across workers. Workers do
not share model objects, random-number generators, SQLite connections, or
checkpoint writers.

### 4.4 Validated merge

The parent treats worker checkpoint files as untrusted input. For every
expected dataset it:

1. locates exactly one worker record in the preregistered shard;
2. verifies worker and scientific provenance;
3. reconstructs the typed Exp417 dataset result and complete checkpoint
   evidence;
4. recalculates the canonical payload digest;
5. rejects duplicates, missing datasets, extra datasets, truncated JSON,
   non-finite values, invalid types, or source-shard mismatches;
6. writes the verified record through a single canonical checkpoint writer.

The canonical writer follows the original 1,117-dataset order. Filesystem
publication is no-clobber and atomic. A merge failure preserves worker
checkpoints and the failed staging area for diagnosis but publishes no
official output.

## 5. Runtime Configuration and Worker Selection

The benchmark candidates are exactly `1`, `2`, `4`, and `6` CPU workers.
Every process uses `torch.set_num_threads(1)`,
`torch.set_num_interop_threads(1)`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1`. The one-worker
reference first has to match the stopped Exp417 scientific computation before
parallel candidates can be considered.

The TRAIN-only population is split into three benchmark strata:

1. preprocessing-only: no exact-length group has at least 50 normal series;
2. diagnostic OOF: at least one group has 50--99 normal series and none has
   at least 100;
3. candidate-eligible: at least one group has at least 100 normal series.

Within each non-empty stratum, datasets are ordered by `(cost, dataset)` and
the records at quantiles `0`, `0.25`, `0.50`, `0.75`, and `1.0` are selected
using index `floor(q * (n-1))`. Duplicate selections are removed without
replacement. The resulting sorted dataset names and identities are written
to the benchmark manifest before timing begins. The same subset, inputs, and
seeds are used for every worker count.

The selected official worker count must satisfy all of:

1. finite output and no unexplained worker fallback;
2. exact dataset status and candidate-coordinate identity;
3. exact canonical model hashes;
4. exact score-calibration and TRAIN-contract hashes;
5. exact qualification and lane-exclusion decisions;
6. no missing, duplicate, or extra dataset;
7. peak total resident memory at most 80% of physical memory, exactly
   `27,487,790,694` bytes on this 32-GiB host;
8. at least `2.0x` wall-clock speedup relative to the one-worker Exp419
   reference on the exact benchmark subset.

Among passing candidates, Exp419 selects the fastest worker count. If no
candidate with more than one worker passes, Exp419 records
`no_qualified_parallel_runtime` and no full-population run is authorized.

The selected worker count and thread environment are frozen before the
1,117-dataset run. The full run cannot reduce parity requirements, change the
worker count, or substitute a different backend after observing runtime or
unlabeled candidate counts.

## 6. Resume and Failure Handling

Worker and canonical checkpoint reuse requires exact equality of:

- Git commit and all source digests;
- database, population, and lane identities;
- Exp417 scientific configuration;
- Exp419 runtime configuration and shard-plan digest;
- Python, NumPy, Numba, PyTorch, and platform versions;
- worker count, start method, and thread environment;
- model, preprocessing, seed, synthetic, scoring, and merge contracts.

On resume, the parent validates completed worker records before scheduling
only missing records in their original shards. It never moves an incomplete
dataset to another shard during the same official run.

Failure policy is fail-closed:

- one worker failure prevents merge and publication;
- one dataset terminal failure prevents official publication;
- a worker timeout is not guessed or silently retried;
- a corrupt worker root is quarantined without deleting evidence;
- candidate thresholds, horizons, seeds, budgets, and routing are never
  changed to recover from a runtime failure;
- partial worker and canonical checkpoints remain local and are never
  exported to GitHub.

Stopping the parent sends a graceful termination request to workers and waits
for dataset-level atomic checkpoint boundaries. A forced termination may
discard only the currently incomplete dataset in each worker.

## 7. Label Isolation and Governance

Exp419 adds no TEST-label loader. Static source guards must prove that the
Exp419 package cannot import, open, hash, stat, query, or name the frozen
TEST-label artifact. Worker SQL is inherited from the Exp417 values-and-
identity-only path.

Generated predictions, checkpoints, metrics, benchmarks derived from the
private dataset, and dataset-level reports remain local. GitHub may contain
code, preregistration, runbooks, synthetic tests, and synthetic benchmark
evidence only.

The full 1,117-dataset Exp419 run requires a new exact user approval after all
tests, parity checks, and the worker-count benchmark pass.

## 8. Testing Strategy

Implementation follows test-driven development.

### 8.1 Planner tests

- identical inputs always produce the same shard plan and digest;
- every dataset appears exactly once;
- tie-breaking is stable;
- only TRAIN metadata enters the estimator;
- invalid worker counts and non-finite costs fail closed.

### 8.2 Worker tests

- a worker rejects datasets outside its shard;
- two workers use disjoint checkpoint roots;
- worker failure is surfaced to the parent;
- duplicate and truncated records are rejected;
- no TEST-label path or SQL column is reachable.

### 8.3 Merge tests

- shuffled worker completion still yields canonical dataset order;
- exact records merge successfully;
- missing, duplicate, extra, corrupt, or wrong-shard records block merge;
- a failed merge creates no official bundle;
- resume schedules only missing records in their original shards.

### 8.4 Equivalence tests

The same synthetic and representative label-free datasets run through
one-worker and multi-worker paths. Tests compare exact:

- dataset status and qualification;
- candidate coordinates and seed decisions;
- canonical model hashes;
- TRAIN identity and score-calibration hashes;
- workload accounting and manifest identities.

Elapsed time, process identifiers, worker-local paths, and peak memory are
expected runtime metadata and are compared through their schema and bounds,
not byte equality.

### 8.5 Performance benchmark

The benchmark records wall time, CPU time, peak aggregate resident memory,
worker utilization, completed datasets, parity results, and fallback reasons
for `1`, `2`, `4`, and `6` workers. Benchmark output is local, immutable, and
hash-bound into the Exp419 manifest.

## 9. Acceptance Criteria

Exp419 implementation readiness requires:

- all Exp417 regression tests pass unchanged;
- all Exp419 unit, integration, failure-injection, label-isolation, resume,
  and publication tests pass;
- static source scanning finds no TEST-label access path;
- one-worker Exp419 matches the frozen Exp417 scientific computation;
- the selected parallel worker count passes every parity requirement;
- the selected runtime is at least `2.0x` faster on the frozen benchmark;
- peak benchmark memory remains at or below 80% of 32 GiB;
- the worktree is clean at the verified commit;
- the preregistration, runbook, benchmark digest, and checkpoint provenance
  agree exactly.

Only after these conditions pass may the user be asked for the exact approval:

`Exp419 1,117개 전체 label-free 병렬 실행 승인`

## 10. Official Artifacts

After all dataset records and the global workload gate pass, Exp419 atomically
publishes exactly these ten files to a new no-clobber directory:

1. `exp419_shadow_predictions.jsonl`;
2. `exp419_dataset_qualification.csv`;
3. `exp419_oof_normal_metrics.csv`;
4. `exp419_synthetic_safety.csv`;
5. `exp419_backend_benchmark.json`;
6. `exp419_parallel_runtime_benchmark.json`;
7. `exp419_frozen_config.json`;
8. `exp419_shard_plan.json`;
9. `EXP419_RESULT_REPORT_KO.md`;
10. `MANIFEST.json`.

Prediction rows retain the Exp417 label-free schema and contain no TP, FP, FN,
TN, precision, recall, F1, anomaly label/location, or human-reviewed result.
Runtime metadata is restricted to the benchmark, shard-plan, report, and
manifest artifacts.
