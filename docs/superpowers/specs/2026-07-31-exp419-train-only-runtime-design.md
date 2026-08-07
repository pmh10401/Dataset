# Exp419 TRAIN-only Runtime Boundary Design

## Purpose

Repair Exp419's execution plumbing so that its non-official runtime
benchmark is genuinely TRAIN-only and the official worker runtime does not
repeat whole-population context discovery. This is a runtime-only correction:
the Exp417 model, preprocessing, horizons, seeds, early stopping, OOF
calibration, scoring, threshold, qualification, lanes, and publication schema
remain unchanged.

## Problem

The first Exp419 runtime benchmark inherited `freeze_run_context`. That
operation builds an identity-and-length manifest for every TEST series before
the benchmark can start. It is unnecessary for the two-batch normal-TRAIN
smoke, violates its declared boundary, and makes the benchmark spend most of
its time on unrelated SQLite verification and metadata work. Each spawned
worker also independently rebuilt the same frozen context.

## Design

### 1. Benchmark-only train context

Introduce an immutable Exp419 `BenchmarkTrainContext` containing only the
pinned database SHA-256, lexical frozen population, lane snapshot identity,
reviewed Git commit, runtime versions, and frozen configuration. It must not
contain a TEST identity, TEST length, TEST value, label, score, candidate, or
evaluation field.

`freeze_benchmark_inputs` builds this context from the normal-TRAIN lane
snapshot and population query. The 19-dataset selection proof and the
two-batch worker schedule remain exactly as preregistered.

### 2. Worker bootstrap boundaries

Use a dedicated immutable `BenchmarkTrainBootstrap` for bounded smoke workers.
It carries the already-frozen train context and its canonical digest. A smoke
worker validates that digest, verifies its assigned shard is in the frozen
population, and never invokes the Exp417 full run-context freezer.

The official `WorkerBootstrap` continues to carry an exact Exp417
`FrozenRunContext`. The parent freezes and validates it once, then workers
validate the passed context and provenance rather than rebuilding it. A worker
never shares a SQLite connection, model object, random generator, or
checkpoint store with another process.

### 3. Per-worker verified read-only database access

Add a process-local verified read-only database session for each worker. The
worker verifies the pinned database descriptor and SHA-256 once before it
handles its shard, then uses the same read-only connection for that worker's
existing TRAIN and TEST-value queries. Existing SQL, decoding, preprocessing,
model fitting, scores, candidate indices, and checkpoint payloads are not
changed. Descriptor identity is checked again when the session closes; any
mismatch fails the worker and prevents publication.

The bounded smoke uses only the TRAIN query through this session. It does not
invoke a TEST query or construct a TEST identity manifest.

### 4. Provenance and fail-closed rules

Both context kinds have canonical payloads and SHA-256 digests. Runtime
measurement evidence records the benchmark train-context digest. Official
checkpoint provenance remains bound to the exact Exp417 full context.

Any type mismatch, digest mismatch, population/shard mismatch, database
identity mismatch, non-read-only connection, worker failure, or context
reconstruction attempt is fail-closed. Old partial Exp417 checkpoints and the
terminated Exp419 diagnostic root remain untouched and cannot be reused.

### 5. Local progress log

Each benchmark and official run creates a local, append-only runtime log next
to its checkpoint status sidecar. It records the run identifier, parent and
worker PIDs, shard assignment, current dataset, phase, seed/fold when known,
completed count, elapsed time, and a one-line safe error. The same information
continues to be atomically available in each worker's JSON status snapshot.

Logs must never contain raw series, feature values, residuals, scores,
predictions, TEST identities or lengths, labels, anomaly locations, confusion
counts, or performance metrics. They are observation-only: no polling loop,
monitor, keepalive, scheduling, resume, publication, or model decision may
read them.

## Data and policy boundaries

- Benchmark selection and smoke access normal TRAIN metadata and values only.
- The benchmark never accesses TEST values, TEST metadata, labels, anomaly
  locations, candidates, scores, qualification results, or metrics.
- The official prediction path may access raw TEST values only after its full
  TRAIN contract has been frozen, exactly as Exp417 does; it never accesses
  TEST labels or evaluation data.
- No result may be used to retune alpha, thresholds, source, budget, lane, or
  worker count after the benchmark is frozen.
- MLX and MPS remain excluded. Numba remains limited to the existing audited
  deterministic score aggregation path. Each CNN fit and BLAS call remains
  single-threaded; outer/inner layout stays `1→1`, `2→3`, `4→3`, `6→2`.

## Verification

1. A failing bounded-smoke test proves that calling the full Exp417 context
   freezer is rejected; the corrected smoke runs against a train-only fixture
   and reports `test_values_loaded == false`.
2. A failing test proves a passed official context is used directly; an
   attempted worker reconstruction is rejected.
3. A parity test runs the old per-query verifier and the new per-worker session
   on the same synthetic SQLite fixture. It requires byte-identical decoded
   TRAIN and TEST values, identical preprocessing/model hashes, OOF residuals,
   candidate indices, and checkpoint digests.
4. A mutation test changes the frozen database bytes or descriptor metadata and
   requires the session to fail before a result is accepted.
5. Update the Exp419 preregistration, runbook, source manifest, and all
   Exp419/Exp417/Exp414/Exp416 regression suites. Run the corrected 19-dataset
   bounded benchmark before asking for a fresh official 1,117-dataset approval.
6. A log-schema test proves normal progress and failure records contain only
   the safe fields above, and rejects raw/test/prediction/label/metric fields.

## Non-goals

This design does not introduce a new score source, use GPU execution, change
scientific outputs, alter lane routing, perform label-assisted selection, or
publish private artifacts.
