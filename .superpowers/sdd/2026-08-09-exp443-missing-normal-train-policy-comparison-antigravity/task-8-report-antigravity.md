# Exp443 Task 8 — readiness, ordered 11-worker runtime, and parity

## Scope

Task 8 connects the existing label-free Policy A/B/C scorer to the Task 7
ordered checkpoint and parent-owned progress files.  It keeps the frozen
preprocessing, official aeon MiniROCKET calls, candidate thresholds, routing
algebra, seeds, and TEST-label firewall unchanged.  MLX/MPS remain
inapplicable because the official transformer is CPU code.

## Behavior TDD

- RED: the three new focused suites initially ran eight tests and failed with
  the expected missing-module errors for `readiness_antigravity.py` and
  `runtime_antigravity.py`.
- GREEN: the focused readiness/runtime/parity suites now cover 12 tests,
  including exact 57-unit/19-Policy-A shape, clean-head/output-root and
  label-free readiness, six one-thread variables, exact 11-worker SpawnContext,
  ordered-prefix buffering, terminal worker error, Policy-A control gating,
  resumable checkpoint results, SIGTERM-matched progress/checkpoint state, and
  semantic parity/speed evidence.  The pinned run is `12/12 OK`.
- Full Exp443 regression before final parity is `122 tests`, `OK (skipped=2)`
  under Python 3.14.6 with all six numerical thread variables set to `1`.

## Readiness authority

`readiness_antigravity.py` writes one canonical `exp443.readiness.v1` record
only after clean reviewed HEAD, frozen design/config/lock identity, source
hashes and 1,117-dataset/19-affected inventory, routing algebra
`(2116,137,35,2218)`, zero label access, and the pinned 12-logical-CPU/11-worker
environment have all passed.  The module has no scorer, routing, evaluator, or
TEST-label import path; fixture verification confirms those imports are absent.

## Runtime behavior

`runtime_antigravity.py` installs OMP/OpenBLAS/MKL/VECLIB/NUMEXPR/NUMBA=`1`
before numerical imports and starts exactly 11 spawned workers.  Parent code
alone owns Task 7 checkpoint/progress writes.  Out-of-order worker results are
buffered and only the next policy-major, case-sensitive dataset prefix is
committed.  Policy A must pass the existing Task 6 control gate before Policy B
or C is submitted.  Source/config/environment identities are revalidated before
scheduling, after A, before B/C, and before terminal completion.  SIGTERM and
KeyboardInterrupt leave matching `interrupted_resumable` checkpoint/progress
state; worker/runtime errors write one ordered terminal error and close progress
as `error`.

`Exp443CheckpointStoreAntigravity.results_antigravity()` reopens only the
durable scored prefix for control-gate and resume replay; error records never
become score results.

## Real verification

One affected-dataset smoke passed for `DodgerLoopDay_normal_1`:

```text
Policy A status=scored, train_used=19, TEST rows=50, kernel-seed scores=4
pinned database label_access_count=0, source snapshot label_access_count=0
```

The real 19-unit Policy A sequential-versus-11-worker parity was launched as a
separate process with parent PID/log evidence.  The process prints each
dataset/completed count/elapsed/ETA for both stages and writes
`scratch/exp443_worker_parity_report_antigravity.json`; its final semantic,
candidate, control-evidence, and routed-prediction-byte result is recorded here
after the process completes.

## Boundary

No evaluator, TEST labels, publication security, object-tamper audit, or
scientific-rule change was added.  Speedup is recorded as informative evidence
only and cannot change the frozen CPU algorithm.
