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
- GREEN: the focused readiness/runtime/parity/progress/contract suites now cover 22 tests,
  including exact 57-unit/19-Policy-A shape, clean-head/output-root and
  label-free readiness, six one-thread variables, exact 11-worker SpawnContext,
  ordered-prefix buffering, terminal worker error, Policy-A control gating,
  resumable checkpoint results, SIGTERM-matched progress/checkpoint state, and
  semantic parity/speed evidence, resumable pin reuse, finalization-only
  re-entry, idempotent prediction bytes, and spawn-picklable worker results.
  The pinned run is `22/22 OK`.
- Full Exp443 regression after the final parity is `127 tests`,
  `OK (skipped=2)` under Python 3.14.6 with all six numerical thread
  variables set to `1`.

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

The real 19-unit Policy A sequential-versus-11-worker parity completed on clean
reviewed commit `3da75e6e14b007b377499805d113f4719f4dc58b`.  The parent-owned
foreground process was Python PID `49230` (wrapper PID `49228`) and wrote
`/tmp/exp443_task8_worker_parity_antigravity_final.log`; each stage logged
dataset, completed/total, elapsed seconds, and ETA.  The compact report is
`scratch/exp443_worker_parity_report_antigravity.json`:

- `semantic_equal=true` for all 19 Policy A units; sequential and parallel
  semantic payload SHA256 are both
  `16ae73ac00c63e5dbd3adcf39bd3a1d5d580fd17db2a3f79c7ea7b4da3033552`.
- Sequential elapsed was `49.7410828751` seconds and parallel elapsed was
  `7.6217000410` seconds (`speed_ratio=6.5262451431`, informative only).
- The Task 6 control gate passed with max absolute score difference
  `4.5397329895280336e-08`; routed prediction bytes SHA256 is
  `9766c1939fd7c45473ccd35230ca489e2bedc35dabc58080eb26582e4e51e10a`.
- `workers=11`, `logical_cpu_count=12`, `inner_threads=1`, and
  `label_access_count=0`.  Candidate identities, control evidence, and routed
  prediction bytes were compared semantically and exactly; timing/PID fields
  were excluded only from the parity digest.

The finalization seam is resumable after all 57 checkpoint records: an explicit
complete-checkpoint finalization resume reuses the retained database pin, and
prediction artifacts are idempotent only when existing bytes exactly match the
canonical recomputation.  A changed or symlinked artifact fails closed.  The
parent progress writer advances to the next dataset and publishes its
dimensions in one atomic JSON replacement, avoiding a dataset/dimensions
mismatch window.

## Boundary

No evaluator, TEST labels, publication security, object-tamper audit, or
scientific-rule change was added.  Speedup is recorded as informative evidence
only and cannot change the frozen CPU algorithm.
