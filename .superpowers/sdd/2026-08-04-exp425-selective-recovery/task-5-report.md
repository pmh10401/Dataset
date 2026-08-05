# Task 5 — Selective MPS Runtime and Explicit Resume

Status: partial follow-up implementation; the official population and readiness
gate were not started.

## Follow-up hardening — 2026-08-04

This follow-up adds owner-bound atomic PID leases; inactive complete/failed
resume accepts an absent lease, while a live or foreign/malformed lease is never
unlinked.  Complete resume is derived only from descriptor-safe checkpoint
inspection and does not claim a PID or call mutating checkpoint open/recovery.
The inspection now validates each completed record digest and exposes compact
record summaries for exact JSONL reconciliation.  Resume preflight binds the
full provenance/context digest before the lease claim, and permits only a
checkpoint-commit-to-event exact suffix.

Native session authority-critical snapshot/audit/writer/runner bindings are
read-only after registration and are rechecked immediately before capability
minting.  Cleanup releases broker, MPS guard, descriptor, and snapshot despite
an earlier cleanup failure.  A final read-only inventory validation and snapshot
close now occur before `complete_review_only` is published.  Native TRAIN-only
abstentions after model qualification but with no qualified group accept absent
TEST scoring and persist `not_applicable_no_qualified_group`.

Round 2 adds serialized writer ownership and a periodic heartbeat pump.  The
pump writes canonical atomic progress, JSONL, and parent-log evidence during a
blocked dataset stage, preserves the active dataset/phase, and joins on exit.
Every emitted event now includes elapsed/cumulative timing, backend,
checkpoint-root, and monotonic heartbeat sequence.  TRAIN load updates value
and byte totals before preprocessing.  The CLI rejects an already-existing new
root before importing a launch factory and closes a partially returned snapshot
on later launch-shape errors.

Focused verification after this change:

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime \
  experiments.exp425_selective_recovery.tests.test_checkpoints \
  experiments.exp425_selective_recovery.tests.test_cli -v
PASS — 54 tests in 6.038s

python3 -m unittest experiments.exp425_selective_recovery.tests.test_capabilities \
  experiments.exp425_selective_recovery.tests.test_checkpoints \
  experiments.exp425_selective_recovery.tests.test_predecessor \
  experiments.exp425_selective_recovery.tests.test_database \
  experiments.exp425_selective_recovery.tests.test_eligibility -v
PASS — 95 tests in 9.814s; 256MiB snapshot=0.139s, verified_opens=2.636s

python3 -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_runner \
  experiments.exp417_causal_multihorizon_shadow.tests.test_model \
  experiments.exp417_causal_multihorizon_shadow.tests.test_scoring \
  experiments.exp417_causal_multihorizon_shadow.tests.test_qualification -q
PASS (exit 0)
```

This does **not** yet satisfy the commitment review in full: no real mixed
Task2/3 importer→runtime orchestration test, heartbeat implementation/test,
complete CLI matrix, full nested evidence mutation matrix, or preregistered
eligible MPS parity smoke was added.  The missing MPS smoke remains unavailable
without a selected frozen eligible fixture/launch assembly; no TEST labels or
official population was queried.

Actual environment check for the requested MPS smoke:

```text
torch=2.13.0, mps_built=true, mps_available=true
```

The external blocker is not device availability: the Task 5 worktree has no
`experiments/exp424_*` implementation or frozen Exp424 launch/fixture assembly
to execute as the required exact reference path.  A synthetic torch-only MPS
comparison would not meet the requested scientific parity contract, so it was
not substituted.

## TDD evidence

The initial focused command was intentionally red because neither owned module
existed:

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime \
  experiments.exp425_selective_recovery.tests.test_cli -v
ModuleNotFoundError: No module named 'experiments.exp425_selective_recovery.runtime'
```

The implemented coverage currently exercises atomic canonical progress/JSONL
writes, parent-log and PID creation, root lifecycle rejection for `new` and
`resume`, immutable inspection, and duplicate terminal-event rejection.

## MLX / MPS assessment

Exp425 adds no score source and keeps the frozen Exp424 MPS execution unchanged.
Its new runtime operations are descriptor-bound SQLite reads plus checkpoint and
JSON validation, so no exact MLX implementation can be substituted without a
new parity and timing qualification.  MLX is not adopted; the runtime state
records `adopted=false` and the parity/speed-gate fallback reason.  Native work
requires `torch.device("mps")`, a passing non-fallback MPS audit, and an
exclusive single-heavy-process lock.  CPU fallback is rejected.

## Verification to date

## Architecture A pre-data failure and fix — 2026-08-04

The first Task 5 root `exp425_task5_mps_parity_20260804_architecture_a` failed
only in immutable golden decoding before DB snapshot/data/MPS work. PID 32415;
TRAIN=0, TEST values=0, TEST labels=0. Preserved SHA256s: parent.log
`35686f22ba4875bb6e3e6deab8209675c7d36064c4bad22b0cd778dc68a74ef8`,
events `981b231d429ec20c542701aff30fa601ef14304566c13c88312e4db4fd29af78`,
and progress `0523b34d67e338b67c8b86236549c18d18997d22465e53e8bb22a43099dee2b1`.
The corrected harness decodes the reconstructed `DatasetResult` rather than
treating it as raw JSON; it is prepared but not run at the frozen retry root.

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime \
  experiments.exp425_selective_recovery.tests.test_cli \
  experiments.exp425_selective_recovery.tests.test_eligibility \
  experiments.exp425_selective_recovery.tests.test_capabilities \
  experiments.exp425_selective_recovery.tests.test_checkpoints \
  experiments.exp425_selective_recovery.tests.test_predecessor \
  experiments.exp425_selective_recovery.tests.test_database -v
PASS — 97 tests in 11.247s

python3 -m py_compile experiments/exp425_selective_recovery/runtime.py \
  scratch/run_exp425_selective_recovery.py
PASS

git diff --check
PASS
```

The requested TRAIN-only MPS parity smoke has not been run: this worktree has
no configured frozen launch assembly or verified source dataset selected for
that smoke, and running it without those arguments would violate the frozen
contract.  No Exp425 1,117-dataset population, readiness command, or TEST-label
query was started.

## Follow-up verification

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime \
  experiments.exp425_selective_recovery.tests.test_cli \
  experiments.exp417_causal_multihorizon_shadow.tests.test_runner \
  experiments.exp417_causal_multihorizon_shadow.tests.test_model \
  experiments.exp417_causal_multihorizon_shadow.tests.test_scoring \
  experiments.exp417_causal_multihorizon_shadow.tests.test_qualification -v
PASS — focused runtime/CLI plus frozen Exp417 runner/model/scoring/qualification regression.

python3 -m py_compile experiments/exp425_selective_recovery/*.py \
  scratch/run_exp425_selective_recovery.py
PASS

git diff --check
PASS
```

## Completion update — 2026-08-04

Task 5 completion work hardened the runtime's missing-record path without
altering Exp417/Exp424 algorithm code.  A supplied TRAIN payload is
preprocessed once and assessed once.  If all remaining work is identity
insufficient (including a mix of preprocessing abstentions and newly
identity-insufficient groups), the runtime builds a context-bound empty
`DatasetDecisionEvidence` and an `abstained` `DatasetResult` from unchanged
qualification/freeze logic.  It does not call `run_dataset`, model training,
scoring, or the TEST loader.  Native envelopes bind the processed-input
identity, complete captured OOF/final model hashes, canonical float64 OOF
calibration bytes, TEST-scoring final-model hashes, and the backend audit;
the no-model path instead records explicit `not_applicable` model/scoring
evidence.

Progress now has the exact phase sequence, atomic temp cleanup, canonical
JSONL validation, terminal-event uniqueness, non-overlapping accounting
(`imported + native == completed`, `recomputed <= native`), an explicit
OOF-identity-insufficient counter, and resume validation before PID mutation.
Committed-but-not-yet-logged terminal records are logged once on resume;
already logged records are never replayed.  Inspection validates progress and
events without opening a database/checkpoint writer or changing output state.

Commands actually run:

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime \
  experiments.exp425_selective_recovery.tests.test_cli -v
PASS — 9 tests

python3 -m unittest experiments.exp425_selective_recovery.tests.test_capabilities \
  experiments.exp425_selective_recovery.tests.test_checkpoints \
  experiments.exp425_selective_recovery.tests.test_predecessor \
  experiments.exp425_selective_recovery.tests.test_database \
  experiments.exp425_selective_recovery.tests.test_eligibility -v
PASS — 95 tests (database benchmark: 256MiB snapshot=0.127s, verified_opens=2.489s)

python3 -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_runner \
  experiments.exp417_causal_multihorizon_shadow.tests.test_model \
  experiments.exp417_causal_multihorizon_shadow.tests.test_scoring \
  experiments.exp417_causal_multihorizon_shadow.tests.test_qualification -v
PASS
```

The required TRAIN-only Exp424-versus-Exp425 MPS parity smoke remains
intentionally unrun: Task 5 has no verified launch assembly or selected
pre-registered eligible source fixture, and inventing either would violate the
frozen contract.  No readiness task, official population run, or TEST-label
query was performed.

## Mixed-store/session integration — 2026-08-04

Task 5 plan step 1 now has a deterministic three-dataset integration fixture.
It uses a real `FrozenRunContext` with a test-only replaced small config, a
real `Exp425CheckpointStore`, an actual source-authorized
begin/stage/promote import, real `AtomicProgressWriter`, and a real opaque
`Exp425RuntimeSession`.  Only Task4 snapshot method bindings, MPS availability,
the exclusive guard FD, and the private expensive native-compute seam are
replaced.  The compact native factory makes the two native envelopes
deterministic; a separate routing test uses real `EligibilityAssessment` and
`GroupEligibilityEvidence` objects to prove the all-insufficient path reaches
TRAIN preprocessing/eligibility and never enters the downstream runner.

The mixed run asserts: ordered three-record coverage; imported=1, native=2,
recomputed=2, abstained=1, OOF-insufficient=1, failed=0; imported work never
reads TRAIN/TEST or invokes model/scoring; both valid imported/native candidates
pass the compact global gate; and progress/events/heartbeats are canonical,
label-free, terminal, and closed with the PID lease removed.  Captured native
broker claims bind run, dataset, context, trace, record, backend, and the live
session process nonce.  Modified records, every store-consumed wrong claim, and
reused claims fail before a native intent is created.

The session-minted native capability test injects Task2's existing
`intent_durable`, `record_durable`, `progress_durable`, and `intent_removed`
failpoints.  Each boundary is reopened once and resumed once: the intent-only
case recomputes once; later durable-record cases do not recompute; all finish
with exactly one record, terminal event, and complete counter.

Verification:

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime -v
PASS — 15 tests in 0.178s

python3 -m unittest experiments.exp425_selective_recovery.tests.test_capabilities \
  experiments.exp425_selective_recovery.tests.test_checkpoints -v
PASS — 47 tests in 3.988s

python3 -m compileall -q experiments/exp425_selective_recovery && git diff --check
PASS
```

No neural training, MLX decision, MPS/CPU parity claim, official population,
or readiness/Task6 command was executed.

## Global workload compact-adapter parity — 2026-08-04

The compact global adapter now validates a complete, sorted one-record-per-
context-dataset inventory before counting any coordinate.  It accepts only the
effective statuses `scored` and `abstained`; terminal `failed` is a final-gate
error.  Candidate coordinates are canonical `[dataset, index]` lists, sorted
and unique, exact-dataset, non-bool integers, in the frozen TEST identity
manifest, and outside every frozen lane.  Abstentions must be empty.

Native records require the exact compact effective-result schema and its
result-evidence/source bindings; imported records require the exact imported
schema and a source-result digest matching the envelope.  The adapter obtains
the cap from the same `qualification.FROZEN_CONFIG` module used by Exp417 and
rejects a context whose cap differs.  Its count uses `sum(context.test_sizes)`,
after requiring it to equal the frozen identity-manifest total, and Exp417's
unchanged strict `>` comparison.  It returns `complete` or
`global_workload_gate_failed`; the runtime converts the latter to its existing
failed, non-review-only-publication terminal state.  Reconstructing full
decision evidence is unnecessary because the validated coordinates are the
only input to Exp417's global count; real synthetic `DatasetResult` populations
exercise the equivalence directly.

Coverage includes 3/4, 4/4, and 5/4 candidate boundaries (below, exact, and
above cap), global-failure runtime status, terminal failed record, duplicate and
unsorted coordinates, wrong dataset, bool/out-of-manifest index, lane overlap,
candidate-bearing abstention, unknown status/key, missing/wrong native or
imported source fields, incomplete/reordered coverage, store-universe total
mismatch, and context-cap mismatch.

Verification:

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime -v
PASS — 20 tests in 0.195s

python3 -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_qualification -v
PASS — 22 tests in 11.125s

python3 -m compileall -q experiments/exp425_selective_recovery && git diff --check
PASS
```

## Task 5 CLI/PID/inspect/lifecycle matrix — 2026-08-04

STATUS: complete
OBJECTIVE: Complete the bounded `new|resume|inspect` fail-closed CLI lifecycle matrix without changing the runtime implementation.
CHANGES:
- `scratch/run_exp425_selective_recovery.py`: added a read-only output/checkpoint pair validator that cross-binds run identity, all published absolute paths, dataset universe, and reconciled terminal evidence. `inspect` now returns 2 for malformed/mismatched resource contracts, 1 only for validated incomplete/failed state, and 0 only for matching complete review-only state. `resume` now performs the same read-only validation before launch assembly and returns 0 for a matching complete run without a launch factory, PID claim, snapshot, DB, or model work. Launch-path disclosure remains absolute and now shares one construction path.
- `experiments/exp425_selective_recovery/tests/test_cli.py`: expanded from two subprocess checks to an 11-test matrix covering parser misuse/unknown mode; existing new roots; absent/mismatched resume roots; run-identity mismatch; incomplete/failed/complete inspect codes; complete-resume idempotence; live/foreign PID rejection; path disclosure; and inspect forbidden-call/tree byte-inode-mtime immutability. Fixtures exercise real checkpoint inspection with no DB/model/MPS activity.
VERIFIED:
```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_cli -v
PASS — 11 tests in 9.318s

python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime experiments.exp425_selective_recovery.tests.test_cli -v
PASS — 31 tests in 7.973s

python3 -m py_compile scratch/run_exp425_selective_recovery.py experiments/exp425_selective_recovery/tests/test_cli.py && git diff --check
PASS — exit 0; no output

git diff 53f2e70 -- scratch/run_exp425_selective_recovery.py experiments/exp425_selective_recovery/tests/test_cli.py && git status --short
INSPECTED — only the two owned implementation/test files were modified before append-only evidence updates; no output, checkpoint, or readiness artifacts were created.
```
JUDGMENT CALLS: Malformed or cross-root-mismatched inspect inputs are resource-contract failures and therefore exit 2; validated `initializing` and `failed` states return 1. A complete resume is determined solely from the matching durable output/checkpoint pair, so it does not need a launch factory.
GAPS: none

## Task 5 CLI fix round 1/5 — 2026-08-04

STATUS: complete
OBJECTIVE: Fail closed on launch-assembly errors and validate inspect progress accounting before selecting an exit code.
CHANGES:
- `scratch/run_exp425_selective_recovery.py`: separates factory specification, import, invocation, mapping, and required-key assembly failures (all exit 2) from failures after a valid launch is passed to `execute_runtime_command` (exit 1). Adds a read-only progress validator for schema, state/stage, command, identity, non-negative exact counters, population conservation, checkpoint-summary reconciliation, and complete-review-only zero-remaining/zero-failed/no-current-work invariants.
- `experiments/exp425_selective_recovery/tests/test_cli.py`: adds malformed/non-callable/raising/non-mapping factory coverage plus immutable adversarial inspect cases for unknown state, each named complete counter, and impossible incomplete accounting.
VERIFIED:
```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_cli -v
PASS — 15 tests in 7.762s

python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime experiments.exp425_selective_recovery.tests.test_cli -v
PASS — 35 tests in 7.992s

python3 -m py_compile scratch/run_exp425_selective_recovery.py experiments/exp425_selective_recovery/tests/test_cli.py && git diff --check
PASS — exit 0; no output
```
JUDGMENT CALLS: Factory invocation is resource assembly rather than a validated runtime execution, so it returns 2. The CLI derives every persisted counter directly from the read-only checkpoint summaries; any difference is a contract error.
GAPS: none

## Task 5 CLI fix round 1 continuation — 2026-08-04

STATUS: complete
OBJECTIVE: Preserve legitimate nonterminal checkpoint-commit-to-event/progress crash recovery while retaining fail-closed inspection.
CHANGES:
- `scratch/run_exp425_selective_recovery.py`: reads the canonical terminal-event dataset prefix without writes after existing preflight has validated its detailed correspondence to the checkpoint. Nonterminal counters now reconcile to that event prefix, allowing only the preflight-authorized checkpoint suffix; complete-review-only still requires full event/checkpoint equality and exact terminal invariants.
- `experiments/exp425_selective_recovery/tests/test_cli.py`: adds the real durable-checkpoint-ahead-of-progress fixture. Inspect returns validated incomplete (1) without tree mutation, and resume reaches the mocked factory/execution boundary. Counter-prefix and reordered-event inverse cases return 2 and leave the tree unchanged.
VERIFIED:
```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_cli -v
PASS — 18 tests in 7.922s

python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime experiments.exp425_selective_recovery.tests.test_cli -v
PASS — 38 tests in 8.135s

python3 -m py_compile scratch/run_exp425_selective_recovery.py experiments/exp425_selective_recovery/tests/test_cli.py && git diff --check
PASS — exit 0; no output
```
JUDGMENT CALLS: Existing read-only preflight remains the authority for matching terminal-event origin/status/reason details to checkpoint records; the CLI adds only the progress-counter-to-event-prefix check needed to admit the authorized durable suffix.
GAPS: none

## Task 5 MPS smoke lifecycle and sparse-lane preflight — 2026-08-05

A fresh Sol/High review found that the nominal smoke path published success before
closing the session-owned non-idempotent database snapshot and runtime PID writer.
Commit `5b5e160` moved both cleanups before result/terminal-success publication,
retains the first cleanup exception, attempts both releases, and publishes cleanup
failure rather than success. The new lifecycle regression was observed RED on the old
flow (`snapshot.closed == false` at result publication followed by `snapshot closed
twice`) and GREEN on the corrected flow. Focused runtime/CLI/parity tests passed 61,
and the full Exp425 suite passed 171. A fresh Sol/High review returned `ship`.

The authorized retry root then failed closed at PID 79638 during
`Exp425DatabaseSnapshot.freeze_context`, before TRAIN, TEST values, TEST labels, MPS
models, candidates, checkpoint creation, or prediction. Preserved SHA256s are:

- progress `053a1553a8facaa75a8033c952968e7f905bec379ae55dab1bedf094e2cd17f5`;
- events `f6f42551970d255f0733191346320a1a494b4b209453df9f7355011db36facbc`;
- parent `1212d8b6f6ea3f6b47a11d2dec15409d7bad740dbb6343014a31fcca5be48d7d`;
- PID evidence `8b2aa31b7f96b2c561ca99e3994a94bda4b1b4385ede74b19aa59911cfd14770`.

Root-cause enumeration against the immutable production DB and pinned Exp291/292
artifacts found 1,117/1,117 datasets rejected, 195,900 TEST identities, 3,090 routed
lane coordinates, 192,810 intentionally unrouted identities, and zero lane
coordinates outside the DB identity sets. Exp417 uses these lanes as sparse exclusion
sets; Task 4's equality check was therefore overstrict. A new test with two TEST
identities and one routed coordinate per dataset was observed RED on equality and
GREEN after changing only the DB-bound rule to subset containment. The existing
out-of-manifest coordinate rejection remains covered.

The actual production context smoke now reports 1,117 datasets, 195,900 TEST
identities, 3,090 lane coordinates, `full_hash_count=1`, lifecycle `closed`, and owned
link cleanup `removed`. No output/checkpoint root or TEST value/label was opened.
The exact retry2 root is frozen in
`task-5-mps-parity-antigravity-retry2-addendum.md`. It was executed only after full
verification and a fresh Sol/High `ship` review.

## Task 5 retry2 serialization and failure-lifecycle correction — 2026-08-05

Retry2 PID 80796 completed the one-fixture MPS TRAIN computation in 120.025 seconds
and emitted four periodic heartbeats. It then failed before parity comparison,
dataset-record commit, result publication, or success publication because the
allowlisted semantic digest could not serialize nested immutable `mappingproxy`
containers. The exception path also asked the session to close a database snapshot
that the surrounding snapshot context had already closed. Loader counts remained
exactly TRAIN 1, TEST values 0, TEST labels 0; checkpoint completed count remained
zero. No parity outcome or evaluation feedback was observed.

Both defects were reproduced before correction. The immutable-container test raised
the same JSON `TypeError`; the failure-lifecycle test observed snapshot close counts
`(session, snapshot, runtime) = (1, 2, 1)`. The corrected comparison recursively
normalizes only mapping/list/tuple representation while retaining every allowlisted
key and value. The smoke now closes session/runtime owners inside the snapshot scope
on both success and failure. Focused tests are GREEN with close counts `(1, 1, 1)`.

The preserved retry2 hashes and exact retry3 root are frozen in
`task-5-mps-parity-antigravity-retry3-addendum.md`. No frozen scientific setting,
model hash, score source, threshold, alpha, seed, fixture, tolerance, or routing rule
changed. Retry3 remains TRAIN-only and may run once only after complete verification
and fresh Sol/High review.

Pre-review verification:

```text
python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py'
PASS — 174 tests in 17.285s; 256MiB snapshot 0.123s vs verified opens 2.318s

python3 -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_inputs experiments.exp417_causal_multihorizon_shadow.tests.test_contracts experiments.exp417_causal_multihorizon_shadow.tests.test_runner experiments.exp417_causal_multihorizon_shadow.tests.test_reporting
PASS — 83 tests in 217.695s

python3 -m py_compile experiments/exp425_selective_recovery/mps_parity.py scratch/run_exp425_task5_mps_parity.py experiments/exp425_selective_recovery/tests/test_mps_parity.py
PASS
```

## Task 5 retry3 prelaunch review fix — 2026-08-05

The fresh Sol/High review of the first retry3 preparation returned `fix-first` with
two High findings. First, comparison normalized immutable containers only in temporary
copies, while the final result writer still received the raw `mappingproxy` payload
and would fail after checkpoint commit. Second, a session returned by the runtime
factory could fail loader-binding validation before assignment into `main`, leaking
that session and allowing the snapshot context to unwind before runtime-writer
cleanup.

Both consumer-visible regressions were added first and observed RED:

- the real `SmokeEvidenceWriter.result` rejected a nested immutable payload with the
  same `mappingproxy is not JSON serializable` failure;
- the guard-mismatch lifecycle reported close counts `(session, snapshot, runtime) =
  (0, 1, 1)` and ordered snapshot-context exit before runtime-writer cleanup.

The bounded correction applies the already-tested representation-only recursive
normalizer at the final result-publication boundary. On guard mismatch, the helper
closes its not-yet-transferred session and `main` closes the runtime writer inside the
snapshot context while retaining the validation exception. The two tests are GREEN;
the exact failure ordering is now `session.close`, `snapshot.close`, `runtime.close`,
`snapshot_context.exit`, `writer.failure` with close counts `(1, 1, 1)`.

Fresh verification after this correction:

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_mps_parity
PASS — 21 tests in 0.493s

python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py'
PASS — 176 tests in 17.577s; 256MiB snapshot 0.123s vs verified opens 2.327s

python3 -m unittest experiments.exp417_causal_multihorizon_shadow.tests.test_inputs experiments.exp417_causal_multihorizon_shadow.tests.test_contracts experiments.exp417_causal_multihorizon_shadow.tests.test_runner experiments.exp417_causal_multihorizon_shadow.tests.test_reporting
PASS — 83 tests in 232.490s

python3 -m py_compile experiments/exp425_selective_recovery/mps_parity.py scratch/run_exp425_task5_mps_parity.py experiments/exp425_selective_recovery/tests/test_mps_parity.py
PASS
```

Retry3 remains absent and unconsumed. Retry2 progress/events/parent hashes remain
exactly those frozen in the retry3 addendum. A new clean Sol/High verdict is required
before the one authorized retry3 invocation.
