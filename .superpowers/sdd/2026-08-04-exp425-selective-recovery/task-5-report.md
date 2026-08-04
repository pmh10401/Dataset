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
