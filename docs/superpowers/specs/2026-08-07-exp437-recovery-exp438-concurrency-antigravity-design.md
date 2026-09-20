# Exp437 Recovery And Exp438 Concurrent Execution Design

## 1. Decision

The user selected this operating order:

1. recover and resume Exp437 from its frozen checkpoint;
2. verify that the new Exp437 process is genuinely alive;
3. run Exp438 readiness while Exp437 continues;
4. start the official Exp438 1,117-dataset run without stopping Exp437.

This is an operational concurrency change only. It does not change either
experiment's features, score definitions, thresholds, candidate rules,
budgets, frozen lanes, label firewall, or retrospective evaluation.

## 2. Observed Exp437 Incident

The frozen Exp437 root is:

`/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807`

Read-only inspection on 2026-08-07 found:

```text
recorded PID: 81739
actual OS process: absent
recorded state: running
recorded backend: cpu
progress completed: 938 / 1,117
checkpoint record files: 939
last completed event: ShapesAll_normal_17
current recorded dataset: ShapesAll_normal_18
error count: 0
latest_error: null
publication: absent
```

The evidence is consistent with a process exit after the next checkpoint
record became durable but before the matching event/progress update. The
checkpoint remains the authority. No record, intent, event, progress, PID, or
readiness byte may be manually edited, deleted, or renamed.

## 3. Exp437 Recovery Boundary

Exp437 must resume using the exact clean source worktree and reviewed commit:

```text
worktree: /Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery
commit: cc0f507f290dc7f613763d3653498f8f54cad88a
backend: cpu
workers: 11
inner numerical threads: 1
```

The normal fixed CLI accepts only an explicit `interrupted` state. This crash
left a durable `running` lifecycle with a dead PID, so the CLI cannot reach the
already implemented stale-running recovery path. Recovery therefore invokes
the existing public `resume_exp436(...)` runtime API from the clean Exp437
worktree with `EXP437_PROFILE`; it does not patch or monkeypatch Exp437 code.

Before that invocation, a read-only preflight must prove all of the following:

- the worktree is clean and HEAD equals the reviewed commit;
- PID 81739 is still absent at the operating-system level;
- the output-directory advisory lock can be acquired by the recovery owner;
- readiness, run identity, code commit, DB, backend, checkpoint and progress
  bindings match;
- the checkpoint classification is recoverable, not failed;
- the 938-event prefix matches the first 938 checkpoint records;
- the additional record is attributable to the intent-with-record crash
  window;
- TEST label access is still zero and publication/evaluation are absent.

The recovery process claims a new PID, reconciles checkpoint authority through
the existing checkpoint/progress APIs, appends normal runtime evidence, and
continues from the next uncommitted dataset. On checkpoint completion it uses
the existing Exp437 publication path. The exact recovery command and stdout/
stderr are written to a dedicated file whose name contains `antigravity`.

If preflight or recovery classification fails, Exp437 is not mutated and
Exp438 does not start.

## 4. Exp438 Concurrent Readiness Policy

The existing Exp438 readiness accepts only terminal/inactive Exp437. Replace
that operational gate with a fixed two-case policy:

### Case A: predecessor finished

Accept Exp437 when its canonical state is `complete` or
`checkpoint_complete` and its PID is inactive.

### Case B: explicitly authorized concurrency

Accept Exp437 only when every condition below is true:

```text
run identity = exp437-metadata-roundtrip-antigravity-20260807
canonical state = running
recorded pid_active = true
actual OS PID liveness = true
backend = cpu
error count = 0
completed datasets < 1,117
heartbeat is recent
publication and evaluation are absent
concurrency authorization = frozen true
```

Running MLX/MPS Exp437, a stale/dead PID, failed state, identity mismatch,
nonzero error count, old heartbeat, or malformed progress fails before Exp438
creates its run parent.

The Exp438 readiness artifact records:

- the complete verified Exp437 execution snapshot;
- `concurrent_with_exp437=true` or false;
- predecessor PID, backend, completed/total and heartbeat;
- the fact that CPU contention can affect timing;
- exact Numba/MLX/MPS benchmark evidence and selected fallback/backend.

Exp438 still performs the mandatory acceleration assessment. Because the
official Exp438 run will operate under the same concurrent load, the paired
benchmark is treated as an in-situ measurement. CPU remains valid when no
candidate exact stage passes parity and 1.1x speed gates.

## 5. Exp438 Execution Replay

`new` and `resume` revalidate the readiness-bound predecessor snapshot without
requiring Exp437 to remain at the same completed count. The only allowed state
transitions after concurrent readiness are:

```text
running CPU with the same live PID or a verified recovery PID
running CPU -> checkpoint_complete/complete and inactive
```

The following transitions fail before opening Exp438 inputs or workers:

- running -> failed;
- running with a dead/stale PID and no terminal evidence;
- CPU -> MLX/MPS backend change;
- run identity or root change;
- counter regression;
- error count becoming nonzero;
- publication/evaluation appearing in an inconsistent state.

Exp438's official CPU worker count remains
`max(1, logical_cpu_count - 1)`, which is 11 on this Mac. Exp437 also continues
with 11 workers. Oversubscription is explicitly accepted by the user; slower
throughput is expected and is recorded in progress/ETA rather than changing
either frozen worker rule.

## 6. Monitoring And Isolation

Both processes write their own atomic progress and logs. Codex does not poll
them repeatedly.

Exp437 monitoring remains under:

```text
/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/output_antigravity/progress_antigravity.json
/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/output_antigravity/events_antigravity.jsonl
/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/output_antigravity/parent_antigravity.log
/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/output_antigravity/run_antigravity.pid
```

Exp438 monitoring remains under:

```text
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/progress_antigravity.json
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/events_antigravity.jsonl
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/parent_antigravity.log
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/run_antigravity.pid
```

No Exp438 code writes into Exp437. The only Exp437 access by Exp438 readiness is
canonical read-only process-state verification. Post-freeze Exp438 evaluation
still waits for both independent publications.

## 7. Tests

Implementation follows TDD. New tests must first fail and then prove:

- active, fresh, zero-error CPU Exp437 is accepted and fully recorded;
- terminal/inactive Exp437 remains accepted;
- stale PID, dead PID, old heartbeat, MLX/MPS backend, failed state, nonzero
  errors and wrong identity are rejected before Exp438 root creation;
- CLI readiness replay permits same-run forward progress and a valid terminal
  transition;
- CLI replay rejects regression, stale-running, failure and identity/backend
  change before input or worker access;
- prediction TEST-label access remains zero;
- Exp438 scientific config, channels, thresholds and budgets are byte-equal to
  the already frozen values;
- code map, focused Exp438 tests, Exp436 shared tests and Exp425 predecessor
  tests still pass.

For Exp437 recovery, the preflight output records the exact clean commit,
dead original PID, checkpoint/event counts, reconciliation reason and frozen
hashes before the recovery owner is launched.

## 8. Stop Conditions

Stop without starting Exp438 if:

- Exp437 cannot be recovered through the existing frozen runtime API;
- Exp437 recovery immediately fails or does not write a fresh live PID;
- Exp438 source/input/hash/readiness tests fail;
- the DB or frozen lane digest changes;
- TEST label access is nonzero;
- an Exp438 fixed root appears before readiness owns it.

After both processes are alive, report their PIDs and direct log paths. Do not
perform recurring Codex monitoring; the user will report completion.
