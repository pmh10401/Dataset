# Exp443 Task 7 — ordered checkpoint and parent-owned progress report

## Scope

Task 7 adds only the ordered checkpoint store and parent-owned progress
evidence.  It does not change scoring, routing, thresholds, publication, or
TEST evaluation.

## Behavior TDD

- RED: the two new focused suites ran `9 tests` and failed with `9` expected
  `ModuleNotFoundError` errors while `checkpoint_antigravity.py` and
  `progress_antigravity.py` were absent.
- Focused GREEN: the expanded two suites ran `10 tests` in `0.065s`, `OK`.
- Pinned full Exp443 suite (Python 3.14.6, all numerical inner-thread
  variables set to `1`):

  ```text
  Ran 104 tests in 19.184s
  OK (skipped=2)
  ```

The two skips are the existing opt-in real Policy A source/control fixtures;
Task 7 itself is fully fixture/simulation verified and performs no scoring.

## Fix round: interrupted resume and terminal ownership

Source/test fix commit: `856cea2e`.

The fix round adds a validated reopen path for parent progress files. It
restores completed/imported/recomputed/success/abstention/error counters and
the persisted elapsed-time basis, appends to the existing event/text logs,
and atomically binds the replacement process identity. Checkpoint resume also
persists a replacement PID, rejects a second active resume, and reconciles
only one exact next-record orphan left by record-create followed by
state-replace interruption. Any gap, extra orphan, changed bytes, or
record-content mismatch remains fail-closed. Heartbeats after a terminal
progress state are rejected without changing logs, progress bytes, or PID
activity.

Fix-round focused behavior suites: `14/14` passed. Pinned full Exp443 suite
(Python 3.14.6, all numerical inner-thread variables set to `1`) now reports:

```text
Ran 108 tests in 19.158s
OK (skipped=2)
```

The interrupted→resume(new PID)→finish simulation over all 57 ordered units
reported `checkpoint_completed=57`, `progress_completed=57`, terminal
`eta_seconds=null`, replacement `pid=9191`, and `pid.active=false`. The
append-only progress log contained 61 events (two begins, 57 accepts, and
two terminal events for the interrupted and completed phases).

## Fix round 2: explicit resume identity and stale state-temp recovery

Source/test fix commit: `2a0fbb85`.

Checkpoint resume now requires both a replacement
`ProcessIdentityAntigravity` and a PID probe; the former three-argument
legacy path is removed, so a resumed attempt cannot retain or silently reuse
the old PID. A second resume is rejected when the newly bound PID probes
active. When a crash leaves `state_antigravity.tmp` alongside one exact next
orphan record, resume validates the canonical current state, canonical temp
schema/bytes, prior process binding, expected prefix and status, then removes
the validated stale temp and atomically publishes the reconciled prefix under
the replacement process. Tampered, non-regular, unexpected, or mismatched
temps remain fail-closed.

Round-2 focused behavior suites: `16/16` passed. Pinned full Exp443 suite
(Python 3.14.6, all numerical inner-thread variables set to `1`) reports:

```text
Ran 110 tests in 19.894s
OK (skipped=2)
```

The explicit-process interrupted→resume(new PID)→finish simulation over all
57 ordered units again reported `checkpoint_completed=57`,
`progress_completed=57`, terminal `eta_seconds=null`, replacement
`pid=9191`, `pid.active=false`, and 61 append-only progress events.

## Ordered checkpoint evidence

`Exp443CheckpointStoreAntigravity` writes the canonical context, one
create-only record per next sequence, and an atomically replaced state.  The
state carries `record_sha256_prefix`, `completed_count`, terminal status, and
the bound process identity.  Records contain the policy/dataset sequence,
deterministic `PolicyDatasetResultAntigravity` payload (scores, candidates,
audits), timing fields, SQL-trace digest, status, and error field.

Focused behavior checks cover:

- policy-major, case-sensitive unit order and duplicate rejection;
- strict next-sequence and policy/dataset identity checks;
- create-only completion of the exact prefix and no post-completion commit;
- terminal error record with no subsequent records;
- active PID resume rejection and inactive-prefix resume;
- changed binding, changed record bytes, terminal-error, and completed-attempt
  resume rejection;
- resumable interruption state without an error record.

## Parent-owned progress evidence

`ProgressWriterAntigravity` alone writes the four attempt files:

```text
progress_antigravity.json
events_antigravity.jsonl
text_antigravity.log
pid_antigravity.json
```

The JSON schema records PID/active state, current policy/dataset, all active
datasets, TRAIN total/missing/used/excluded, TEST rows, target length,
stage/backend, completed/total, imported/recomputed/success/abstention/error,
throughput, elapsed, ETA, checkpoint, heartbeat, and latest error.  Progress
updates replace JSON atomically, append canonical event lines, and append
Korean operator-readable text.  Heartbeats sort active units by sequence so
`current_dataset` is always the lowest-sequence active unit; terminal states
set `eta_seconds` to JSON `null` and PID `active=false`.

## Simulated 57-unit replay

The progress suite creates the exact three-policy × 19-dataset shape (57
units), begins all units, accepts each small finite result in policy-major
sequence, and closes the attempt as `complete`:

```text
completed=57
recomputed=57
success=57
active=false
eta_seconds=null
events=59  (begin + 57 accepts + terminal)
pid.active=false
```

No real score source, routing function, threshold, database label, or TEST
evaluator was opened by this replay.
