# Task 4 — One-Hash Database Snapshot

Implemented `experiments/exp425_selective_recovery/database.py` and its public
exports. The snapshot opens a no-follow regular database descriptor with its
parent descriptor, rejects SQLite sidecars, hashes the accepted descriptor once,
and retains one `mode=ro&immutable=1`, `query_only` connection. It uses the
fixed Exp417 TEST identity SQL and the Exp417 connection-only raw TRAIN/TEST
decoders; TEST label access is never introduced.

On macOS the accepted inode receives a unique, private, owned hard link. The
post-link ctime/link-count state is the operational baseline. Each named
boundary verifies the parent, accepted descriptor, published pathname, owned
link, and sidecar inventory without rehashing. Close verifies first, removes
only the owned link, and then allows the expected link-count/ctime transition
while retaining identity, size, mtime, pathname, and sidecar checks.

TDD evidence:

- First command, before implementation:
  `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -v`
  failed as expected with `ModuleNotFoundError: No module named
  'experiments.exp425_selective_recovery.database'`.
- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -v`
  — PASS, 5 tests. Covers one descriptor hash over explicit boundaries and
  close, replacement, sidecar, symlink, invalid digest, and closed use.
- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database experiments.exp417_causal_multihorizon_shadow.tests.test_inputs -q`
  — PASS, 30 tests.
- `python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py' -q`
  — PASS, 86 tests in 6.276s.
- `python3 -m compileall -q experiments/exp425_selective_recovery` — PASS.
- `git diff --check` — PASS.

Benchmark (temporary 256 MiB SQLite file, twenty immutable-connection synthetic
reads versus twenty descriptor hashes): 268,705,792 bytes;
`full_hash_count=1`; decoded lengths identical; one snapshot/20 reads 0.129s;
twenty full hashes 2.526s; snapshot path faster. No timing branch was added.

Residual risk: the synthetic test set currently exercises the descriptor and
connection boundary but not the full official 1,117-dataset context freeze; no
official dataset run was started.

## Fix round 1

Fixed the omitted `manifest[dataset] = identities` assignment in
`freeze_context`; context construction now receives complete exact per-dataset
TEST identity coverage rather than an empty manifest. Acceptance now requires
an unaliased pre-link regular inode, validates the owned hard link immediately,
sets post-link metadata as the operational baseline, opens immutable SQLite
without `resolve()` pathname substitution, and validates after connection open.
Later aliases are detected by link-count metadata. `close()` retains the
primary boundary failure while still closing SQLite and performing safe
identity-checked owned-link cleanup where possible.

Additional verification:

- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -q` — PASS, 6 tests.
- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database experiments.exp417_causal_multihorizon_shadow.tests.test_inputs -q` — PASS, 31 tests in 0.109s.
- `python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py' -q` — PASS, 87 tests in 6.366s.
- `python3 -m compileall -q experiments/exp425_selective_recovery` and `git diff --check` — PASS.

Residual risk remains: the full official population and all adversarial
parent-namespace/live-writer scenarios were not run against the official DB.

### Continued fix round 1

Added retained grandparent descriptor plus parent-name namespace verification;
renaming/replacing the parent directory is now detected at every boundary. The
audit records operational ctime/link count and accepted parent identity. A
committed parent-replacement regression now passes alongside alias tests.

- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -q` — PASS, 7 tests.
- `python3 -m compileall -q experiments/exp425_selective_recovery` and `git diff --check` — PASS.

### Completion of fix round 1

The snapshot now retains a non-blocking SQLite POSIX shared byte-range lock on
the SQLite locking page.  A separate journal-mode SQLite writer deterministically
fails its commit while the snapshot is live.  Its attempted journal namespace
transition is also rejected at the next named boundary before a loader can
return data.  This is an operational lock, not a timing-based claim.

Acceptance is now revalidated from the retained parent descriptor after the
one descriptor hash and again after macOS owned-link creation.  The published
path and owned link must both bind the accepted inode; the post-link
ctime/nlink state is recorded as the immutable operational baseline.  Unknown
SQLite-basename artifacts fail closed.  Exceptional cleanup verifies the owned
directory and link identities before unlinking and reports a replacement rather
than unlinking it.  The audit is a frozen value at every observation and now
records both lifecycle and whether the retained SQLite read lock is held.

`freeze_context` now uses the same Exp417 universe semantics: the frozen lane
population must be contained in the sorted DB population and produces the exact
sorted selected universe (DB-only rows outside the lane universe remain
excluded, as Exp417 requires).  It requires `lanes.config == FROZEN_CONFIG`,
constructs the full exact TEST `(instance_index, float32_length)` manifest with
the Exp417 SQL, and executes its after-boundary even if lane/config/query/context
construction fails.  TRAIN and TEST loaders likewise retain their primary
decoder/contract errors while always running their after-boundary.

Original review findings → resolved evidence:

1. Missing TEST manifest assignment: `test_one_hash_context_two_train_two_test_and_three_boundaries` asserts exact A/B manifests and decoded identities.
2. Weak snapshot acceptance race: retained-descriptor pre-link, immediate owned-link, and post-connect boundary revalidation in `open_database_snapshot`.
3. Parent pathname/namespace ABA: retained grandparent descriptor checks plus `test_boundary_rejects_parent_namespace_replacement`.
4. Frozen lane/config/universe drift: `freeze_context` requires exact frozen config and mirrors Exp417 selected-population semantics; `test_context_rejects_nonfrozen_config_and_nonexact_population`.
5. Repeated/full-hash coverage omitted: end-to-end test freezes context, loads two TRAIN and two TEST datasets, uses three explicit boundaries, closes, and proves exactly one descriptor hash.
6. Incomplete mutation/sidecar/alias matrix: pre/post known artifacts, unknown relevant artifact, truncation, byte mutation, chmod ctime, disappearance, replacement, and hardlink alias tests fail closed.
7. Journal-enabled writer not controlled: `test_retained_sqlite_read_lock_blocks_a_journal_writer` proves the external commit fails and the next boundary detects the attempted namespace mutation.
8. Exception/cleanup safety gaps: malformed-row, decoder, wrong context/contract, after-boundary, double-close, and macOS replaced-owned-link tests preserve primary errors and never unlink replacements.
9. Audit/benchmark evidence incomplete: frozen audit exposes operational metadata, lock and lifecycle; committed `DatabaseBenchmarkTests` exercises a real 256 MiB SQLite file with twenty Exp417-decoded snapshot reads versus twenty Exp417 verified-open/decoded reads (therefore twenty full descriptor hashes).

Fresh verification on macOS (Python 3.14):

- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -v` — PASS, 16 tests in 3.207s; benchmark `256MiB snapshot=0.129s full_hashes=2.555s`.
- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database experiments.exp417_causal_multihorizon_shadow.tests.test_inputs -v` — PASS, 41 tests in 3.305s; benchmark `snapshot=0.130s`, `full_hashes=2.545s`.
- `python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py' -v` — PASS, 97 tests in 9.495s; benchmark `snapshot=0.130s`, `full_hashes=2.557s`.
- `python3 -m compileall -q experiments/exp425_selective_recovery` — PASS.
- `git diff --check` — PASS.

Final post-review verification:

- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -q` — PASS, 18 tests after the final cleanup-race coverage additions.
- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database experiments.exp417_causal_multihorizon_shadow.tests.test_inputs -q` — PASS, 41 tests in 3.296s; `snapshot=0.129s`, `full_hashes=2.551s`.
- `python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py' -q` — PASS, 97 tests in 9.408s; `snapshot=0.124s`, `full_hashes=2.459s`.
- Explicit latest benchmark: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database.DatabaseBenchmarkTests.test_256mib_snapshot_reads_beat_twenty_full_descriptor_hashes -v` — PASS, 3.103s; `snapshot=0.131s`, `verified_opens=2.551s`.

Reviewer-fix final verification:

- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -q` — PASS, 18 tests in 3.220s; `snapshot=0.133s`, `verified_opens=2.562s`.
- `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database experiments.exp417_causal_multihorizon_shadow.tests.test_inputs -q` — PASS, 43 tests in 3.352s; `snapshot=0.135s`, `verified_opens=2.571s`.
- `python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py' -q` — PASS, 99 tests in 9.492s; `snapshot=0.132s`, `verified_opens=2.575s`.
- `python3 -m compileall -q experiments/exp425_selective_recovery` and `git diff --check` — PASS.

No official run or Task 5 work was started.  Residual: a process with the
same OS identity can always deliberately bypass POSIX advisory locks by writing
the file descriptor directly; boundaries fail closed on the resulting metadata
or namespace mutation.  The supported SQLite writer path is prevented from
committing by the retained SQLite-compatible read lock.
