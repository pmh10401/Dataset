# Exp444 Task 3 implementation report

## Result

Task 3 now binds the exact frozen Exp442 publication/attempt artifacts and the
Exp404 Hard prediction, verifies their hashes and candidate semantics, and
exposes only copied float32 normal-TRAIN values plus the 30 bound TEST values.
The SQLite boundary is descriptor-pinned, immutable/read-only, schema and
query-shape allowlisted, and has zero TEST-label access.

The Task 2 Exp442 path defect was corrected to the existing 20260808 frozen
publication/attempt artifacts. No Exp444 official root was created and no
Exp440/442/443 artifact was modified.

## TDD evidence

The required RED run was executed before implementation:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_config_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity
```

It failed on the exact Exp442 path assertion because the config still pointed
at the nonexistent 20260809 location, and on
`ModuleNotFoundError: No module named 'experiments.exp444_train_only_stability_precision_gate_antigravity.sources_antigravity'`.

Focused GREEN run:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_config_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_contracts_antigravity
.....................
----------------------------------------------------------------------
Ran 21 tests in 0.007s

OK
```

### Map, root, and diff verification

The synchronized map validation returned:

```text
CODEMAP_OK nodes=20 edges=62 flows=5 anchors=all unknown=[]
HTML_EMBED_OK
LOCK_FINGERPRINTS_OK
```

The exact official-root absence command and output were:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python - <<'PY'
from pathlib import Path
from experiments.exp444_train_only_stability_precision_gate_antigravity.config_antigravity import FROZEN_CONFIG
root = Path(FROZEN_CONFIG.run_root)
assert not root.exists(), root
print('OFFICIAL_ROOT_ABSENT', root)
PY
OFFICIAL_ROOT_ABSENT /Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810
```

`git diff --check` exited 0 with no output.

### Candidate-metadata attestation RED → GREEN

The mutation check then exercised a same-coordinate snapshot with an altered
raw length and a detached snapshot without the parent-owned verified session.
Before the signature/session attestation was added, the exact two-test command
failed after the first case had already executed SQLite queries:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_loader_rejects_altered_candidate_metadata_before_any_query_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_loader_requires_parent_owned_verified_session_before_database_open_antigravity
FF
----------------------------------------------------------------------
Ran 2 tests in 4.989s

FAILED (failures=2)
```

After sessions retained the exact sorted candidate signatures and the loader
required that parent-owned verified session, the same command passed:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_loader_rejects_altered_candidate_metadata_before_any_query_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_loader_requires_parent_owned_verified_session_before_database_open_antigravity
..
----------------------------------------------------------------------
Ran 2 tests in 3.332s

OK
```

### Final test and real-source verification

The final focused Task 3 command passed all behavioral and adversarial tests:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity
....................
----------------------------------------------------------------------
Ran 20 tests in 5.084s

OK
```

The full Task 2 + Task 3 package suite passed:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests -p 'test_*antigravity.py'
.........................................
----------------------------------------------------------------------
Ran 41 tests in 5.072s

OK
```

Dedicated pin failure/success tests passed:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_close_failure_injections_preserve_truthful_pin_and_root_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_successful_close_removes_only_owned_pin_and_restores_link_count_antigravity
..
----------------------------------------------------------------------
Ran 2 tests in 0.017s

OK
```

The real immutable source replay, 30 candidate loads, zero labels, 60 executed
values-only queries, and successful owned-pin cleanup returned:

```text
EXP444_SOURCE_OK 30 0 60
EXP444_PIN_SUCCESS_OK False False
```

The original Task 2 + Task 3 package command ran 28 tests successfully. Real
default-source verification and loading returned:

```text
EXP444_SOURCE_OK 30 0 60
PIN_CLOSED_OK
```

The first value is the candidate count, the second is `label_access_count`,
and the final value is the executed values-only query count.

## Source and firewall checks

- Exact configured source paths and all six frozen identities are checked;
  artifact map, manifest, selection, prediction, ranking, source-audit, status,
  and Exp404 Hard governance are replayed before opening the database.
- Prediction, ranking, and selection candidates are cross-checked by complete
  coordinate/raw-length/depth/rank signatures; the canonical selected-coordinate
  identity is `db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048`.
- JSON objects reject duplicate keys, JSONL requires a final newline and
  nonblank rows, and regular-file reads reject symlink/FIFO/non-regular files,
  path replacement, inode/metadata changes, and SHA mismatches.
- Only the exact normal-TRAIN values SQL and bound TEST-values SQL are
  registered. TEST labels, `labels_blob`, `SELECT *`, unknown tables/columns,
  and unregistered SQL fail closed before execution; denied queries do not
  increment label access.
- The database is hashed through an open descriptor, hard-linked into a private
  pin, reopened read-only through that descriptor, and checked before/after each
  query and close. A failed close retains its own failure pin/root; successful
  close removes only that pin and restores the source link count.
- Whole-row nonfinite TRAIN rows are removed before exact-length grouping. Each
  selected TEST row is bound by dataset/index, must occur exactly once, and must
  match the frozen raw length. Returned train/test arrays are detached,
  little-endian float32, and read-only.

## Code-map synchronization

`docs/codemap/codemap.json`, `codemap.html`, and `codemap.lock` were regenerated
together. Validation passed with 20 nodes, 62 edges, 5 flows, and an exact
base64 HTML payload match. The Exp444 node now documents the Task 3 source
entrypoints, constraints, evidence, and source tests. The lock fingerprint
includes the new source/test files.

## Files and commit

- `experiments/exp444_train_only_stability_precision_gate_antigravity/sources_antigravity.py`
- `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_sources_antigravity.py`
- `experiments/exp444_train_only_stability_precision_gate_antigravity/contracts_antigravity.py`
- `experiments/exp444_train_only_stability_precision_gate_antigravity/config_antigravity.py`
- `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_config_antigravity.py`
- synchronized `docs/codemap/codemap.{json,html,lock}`

No MLX work was applicable: this task is an immutable values-only source
boundary, not a score-source experiment; no frozen algorithm was rewritten.

## Fix Round 1

### Reviewer findings and corrections

1. Every production database session is now constructed from the verified
   frozen 30 candidates. Parameter validation binds normal-TRAIN reads to their
   30 datasets and TEST reads to their exact 30 coordinates. Unbound queries
   are denied before `sql_trace`, execution count, or label-access count changes.
2. `FrozenSourceSnapshotAntigravity` now requires exactly 30 candidates, one
   per dataset, ranks exactly 1-30, valid immutable candidate contracts, and a
   recomputed canonical coordinate identity equal to the frozen identity.
   Loader preflight repeats population, hashes, database path, identity, and
   session-bound checks before opening or querying SQLite.
3. Has-attribute placeholders were replaced by behavioral adversarial tests
   covering regular-file/path replacement, symlink/FIFO/directory rejection,
   strict JSON/JSONL, duplicate/rank/dimension/source-map/replay failures,
   database replacement/metadata drift, exact TEST identity, SQL shape and
   parameter denial, counters, nonfinite TRAIN filtering, and immutable arrays.
4. Close validates link count and root inventory while the pin exists. Injected
   failures before descriptor close, before unlink, after unlink, and before
   root removal retain or restore the exact pin/root. Successful close removes
   only its own pin/root and restores the original source link count.

### RED evidence

The reviewer tests were added before the production corrections and run with
the exact isolated interpreter:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity
E.EEEEEEEF....EEEEEEEE.
----------------------------------------------------------------------
Ran 17 tests in 2.019s

FAILED (failures=1, errors=16)
```

The intended Critical failures were observed directly: a fabricated
one-candidate snapshot raised no error, while bound-session construction,
candidate preparation, and cross-source replay validation were absent. The
remaining errors were downstream tests depending on those missing boundaries.

### Focused GREEN evidence

After the minimal production changes and test-fixture cleanup correction, the
same focused command passed:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity
..................
----------------------------------------------------------------------
Ran 18 tests in 1.689s

OK
```

## Fix Round 2

### Boundary correction

- Removed production `_from_test_fixture_antigravity`; there is no longer a
  SQLite-capable fixture or public constructor that accepts caller-selected
  paths, hashes, datasets, coordinates, or candidate metadata.
- `verify_frozen_sources_antigravity` first proves exact frozen Exp442
  prediction/ranking/selection agreement, orders candidates by contiguous
  original rank 1-30, and only then mints a module-private identity-token
  attestation used to initialize the database session.
- The canonical full-candidate SHA covers tuple order, dataset, instance index,
  raw length, all original depths, and original rank. Snapshot construction,
  loader preflight, and the verifier-owned session independently retain or
  recompute that attestation before any query.
- Existing SQL firewall, zero-label counters, source/pin identity checks, and
  all four injected close-failure pin-retention guarantees now run against the
  real verifier-owned frozen session rather than an arbitrary test database.

### Exact RED evidence

Before the production correction, the three new tests were run in the current
round-1 tree with this exact command:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_reversed_snapshot_rank_order_is_rejected_before_query_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_public_session_rejects_jointly_forged_depths_before_database_open_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_public_session_rejects_caller_chosen_database_and_hash_antigravity
FFF
======================================================================
FAIL: test_reversed_snapshot_rank_order_is_rejected_before_query_antigravity
AssertionError: ValueError not raised
======================================================================
FAIL: test_public_session_rejects_jointly_forged_depths_before_database_open_antigravity
AssertionError: ValueError not raised
======================================================================
FAIL: test_public_session_rejects_caller_chosen_database_and_hash_antigravity
AssertionError: ValueError not raised
----------------------------------------------------------------------
Ran 3 tests in 6.895s

FAILED (failures=3)
```

The requested historical baseline was also reconstructed without altering the
branch: disposable worktree
`/tmp/exp444_round2_red_antigravity_FyiGHY` used detached base
`127f0dbfbac168357f1eba8db927c7904bcfbd4e` plus the reviewer RED test file.
The same three-test command failed with three errors (missing execution counter
and the older constructor rejecting the later `candidates` keyword), proving
that baseline was not GREEN. `git worktree remove --force` removed that exact
temporary worktree and `test ! -e` printed `TEMP_RED_WORKTREE_REMOVED`.

### GREEN and real-source evidence

The same three-test command after the correction returned:

```text
...
----------------------------------------------------------------------
Ran 3 tests in 4.954s

OK
```

Focused and full-package suites returned:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity
.......................
----------------------------------------------------------------------
Ran 23 tests in 26.434s

OK

/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests -p 'test_*antigravity.py'
............................................
----------------------------------------------------------------------
Ran 44 tests in 26.399s

OK
```

Dedicated close tests passed 2/2 in 8.184s. Exact real replay returned:

```text
EXP444_SOURCE_OK 30 0 60
EXP444_PIN_SUCCESS_OK False False
```

MLX remains inapplicable: this correction changes only source provenance and
session authorization, not a score-source operation or frozen algorithm.

## Fix Round 3

### Truthful failure evidence correction

On every failed close, the session now verifies the retained pin against the
original descriptor-bound device, inode, and database SHA. If the owned pin was
replaced, only that replacement is unlinked and the exact original inode is
relinked from verified source authority. A duplicated verified descriptor stays
open through cleanup; if exact relinking is impossible, its exact bytes are
written to a separate private `antigravity` evidence root, SHA-verified, retained,
and reported in the combined close error. Pin inode replacement, source pathname
replacement, and source metadata drift are separate behavioral tests.

### Exact reconstructed RED

Disposable worktree `/tmp/exp444_round3_red_antigravity_qhWCWb` used detached
base `40fa7a04` plus only the new RED test changes. Exact command:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest -v experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_owned_pin_inode_replacement_restores_truthful_evidence_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_source_database_pathname_replacement_retains_original_pin_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_source_database_metadata_drift_retains_original_pin_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_unrestorable_pin_preserves_truthful_descriptor_evidence_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_close_failure_injections_preserve_truthful_pin_and_root_antigravity
test_owned_pin_inode_replacement_restores_truthful_evidence_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_owned_pin_inode_replacement_restores_truthful_evidence_antigravity) ... FAIL
test_source_database_pathname_replacement_retains_original_pin_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_source_database_pathname_replacement_retains_original_pin_antigravity) ... ok
test_source_database_metadata_drift_retains_original_pin_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_source_database_metadata_drift_retains_original_pin_antigravity) ... ok
test_unrestorable_pin_preserves_truthful_descriptor_evidence_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_unrestorable_pin_preserves_truthful_descriptor_evidence_antigravity) ... ERROR
test_close_failure_injections_preserve_truthful_pin_and_root_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_close_failure_injections_preserve_truthful_pin_and_root_antigravity) ... ok

======================================================================
ERROR: test_unrestorable_pin_preserves_truthful_descriptor_evidence_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_unrestorable_pin_preserves_truthful_descriptor_evidence_antigravity)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/private/tmp/exp444_round3_red_antigravity_qhWCWb/experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_sources_antigravity.py", line 504, in test_unrestorable_pin_preserves_truthful_descriptor_evidence_antigravity
    evidence = session.failure_evidence_path
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'ValuesOnlyDatabaseSessionAntigravity' object has no attribute 'failure_evidence_path'

======================================================================
FAIL: test_owned_pin_inode_replacement_restores_truthful_evidence_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_owned_pin_inode_replacement_restores_truthful_evidence_antigravity)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/private/tmp/exp444_round3_red_antigravity_qhWCWb/experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_sources_antigravity.py", line 448, in test_owned_pin_inode_replacement_restores_truthful_evidence_antigravity
    _assert_truthful_failure_pin_antigravity(self, session, pin)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/private/tmp/exp444_round3_red_antigravity_qhWCWb/experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_sources_antigravity.py", line 83, in _assert_truthful_failure_pin_antigravity
    testcase.assertEqual((pin.stat().st_dev, pin.stat().st_ino), session.descriptor_inode)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Tuples differ: (16777230, 17367064) != (16777230, 17367059)

First differing element 1:
17367064
17367059

- (16777230, 17367064)
?                  ^^

+ (16777230, 17367059)
?                  ^^


----------------------------------------------------------------------
Ran 5 tests in 16.350s

FAILED (failures=1, errors=1)
```

The exact cleanup returned:

```text
ROUND3_RED_WORKTREE_REMOVED /tmp/exp444_round3_red_antigravity_qhWCWb
```

### Exact GREEN

The identical five-test `unittest -v` command returned:

```text
test_owned_pin_inode_replacement_restores_truthful_evidence_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_owned_pin_inode_replacement_restores_truthful_evidence_antigravity) ... ok
test_source_database_pathname_replacement_retains_original_pin_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_source_database_pathname_replacement_retains_original_pin_antigravity) ... ok
test_source_database_metadata_drift_retains_original_pin_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_source_database_metadata_drift_retains_original_pin_antigravity) ... ok
test_unrestorable_pin_preserves_truthful_descriptor_evidence_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_unrestorable_pin_preserves_truthful_descriptor_evidence_antigravity) ... ok
test_close_failure_injections_preserve_truthful_pin_and_root_antigravity (experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity.Exp444SourceBoundaryAntigravityTests.test_close_failure_injections_preserve_truthful_pin_and_root_antigravity) ... ok

----------------------------------------------------------------------
Ran 5 tests in 25.394s

OK
```

The final focused source suite passed 26 tests in 45.098s. The full package
suite passed 47 tests in 51.230s. Real frozen replay returned:

```text
EXP444_SOURCE_OK 30 0 60
EXP444_PIN_SUCCESS_OK False False
```

MLX remains inapplicable because this is a source-boundary correctness repair,
not a new score-source operation.
