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
