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
FAIL: exact Exp442 path expected 20260808/publication/attempt artifact, config pointed at 20260809
ERROR: missing experiments...sources_antigravity module
```

Focused GREEN run:

```text
... -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_config_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_contracts_antigravity
Ran 21 tests in 0.007s
OK
```

All Task 2 + Task 3 package tests also passed (`Ran 28 tests ... OK`). Real
default-source verification and loading passed:

```text
EXP444_SOURCE_OK 30 0 60
PIN_CLOSED_OK
```

The first value is the candidate count, the second is `label_access_count`,
and the final value is the number of registered values-only queries.

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
