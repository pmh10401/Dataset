# Exp444 Task 6 report — ordered checkpoint and self-written progress

## Scope

Task 6 adds the durable, label-free execution evidence used by the later
runtime.  It owns only the exact 30-candidate prefix and never imports TEST
labels, evaluation metrics, or anomaly positions.

- `checkpoint_antigravity.py`: create-only canonical records, fsync-before-
  state ordering, atomic state replacement, record-hash prefix, exact-next
  orphan reconciliation, stale/live PID ownership, interruption resume, and
  read-only complete finalization re-entry.
- `progress_antigravity.py`: parent-only atomic progress/PID JSON and fsynced
  append-only JSONL/Korean text logs.  Live records include dataset,
  candidate/view dimensions, TRAIN/TEST sizes, backend, counters, throughput,
  elapsed time, ETA, checkpoint, and latest error.  Terminal records clear
  current dimensions and set ETA to JSON `null`.

## TDD evidence

### RED

Command:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_checkpoint_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_progress_antigravity
```

Result before implementation: `Ran 6 tests ... FAILED (errors=6)` with
`ModuleNotFoundError` for the two missing Task 6 modules.

### GREEN

Focused command:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_checkpoint_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_progress_antigravity
```

Result: `Ran 6 tests ... OK`.

Full Exp444 suite:

```text
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests -p 'test_*.py'
```

Result: `Ran 63 tests ... OK` (no failures or skips).

Additional checks:

```text
git diff --check
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m py_compile experiments/exp444_train_only_stability_precision_gate_antigravity/checkpoint_antigravity.py experiments/exp444_train_only_stability_precision_gate_antigravity/progress_antigravity.py experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_checkpoint_antigravity.py experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_progress_antigravity.py
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
```

All completed successfully.  The code-map validator found 20 nodes, 62
evidence-backed edges, 5 flows, and zero unknown relationships; every node,
evidence path, symbol, edge reference, and flow step validated.

## Commits

- `d9ef121d9a76c65f1c535d20bb9c36c74e563e21` —
  `feat(exp444): persist ordered stability progress`
- `f760056458e09a29da4381acd0ed4b6b1ac17626` —
  `docs(exp444): map checkpoint progress evidence`

The worktree is clean at the second commit.  No official Exp444 root was
created and no database or TEST-label access occurred.
