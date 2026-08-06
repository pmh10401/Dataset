# Task 6 report — resumable deterministic runtime

## Reused persistence map (before edits)

The current `docs/codemap/codemap.json` was read before changing code.

| node | callers | downstream effect | matching tests |
| --- | --- | --- | --- |
| `artifact_utils` (`experiments/exp403_408/artifact_io.py`) | Exp417/Exp425/Exp426 checkpoint and publication modules | Canonical JSON bytes, SHA256 and atomic artifact writes define scientific hashes | `experiments/exp403_408/tests/test_prediction_boundary.py` |
| `exp426_checkpoint` (`experiments/exp426_verified_baseline/checkpoint_antigravity.py`) | `exp426_runtime`, Exp426 publication/reporting | Parent-only checkpoint records, provenance and recovery | `experiments/exp426_verified_baseline/tests/test_checkpoint_antigravity.py` |
| `exp426_runtime` (`experiments/exp426_verified_baseline/runtime_antigravity.py`) | Exp426 CLI/publication | Parent-owned DB verification, worker orchestration, checkpoint commits, progress/events/PID | `experiments/exp426_verified_baseline/tests/test_runtime_antigravity.py` |

Exp435 uses only the canonical byte/fsync patterns.  It does not reuse Exp426
schemas or record formats.

## Scope

Added the three runtime modules and their focused tests:

- `checkpoint_antigravity.py`: canonical scalar context fingerprint, no-clobber
  dataset records, atomic record-set digest, intent reconciliation, inventory
  and candidate-budget validation, read-only inspection and crash accounting.
- `progress_antigravity.py`: atomic progress JSON, fsync JSONL events, durable
  text log, PID ownership, 30-second heartbeat, lifecycle transitions and a
  recursive label/value firewall.
- `runtime_antigravity.py`: parent-only values loading, TRAIN-before-TEST
  boundary, CPU `ProcessPoolExecutor(max_workers=7)` with prefetch 14 and
  single-process MLX mode, deterministic name-ordered commits, resume and
  read-only inspect.

To keep the production path truthful, the existing values-only input boundary
was extended (without introducing any TEST label access):

- Exp417 accepts a concrete `RawTrainDataset` as prior TRAIN evidence before
  executing the fixed values-only TEST query.
- Exp425's retained descriptor exposes the matching before/after-boundary
  loader.
- Exp435 delegates `load_test_values(dataset, train)` to that path.
- `Exp435RunSummary` has `checkpoint_complete`, which means all 1,117 datasets
  are durably checkpointed but publication hashes have not yet been produced.

## Verification

```text
python3 -m unittest discover \
  -s experiments/exp435_train_only_consensus_recovery_antigravity/tests \
  -p 'test_*antigravity.py'
Ran 92 tests ... OK

python3 -m unittest \
  experiments.exp426_verified_baseline.tests.test_checkpoint_antigravity \
  experiments.exp426_verified_baseline.tests.test_runtime_antigravity
Ran 18 tests ... OK

python3 -m py_compile \
  experiments/exp417_causal_multihorizon_shadow/inputs.py \
  experiments/exp425_selective_recovery/database.py \
  experiments/exp435_train_only_consensus_recovery_antigravity/*.py \
  experiments/exp435_train_only_consensus_recovery_antigravity/tests/test_*antigravity.py
PASS

git diff --check
PASS
```

The runtime candidate-budget fixture observed 101 candidates, persisted a
`candidate_budget_exceeded` event before publication, and created no record for
the over-budget dataset.
