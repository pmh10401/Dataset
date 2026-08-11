# Exp446 Task 5 report

## RED and GREEN

RED: `python3 -m unittest -v experiments.exp446_exp445_publication_materialization_repair_antigravity.tests.test_lifecycle_antigravity` — 5 tests / 9 subcases failed because the new lifecycle facade was absent.

GREEN: the same command — 5 tests / 9 subcases passed. Inherited regression:
`python3 -m unittest -v experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_runtime_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_checkpoint_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_progress_antigravity experiments.exp445_exp444_publication_source_authority_repair_antigravity.tests.test_lifecycle_antigravity` — 61 passed.

## Failure ordering

| Failure | Durable result |
| --- | --- |
| CPU worker is not 11 or labels are nonzero | rejected before root allocation |
| wrapper mismatch at 0, 15, or 29 | exact-next error record, checkpoint error, inactive progress/PID, no publication |
| aggregate or fresh predecessor mismatch | index 29 exact-next error, checkpoint error, inactive progress/PID, no publication |
| publication without terminal parity | facade rejects before importing a publisher |

## Recovery and root evidence

Existing 61-test regression confirms ordered checkpoint/error reconciliation and inactive progress/PID recovery contracts. Task fixtures use only temporary 30-record bytes, assert zero labels on success, never allocate the official Exp446 root, and do not run a scientific recomputation.

## Map and commit

Pending final map-triplet regeneration and commit.

## Concern

The Task 5 facade intentionally delegates a terminal-eligible publication to Task 6's reporting entrypoint, which is not yet implemented in this base; it rejects all pre-terminal attempts first.
