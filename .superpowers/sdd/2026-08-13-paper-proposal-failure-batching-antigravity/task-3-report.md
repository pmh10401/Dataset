# Task 3 failure batching report

## Scope delivered

- Reconstructed Task4+5 work from verified success records only; the pending queue is the catalog population minus those records.
- Added parent-only, create-only canonical failure attempt records in `proposal_work/task4_task5_attempts`, bound to context, catalog, backend, scientific configuration, code closure, raw identity, wave, and exception details.
- Continued after individual worker failures, immediately publishing successful results, and retried only the exact pending population on the next invocation.
- Replaced eager process submission with a rolling `min(pending, 2 * dataset_workers)` window. A replacement is submitted only after the completed future has been parent-persisted as success or failure.
- Added deterministic label-free risk-first scheduling for nonfinite, unsupported, small-TRAIN, large-TRAIN, and long-series cases. Reconstructed result mappings remain catalog-sorted.
- Added typed `proposal_score_incomplete` handling. Incomplete work blocks Task6/7, final shards/manifests, and the proposal-score journal completion; runner status reports exact verified-success, failed, and pending counts.
- Extended journal verification to accept only the registered, strictly validated success/attempt work namespace and to fail closed on ledger tampering or namespace pollution.

## TDD evidence

- RED: focused tests first failed because `ProposalScoreIncomplete` did not exist and eager submission reached 5 in-flight futures with a 4-future cap.
- GREEN: focused artifact/runner suite passed with 32 tests.

## Verification

- `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper -v` — 37 passed after the final runner boundary regression.
- `python3 -m py_compile paper/equipment_anomaly_detection/artifacts_paper.py paper/equipment_anomaly_detection/proposal_score_paper.py paper/equipment_anomaly_detection/runner_paper.py paper/equipment_anomaly_detection/pipeline_paper.py` — passed.
- Strict codemap-lock hash closure and `validate_codemap_triplet` — passed.
- `git diff --check` — passed.
- No proposal/training process or root launch was performed.

## Fix round 1

- Exact verified coverage now returns from Task4+5 before emitting zero-total progress, so restart reconstruction neither schedules workers nor divides by zero; composition then runs Task6 and Task7 exactly once.
- Incomplete payloads expose the exact pending dataset list and only the truthful `attempts_path`; the misleading `queue_path` alias was removed.
- Added focused interruption coverage: a parent-persisted coordinate survives interruption and only the remaining submitted-but-uncommitted coordinate is scheduled on restart, within the same rolling-window bound.

## Fix round 2

- Added a seven-coordinate, two-worker failure-wave stress test. It records the submitted/uncommitted window, verifies the cap remains four, verifies later coordinates replenish, and verifies a parent persistence occurs before replenishment.

## Final runner boundary regression

- Replaced the self-fulfilling lifecycle-mocked case with two real `run_pipeline` invocations using real `_write_complete`, journal verification/status, success/attempt records, shard writer, and freeze. Only catalog/data and scientific leaf computations are mocked: the first wave persists `alpha`, fails `beta`, and has no Task6/7/freeze/proposal journal; the second scores only `beta`, runs Task6/7 once, then creates and verifies the actual freeze/proposal completion evidence.

## Judgment calls

- Attempt records intentionally omit timestamps: their monotonic parent-assigned attempt number provides operational ordering without adding time to scientific identity.
- The existing CPU worker/backend policy is unchanged; the only scheduling change is the bounded rolling submission window.
