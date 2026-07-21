# Task 7: Shared Source Cache And Frozen Prediction Pipeline

## Scope

- Added `source_registry.py` with concrete source/variant registrations.
- Added `source_pipeline.py` with atomic cache, cache validation, source-level
  manifests, label-free prediction freeze, and SHA-256 verification.
- Added focused tests in `tests/test_source_pipeline.py`.

## Contracts Verified

- The TCN median and intersection views share one `normal_forecast` evidence
  identity, so downstream routing cannot count them as independent votes.
- Each cache unit stores TRAIN OOF and TEST score arrays, provenance, config,
  DB, source-code, and input hashes.  Tampered, stale, incomplete, or
  self-hash-invalid cache units rebuild before reuse.  Prediction freeze
  rechecks current DB/code/input hashes before accepting a cache unit.
- CPU sources use no more than seven process workers.  TCN requires exactly
  one worker and a pre-verified, frozen device benchmark record.
- Worker failures are retained in sorted `cache_errors.csv`; the source
  manifest is marked incomplete and prediction freeze rejects it.
- Frozen JSONL records contain no TEST labels or evaluation metrics, are
  canonically ordered by source/variant/dataset/alpha, and are SHA-256 locked
  before downstream evaluation.
- Explicit calibration abstention is a complete cache result.  It serializes
  absent p-value resolution as JSON `null`, never `NaN`.

## Validation

Executed:

```text
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_source_pipeline -v
python3 -m py_compile experiments/exp267_273_zero_support_fn_recovery/source_registry.py experiments/exp267_273_zero_support_fn_recovery/source_pipeline.py experiments/exp267_273_zero_support_fn_recovery/tests/test_source_pipeline.py
python3 -m unittest discover -s experiments/exp267_273_zero_support_fn_recovery/tests -v
git diff --check
```

Focused tests cover cache reuse/rebuild, atomic-resume recovery, deterministic
error manifests, incomplete-source rejection, canonical order, alpha boundary,
prediction byte/content tampering, abstention serialization, and fixture-level
coverage contracts.  A label-free real-data smoke cache and freeze completed
for `ACSF1_normal_0` using the scattering source.  Full 1,117-dataset source
execution remains a later pipeline stage; no TEST-label evaluation was
performed in this task.
