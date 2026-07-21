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
  before downstream evaluation.  Verification now requires an independently
  supplied (or frozen Exp266-derived) dataset population and exact key
  universe, rather than trusting a rewritten manifest.
- Explicit calibration abstention is a complete cache result.  It serializes
  absent p-value resolution as JSON `null`, never `NaN`.
- Cache reuse and prediction freeze now carry a run-contract fingerprint that
  includes local source/config/contracts/registry/pipeline code, Exp239 loader
  and calibration code, shared database/preprocessing code, policy constants,
  seeds, device request, DB hash, and the frozen TCN benchmark record hash.
- TCN cache execution holds a nonblocking `fcntl` exclusive lock throughout
  the whole source run.  Lock owner/timing details are operational manifest
  data only and are excluded from reproducibility hashes.

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

Follow-up validation added after the initial implementation:

```text
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_source_pipeline -v
python3 -m experiments.exp267_273_zero_support_fn_recovery.tests.spawn_smoke_helper
```

The focused suite now covers duplicate source input, duplicate/missing/extra
row keys, wrong alpha, out-of-range candidates, manifest plus JSONL tampering,
common-dependency and runtime-contract drift, TCN lock contention, and a
two-worker macOS `spawn` smoke using real scattering cache units.

Final verification after the hardening pass: focused Task 7 tests passed and
the Exp267-273 package suite passed `91/91` tests.  The real DB one-dataset
smoke again completed `cache -> freeze -> exact verify` for `ACSF1_normal_0`.
