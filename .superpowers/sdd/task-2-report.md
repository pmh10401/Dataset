# Task 2 Report: Population And Frozen Source Coverage

## Files

- Created: `experiments/exp263_264_full_missing_aware_integration/inputs.py`
- Updated: `experiments/exp263_264_full_missing_aware_integration/tests/test_inputs.py`
- Audit outputs written read-only from sources:
  - `/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/00_source_contract_audit.json`
  - `/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/01_source_coverage.csv`

## Implementation

- Added `load_population()`, `load_source_maps()`, `audit_source_contract()`, and exact `(dataset, variant, alpha)` selection.
- Reads only DB dataset names, `train_normal_count`, and `test_total_count` through SQLite read-only mode. It does not read `instances`, TEST labels, anomaly positions, or test-label aggregates.
- Validates label-free metadata, source-manifest SHA256, selected-row coverage, DB/source names, `n_test`, candidate index bounds, source experiment metadata, and missing override source maps.
- The module CLI returns exit code `2` for missing, duplicate, mismatched, or out-of-range source contracts.

## Commands And Results

- `python3 -m unittest experiments.exp263_264_full_missing_aware_integration.tests.test_inputs -v`
  - Result: 5 tests passed.
- `python3 -m experiments.exp263_264_full_missing_aware_integration.inputs`
  - Result: exit 0; wrote both audit artifacts.
- `python3 -m py_compile experiments/exp263_264_full_missing_aware_integration/inputs.py`
  - Result: exit 0.
- `git diff --check`
  - Result: clean before the implementation commit.

## Coverage Facts

- Frozen source-backed DB population: 1,117 datasets.
- Baseline, statistical, structural, and change-point coverage: 1,117 each.
- Reconstruction coverage: 251 datasets (`train_normal_count >= 100`).
- Missing override coverage: 20 datasets; no missing dataset lacks its finalized override.
- Audit records `contains_test_labels: false`, `contains_evaluation_metrics: false`, and `database_read_only: true`.

## Commit

- Implementation: `c737b0db372475c7f1e371d8c463baa7b06b9057` (`feat: audit exp263 frozen score sources`)

## Concerns

- The raw DB has 1,119 datasets; `CornellWhaleChallenge` and `Wafer_normal_1` are absent from the frozen Exp253 baseline, so the validated population is the 1,117 source-backed DB intersection.
- The finalized baseline and missing-override artifacts do not carry `variant` or `alpha`; their exact schemas and manifest SHA256 are validated instead. The four score-source artifacts use strict `(dataset, variant, alpha)` selection.
