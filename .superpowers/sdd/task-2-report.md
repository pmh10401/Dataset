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

## Review Fixes

- Enforced the raw DB contract: exactly 1,119 unique names, explicit frozen exclusions of `CornellWhaleChallenge` and `Wafer_normal_1`, and exact equality between the remaining 1,117 DB names and Exp253 baseline coverage.
- Added exact baseline schema, row identity, Exp253 manifest, and Exp253 audit provenance validation. Added exact Exp260 override schema, 20-row manifest/hash validation, and Exp260 audit provenance validation for experiments `[260, 261, 262]`, `alpha=0.005`, and the config-declared `corrected_four_sources` output.
- All malformed source conditions now raise `SourceContractError`; the CLI returns exit code 2 for that error class. This includes missing DB datasets, required candidate/quality fields, non-list index fields, schema mismatches, missing experiment identity, and manifest hash/count mismatches.
- Numeric validation is exact: alpha selection has no tolerance and rejects non-float values; `n_test` and every index reject booleans and fractional values.

### Added Negative Coverage

- Boolean and non-exact alpha selection.
- Boolean and fractional indices and `n_test`.
- Missing `candidate_indices` and `quality_flag_indices`.
- Non-list index fields, schema mismatch, and missing experiment.
- Baseline dataset absent from a count-valid raw DB fixture.
- Manifest SHA256 and row-count mismatch.
- CLI exit code 2 for a contract error.

### Commands And Results

- `python3 -m unittest experiments.exp263_264_full_missing_aware_integration.tests.test_inputs -v`
  - Result: 16 tests passed.
- `python3 -m py_compile experiments/exp263_264_full_missing_aware_integration/inputs.py`
  - Result: exit 0.
- `python3 -m experiments.exp263_264_full_missing_aware_integration.inputs`
  - Result: exit 0; live read-only audit wrote both audit artifacts.
- Live audit result: raw DB `1119`; frozen exclusions exactly `CornellWhaleChallenge`, `Wafer_normal_1`; frozen population `1117`; baseline/statistical/structural/change-point coverage `1117`; reconstruction `251`; missing override `20`.
- `git diff --check`
  - Result: clean before the implementation commit.

### Review Fix Commit

- `5905f3e5f75667756e3da24dc5147b030b68d74d` (`fix: harden exp263 source contract audit`)
