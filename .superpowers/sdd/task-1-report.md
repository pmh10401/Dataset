# Task 1 Report

## Status

DONE_WITH_CONCERNS

## Files changed

- `docs/EXP263_264_PREREGISTRATION_KO.md`
- `experiments/exp263_264_full_missing_aware_integration/config.py`
- `experiments/exp263_264_full_missing_aware_integration/tests/test_inputs.py`

The pre-existing dirty worktree files were not modified, staged, or included.

## Implementation

- Added the preregistered constants `ALPHA = 0.005`, `MIN_TRAIN_FOR_PROMOTION = 100`,
  `REQUIRED_PROMOTION_AGREEMENT = 3`, `PRIORITY_AGREEMENT = 2`, and the fixed source variants.
- Added frozen `SourceSpec` records and the six-entry `SOURCE_CONTRACT` for Exp253,
  Exp252, Exp254, Exp255, Exp257, and the Exp260-262 missing-data override.
- Added absolute DB and output paths.
- Documented the 1,117-dataset scope, fixed routing rules, forbidden prediction/evaluation
  information, and retrospective interpretation limits in Korean.

## Commands and output summary

1. `python3 -m unittest experiments.exp263_264_full_missing_aware_integration.tests.test_inputs -v`
   - RED: failed with the expected missing `config` import before implementation.
   - GREEN: passed, `Ran 1 test`, `OK`.
2. `python3 -m py_compile experiments/exp263_264_full_missing_aware_integration/config.py experiments/exp263_264_full_missing_aware_integration/tests/test_inputs.py`
   - Passed with no output.
3. Source artifact existence check
   - `source_contract_entries=6`
   - `missing_artifacts=[]`
4. `git diff --check` and staged diff check
   - Passed with no whitespace errors.

## Commits

- Implementation commit: `286dd24ca2ee871f5dd096dd8b674fb8e533c246`
- Report commit: recorded after this report is committed.

## Concerns

- The specified Task 1 test covers the policy constants only; full source coverage,
  database validation, routing, prediction, and evaluation remain Task 2 onward.
- The implementation commit deliberately excludes all unrelated pre-existing dirty
  and untracked files in the worktree.

## Fix Review

The review finding was addressed by extending `test_inputs.py` with focused assertions
for the exact six-entry `SOURCE_CONTRACT`, every fixed variant and alpha, the exact
absolute DB/output paths, and every exact absolute source artifact path.

Command:

`python3 -m unittest experiments.exp263_264_full_missing_aware_integration.tests.test_inputs -v`

Output summary:

`Ran 3 tests in 0.000s` and `OK`.

Fix commit SHA: `ebe1115445e9f35afd4feb5f447d85083140890d`

## Fix Review 2

The remaining review gap was addressed by asserting every `SourceSpec.experiment` and
`SourceSpec.scope`. The test now fixes `all_1117` for the baseline, statistical,
structural, and change-point sources, `n_train_at_least_100` for reconstruction, and
`missing_20` for the missing-data override.

Command:

`python3 -m unittest experiments.exp263_264_full_missing_aware_integration.tests.test_inputs -v`

Output summary:

`Ran 3 tests in 0.000s` and `OK`.

Fix commit SHA: `33538039d12c3cdcae85e652d23f42542f35d348`
