# Exp455 Task 4 report — four comparisons, artifacts, and replay

## Preflight map answers

- Callers: the Task 3 router supplies eight fold decisions; Task 4 evaluation supplies artifact publication, semantic replay, and aggregate dashboard export.
- Effects: the evaluator sums exactly 48 immutable confusion rows for four comparison methods.  Publication is restricted to an injected temporary root and replay re-computes the result; neither changes routing or official experiment state.
- Tests: `test_evaluation_antigravity.py`, `test_artifacts_antigravity.py`, `test_replay_antigravity.py`, and `test_dashboard_export_antigravity.py` cover the new boundary.  Task 2–3 tests cover input construction and LODO selection.

## TDD evidence

- RED: the new focused suite ran before modules existed: 6 tests, 6 expected `ModuleNotFoundError` failures for evaluation/artifacts/replay/dashboard export.
- First GREEN: `python3 -m unittest ...test_evaluation... ...test_artifacts... ...test_replay... ...test_dashboard_export...` — 6 tests passed in 0.024s.
- Final GREEN 1: the same four focused modules — 6 tests passed in 0.018s.
- Final GREEN 2: the same four focused modules — 6 tests passed in 0.018s.

## Hand-calculated synthetic fixture

- `always_baseline`: `(TP, FP, FN, TN) = (480, 48, 240, 4320)`.
- `family_wide_local_gap`: `(496, 56, 226, 4288)`.
- `oracle_per_dataset`: `(490, 53, 220, 4300)`.
- `lodo_router`: `(490, 53, 220, 4300)`.
- LODO family FN changes: ECG5000 `-20`, ChlorineConcentration `0`.
- This synthetic fixture qualifies only as a calculation check; it is not an official Exp455 result.

## Semantic replay and artifact evidence

- Publication creates exactly the eight required leaves with canonical JSON/JSONL and a self-excluding map.
- Existing root, partial root, and symlink destination rejection are tested in a `TemporaryDirectory`.
- Replay first matches the synthetic evaluation.  Then a baseline FN is altered and the artifact map is recomputed; semantic replay rejects the changed result.
- Dashboard export test accepts only experiment ID, status, completed folds, four aggregate method rows, two family FN-delta rows, and pass/fail.

## Regression and boundary checks

- Task 2–3 regression: `test_features_antigravity` + `test_router_antigravity` — 7 tests passed in 0.011s; no real source capture was invoked.
- `python3 scratch/build_codemap_exp455_antigravity.py --validate-only` passed.
- `git diff --check` passed; four touched HTML files parsed with `html.parser`.
- The fixed official Exp455 root remains absent. No official command, TEST-label read, MLX work, routing activation, champion/pending change, or promotion occurred.
- The plan has Task 4 heading and its eight steps checked; Tasks 5–6 remain unchecked. Dashboard is `Tasks 4/6`, `8/8` folds, calculation-ready, official result `not_started`.

## Commit and self-review

- Commit: `8708c222d53fce3a996dde0719da97ae05df52c7` (`feat(exp455): evaluate conservative routing`).
- Self-review: exact 48-row validation rejects missing, duplicated, and boolean confusion data; F1 is derived from the exact integer numerator/denominator before conversion to float. Replay recomputes LODO decisions and all semantic leaves rather than trusting stored metrics.
- Concerns: none.
