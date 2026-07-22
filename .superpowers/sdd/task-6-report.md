# Task 6 Report: Independent Reaggregation And Interpretation

## Source And Artifact SHA

- Required starting code SHA: `fdc6c35f` (`fix(exp264): enforce evaluation artifact contract`)
- Frozen prediction SHA256: `8c7a2f52027c6a3127c79166bb109b61e348558574d98a42a8ccf5622921d653`
- Configured DB: `/Users/minho/Documents/Dataset/univariate_ts.db`
- DB SHA256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Full file hashes are in `/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/MANIFEST.json`.
  The manifest records configured and resolved DB paths and intentionally does not hash itself.

## Files

- `experiments/exp263_264_full_missing_aware_integration/reporting.py`
- `experiments/exp263_264_full_missing_aware_integration/tests/test_reporting.py`

## Commands And Results

- `python3 -m unittest experiments.exp263_264_full_missing_aware_integration.tests.test_reporting -v`
  - Initial TDD run failed because `reporting.py` did not exist.
  - Final focused run passed: 2 tests.
- `python3 -m unittest discover -s experiments/exp263_264_full_missing_aware_integration/tests -v`
  - Passed: 43 tests.
- `python3 -m py_compile experiments/exp263_264_full_missing_aware_integration/reporting.py experiments/exp263_264_full_missing_aware_integration/tests/test_reporting.py`
  - Passed.
- `generate_report(/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration)`
  - Recomputed all lane totals directly from the persisted `results.csv`.
  - Required exact agreement with `summary.json` for datasets, alerts, TP, FP,
    micro precision, macro Mean F1, and Zero-F1 counts for baseline Hard,
    autonomous Hard, Standard, Priority, and combined review.
  - Wrote `independent_reaggregation.json`, `RESULT_REPORT.md`, and `MANIFEST.json`.

## Results

- Exact summary agreement: `true` from aggregate totals across all 1,117 result rows.
- Autonomous Hard: 2,255 alerts, 1,917 TP, 338 FP, precision `0.850111`,
  Mean F1 `0.608711`, and 356 Zero-F1 datasets.
- Source raw-candidate TP/FP: statistical `703/139`, structural `861/142`,
  change point `613/137`, reconstruction `559/88`.
- Baseline-Standard source agreement distribution (0/1/2/3/4):
  `700/106/22/2/0`.
- Promoted: 2 candidates, 2 TP, 0 FP. Quality-blocked: 0 candidates.
- Hard confidence-confirmed: 1,115 candidates, 1,022 TP, 93 FP.
- Standard review workload changed `830 -> 806` (-24); Priority changed `9 -> 31` (+22).

## Concerns

- This is retrospective integration validation, not prospective validation or
  end-to-end strict TRAIN-only operating evidence. No policy, threshold,
  source, or lane was selected from these TEST results.
- Exp253 provenance remains `selected_from_prior_test_review=True` and
  `deployment_eligible=False`; autonomous Hard and human-assisted review
  metrics remain separate.
- The configured DB path resolves through a symlink. Both paths and the
  content hash are retained in the manifest for reproducibility.
- There were no quality-blocked candidates in this frozen result, so that
  diagnostic path is covered by the focused fixture test rather than live rows.

## Review Fixes

### Routing Contract Reaggregation

- Before loading any TEST truth, reporting now rederives `promoted_indices` and
  `quality_blocked_indices` from baseline Standard, four-source agreement,
  `n_train`, and quality flags.
- The rederived promotion rule exactly mirrors routing: `n_train >= 100`, at
  least three supporting sources, and no quality flag. The quality-blocked rule
  records flagged baseline-Standard candidates with at least three supporting
  sources.
- A mismatch fails closed with the dataset, field, expected set, and persisted
  set. Focused tests prove impossible promoted sets and stale quality-blocked
  sets fail before the truth loader is called.
- The fixture quality-blocked index is now inside baseline Standard and is a
  flagged three-of-four candidate.

### Manifest And Wording

- `MANIFEST.json` now hashes `reporting.py`, `config.py`, `prediction.py`,
  `routing.py`, `evaluation.py`, `inputs.py`, and
  `exp226_235_hard_alert_quality/database.py` in addition to the DB, inputs,
  frozen prediction, and non-manifest outputs.
- The current real report run recorded behavioral Git commit SHA
  `23477441eeb08ad7a5c56bcf1f3fdcb94da9141e` in `MANIFEST.json`.
- The agreement table now explicitly states that it is an aggregate across all
  result rows, not a per-dataset agreement measure.

### Review Verification

- `python3 -m unittest experiments.exp263_264_full_missing_aware_integration.tests.test_reporting -v`
  - Passed: 4 tests.
- `python3 -m unittest discover -s experiments/exp263_264_full_missing_aware_integration/tests -v`
  - Passed: 45 tests.
- `python3 -m py_compile experiments/exp263_264_full_missing_aware_integration/reporting.py experiments/exp263_264_full_missing_aware_integration/tests/test_reporting.py`
  - Passed.
- Production `generate_report(...)`
  - Passed routing-contract rederivation for all 1,117 result rows before TEST
    truth was loaded.
  - Exact summary agreement remains `true`; promoted remains 2 TP / 0 FP and
    quality-blocked remains empty.
  - `MANIFEST.json` contains all seven required code hashes, the available Git
    SHA, and still excludes self-hashing.

### Complete Label-Free Preflight

- `preflight_routing_rows` now validates every persisted result row before the
  scoring pass can call the TEST truth loader. It checks baseline/source/final
  lanes, source-agreement-derived promotions, quality blocks, and the derived
  Standard/Priority routing partition.
- A two-row regression fixture has a valid first row and invalid second row;
  the instrumented truth loader records zero calls when the second-row preflight
  fails.

### Final Verification

- `python3 -m unittest experiments.exp263_264_full_missing_aware_integration.tests.test_reporting -v`
  - Passed: 5 tests.
- `python3 -m unittest discover -s experiments/exp263_264_full_missing_aware_integration/tests -v`
  - Passed: 46 tests.
- `generate_report(...)` after behavioral commit `23477441`
  - Completed the label-free preflight across all 1,117 rows before the separate
    scoring pass.
  - Exact summary agreement remains `true` and the refreshed `MANIFEST.json`
    records Git SHA `23477441eeb08ad7a5c56bcf1f3fdcb94da9141e`.
  - Refreshed manifest SHA256: `ff1d3732d4c1440be3d6104d8fa2da6f1168df3dd470b2827fb0e1bee425d9fa`.
