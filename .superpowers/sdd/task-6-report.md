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

- Exact summary agreement: `true` for all 1,117 datasets.
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
