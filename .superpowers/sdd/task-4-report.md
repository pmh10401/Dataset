# Task 4 Report: Exp263 Full Prediction Artifact

## Implementation SHA

`49aa572c` (`feat(exp263): freeze full missing-aware predictions`)

## Files

- `experiments/exp263_264_full_missing_aware_integration/prediction.py`
- `experiments/exp263_264_full_missing_aware_integration/tests/test_prediction_evaluation_boundary.py`

## Commands And Results

- `python3 -m unittest experiments.exp263_264_full_missing_aware_integration.tests.test_inputs experiments.exp263_264_full_missing_aware_integration.tests.test_routing experiments.exp263_264_full_missing_aware_integration.tests.test_prediction_evaluation_boundary -v`
  - 31 tests passed.
- `python3 -m py_compile experiments/exp263_264_full_missing_aware_integration/prediction.py experiments/exp263_264_full_missing_aware_integration/tests/test_prediction_evaluation_boundary.py`
  - Passed.
- `! rg -n "y_test|load_test_labels_for_evaluation|evaluation import" experiments/exp263_264_full_missing_aware_integration/prediction.py`
  - Passed; the prediction boundary has no TEST-label or evaluation-loader import.
- `predict_all(OUTPUT_ROOT)`
  - Wrote `/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/frozen_predictions.jsonl`, `prediction_manifest.json`, and `source_agreement_diagnostics.csv`.
- Persisted artifact verification
  - 1,117 rows, 1,117 unique names, sorted dataset order, and 1,117 diagnostics rows.
  - Manifest SHA matches the frozen JSONL: `8c7a2f52027c6a3127c79166bb109b61e348558574d98a42a8ccf5622921d653`.
  - All final lanes are disjoint and in range; every baseline Hard index remains Hard.
  - All 20 missing rows reproduce Exp260 `n_train`, `n_test`, four source maps, TRAIN-q99 flags, quality-blocked indices, and final routing fields exactly.

## Coverage

- Full frozen population: 1,117 datasets.
- Missing-aware override: exactly 20 datasets.
- Verified nonmissing datasets: 1,097, with empty quality flags.
- Source agreement diagnostics: one deterministic CSV row per dataset.

## Concerns

- This task is prediction-only. It freezes the SHA before the future evaluation stage; it does not read TEST labels or calculate evaluation metrics.
