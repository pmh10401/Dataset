# Task 5 Report: Exp264 Post-Hoc Evaluation

## Source And Prediction SHA

- Committed code/artifact baseline: `f91bb53d` (`docs(exp263): record task 4 prediction report`)
- Frozen prediction SHA256: `8c7a2f52027c6a3127c79166bb109b61e348558574d98a42a8ccf5622921d653`
- Artifact: `/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/frozen_predictions.jsonl`

## Files

- `experiments/exp263_264_full_missing_aware_integration/evaluation.py`
- `experiments/exp263_264_full_missing_aware_integration/tests/test_prediction_evaluation_boundary.py`

## Commands And Results

- `python3 -m unittest experiments.exp263_264_full_missing_aware_integration.tests.test_prediction_evaluation_boundary -v`
  - Initial TDD run failed because `evaluation.py` did not exist.
- `python3 -m unittest discover -s experiments/exp263_264_full_missing_aware_integration/tests -v`
  - 34 tests passed.
- `python3 -m py_compile experiments/exp263_264_full_missing_aware_integration/evaluation.py experiments/exp263_264_full_missing_aware_integration/tests/test_prediction_evaluation_boundary.py`
  - Passed.
- `evaluate_all(frozen_predictions.jsonl, expected_sha256=<manifest SHA>)`
  - SHA256 was verified before the lazy TEST-label loader was imported or called.
  - Wrote `/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/results.csv`, `summary.json`, and `lane_transitions.csv`.
  - Verified 1,117 result rows, 1,117 lane-transition rows, and a summary SHA equal to the frozen prediction SHA.

## Actual Metrics

| Lane | Alerts | TP | FP | FN | Micro precision | Micro recall | Macro mean F1 | Zero-F1 datasets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp253 baseline Hard | 2,253 | 1,915 | 338 | 2,003 | 0.849978 | 0.488770 | 0.608659 | 356 |
| Exp264 autonomous Hard | 2,255 | 1,917 | 338 | 2,001 | 0.850111 | 0.489280 | 0.608711 | 356 |
| Standard review | 806 | 365 | 441 | 3,553 | 0.452854 | 0.093160 | 0.115163 | 901 |
| Priority review | 31 | 26 | 5 | 3,892 | 0.838710 | 0.006636 | 0.007039 | 1,097 |
| Standard plus Priority review | 837 | 391 | 446 | 3,527 | 0.467145 | 0.099796 | 0.119979 | 891 |

All lanes use the same 3,918 TEST anomalies. Zero precision, recall, or F1 denominators are explicitly emitted as `0.0`.

### Paired Exp253 Comparison

- Autonomous Hard versus Exp253 baseline Hard: `+2` alerts, `+2` TP, `0` FP, `-2` FN.
- Dataset-level macro mean F1 delta: `+0.0000519568`.
- F1 improved for 2 datasets, was unchanged for 1,115, and worsened for 0.
- Zero-F1 dataset delta: `0`.

### Diagnostic Subgroups

| Group | Datasets | Hard alerts | Hard TP | Hard FP | Hard precision | Hard recall | Hard macro mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Missing override | 20 | 34 | 22 | 12 | 0.647059 | 0.271605 | 0.224960 |
| Nonmissing | 1,097 | 2,221 | 1,895 | 326 | 0.853219 | 0.493875 | 0.615707 |
| `n_train < 100` | 866 | 684 | 576 | 108 | 0.842105 | 0.610169 | 0.635104 |
| `n_train >= 100` | 251 | 1,571 | 1,341 | 230 | 0.853596 | 0.450908 | 0.517650 |

Every subgroup is marked `used_for_policy_selection: false` in `summary.json`.

## Concerns

- This is retrospective post-hoc evaluation: TEST labels are intentionally read only after prediction SHA verification, but the result is not prospective or end-to-end strict TRAIN-only evidence.
- Exp253 retains the `selected_from_prior_test_review=True` provenance limitation, so `deployment_eligible` remains `false`.
- The two Hard additions produced true positives without new false positives, but the aggregate F1 change is small. No threshold, alpha, source variant, agreement count, or `n_train` rule was changed after these results.
- The missing-override subgroup has substantially lower autonomous-Hard precision than the nonmissing subgroup. It is a diagnostic observation only, not a trigger for policy adjustment in this task.
