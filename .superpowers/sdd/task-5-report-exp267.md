# Task 5 Report: Exp270-A Lightweight Normal Forecasting

## Status

Implemented the label-free `RFFNormalForecastResidual` source for Exp270-A.
It learns a normal forecasting relationship from TRAIN-normal traces only and
returns one residual-deviation score per TRAIN OOF or TEST trace. No database,
TEST label, family, or dataset-name input is read by the module.

## Files

- `experiments/exp267_273_zero_support_fn_recovery/forecast_rff_source.py`
- `experiments/exp267_273_zero_support_fn_recovery/tests/test_forecast_rff_source.py`
- `.superpowers/sdd/task-5-report-exp267.md`

No existing dirty `.superpowers` files, progress ledger, or unrelated source
files were modified.

## RED Evidence

Before implementation:

```bash
python3 -m unittest \
  experiments.exp267_273_zero_support_fn_recovery.tests.test_forecast_rff_source -v
```

Result: expected import failure because
`forecast_rff_source.py` did not yet exist:

```text
ModuleNotFoundError: No module named
'experiments.exp267_273_zero_support_fn_recovery.forecast_rff_source'
```

## Frozen Implementation Contract

- Eligibility is `n_train >= 20`; lower counts return the explicit,
  label-free `n_train_below_20` abstention.
- The actual TRAIN matrix width determines configuration. Requested contexts
  are `(8, 16, 32)`, clipped to retain a valid forecast origin; the horizon is
  `max(1, min(8, train_target_length // 16))`.
- The RFF map is `sklearn.kernel_approximation.RBFSampler` with 256 components
  and fixed random seed `20260721`. Ridge candidates are exactly
  `(0.01, 0.1, 1.0, 10.0)`.
- Each outer OOF query fold is excluded from model fitting, residual-reference
  fitting, and nested alpha selection. Alpha selection is deterministic
  TRAIN-only cross-fitting with numeric-alpha tie breaking.
- Absolute residuals are aggregated by median, p95, maximum, and longest run
  above the model's TRAIN residual p95. The four aggregates are independently
  robust-standardized using the applicable fit TRAIN rows and summed with
  equal weights.
- One-point traces abstain with `target_length_below_2`; two-point traces use
  the clipped one-value context and remain finite.

## GREEN Evidence

Focused Task 5 suite:

```bash
python3 -m unittest \
  experiments.exp267_273_zero_support_fn_recovery.tests.test_forecast_rff_source -v
```

Result:

```text
Ran 9 tests in 6.486s
OK
```

The tests cover the 20-row eligibility boundary, OOF and nested selection
query exclusion, deterministic finite scores, TRAIN-derived configuration,
constant-trace numerical stability and numeric alpha tie breaking, short trace
handling, TRAIN/TEST width compatibility, and a synthetic late shift producing
a larger residual score.

Package regression suite:

```bash
python3 -m unittest discover \
  -s experiments/exp267_273_zero_support_fn_recovery/tests -v
```

Result:

```text
Ran 50 tests in 14.195s
OK
```

Additional checks passed:

```bash
python3 -m py_compile \
  experiments/exp267_273_zero_support_fn_recovery/forecast_rff_source.py \
  experiments/exp267_273_zero_support_fn_recovery/tests/test_forecast_rff_source.py
git diff --check -- \
  experiments/exp267_273_zero_support_fn_recovery/forecast_rff_source.py \
  experiments/exp267_273_zero_support_fn_recovery/tests/test_forecast_rff_source.py
```

## Limitations

- This task implements a source only. It does not calibrate candidates,
  route review lanes, read TEST labels, or claim retrospective performance.
- Nested TRAIN-only cross-fitting is deliberately more expensive than a single
  in-sample fit. The implementation keeps memory bounded by accumulating RFF
  ridge sufficient statistics rather than materializing all forecast windows.
- The fixed RFF representation, contexts, horizon, ridge grid, and residual
  aggregates are frozen here and must not be changed after TEST evaluation.
