# Task 5 Report: Exp270-A Lightweight Normal Forecasting

## Status

Implemented the label-free `RFFNormalForecastResidual` source for Exp270-A and
subsequently corrected its P1 operational runtime issue. It learns a normal
forecasting relationship from TRAIN-normal traces only and returns one
residual-deviation score per TRAIN OOF or TEST trace. No database, TEST label,
family, or dataset-name input is read by the module.

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
Ran 10 tests in 4.378s
OK
```

The tests cover the 20-row eligibility boundary, OOF and nested selection
query exclusion, deterministic finite scores, TRAIN-derived configuration,
constant-trace numerical stability and numeric alpha tie breaking, short trace
handling, TRAIN/TEST width compatibility, and a synthetic late shift producing
a larger residual score. The runtime regression test additionally proves from
source provenance that only five global fold-group statistics are built, every
TRAIN row is transformed once per context for those statistics, all four alpha
candidates share the same validation transforms, and the expected 16 cached
coefficient families use 48 eigendecompositions for three contexts.

Package regression suite:

```bash
python3 -m unittest discover \
  -s experiments/exp267_273_zero_support_fn_recovery/tests -v
```

Result:

```text
Ran 51 tests in 10.961s
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

## P1 Runtime Correction

The original implementation rebuilt RFF sufficient statistics for each nested
fit subset and re-transformed each validation window once per alpha candidate.
That behavior was statistically valid but operationally too expensive.

The corrected implementation keeps the frozen policy unchanged:

- it builds one deterministic five-fold TRAIN-row partition;
- it streams each TRAIN row once per context into its fold group's additive
  `X'X` and `X'y` statistics, with no per-row `256 x 256` allocation;
- every outer or nested fit is created by combining only the eligible fold
  groups, so its query group is absent from both model fitting and alpha
  selection;
- one symmetric eigendecomposition per context produces coefficient vectors
  for all four frozen ridge alphas; and
- one RFF validation transform evaluates those four vectors in a batch.

The full TRAIN cross-fit alpha loss is computed once and reused for the final
TEST model alpha. The frozen grid and numeric tie break remain exactly
`(0.01, 0.1, 1.0, 10.0)` and lowest-loss-then-smallest-alpha.

### Local Benchmark

The benchmark used the same deterministic synthetic normal bundle before and
after the change: `n_train=100`, `target_length=256`, four TEST traces, seed
`20260721`, Apple Silicon local runtime.

```text
pre-optimization commit ac8e08aa: 8.391758 seconds
optimized working tree:             4.459730 seconds
reduction:                          46.9 percent
```

The optimized run recorded 300 streaming TRAIN row-context statistic
transforms (`100 rows x 3 contexts`), 25 batched all-alpha validation calls,
and 48 eigendecompositions (`16 cached fold combinations x 3 contexts`). This
is a diagnostic benchmark, not a timing test threshold and not a performance
claim for every dataset length.

## Limitations

- This task implements a source only. It does not calibrate candidates,
  route review lanes, read TEST labels, or claim retrospective performance.
- Nested TRAIN-only cross-fitting remains more expensive than a single
  in-sample fit because TRAIN-only residual p95 thresholds must be recomputed
  for each distinct fit set. The implementation avoids per-row Gram matrices
  and avoids RFF statistics or validation transforms repeated for alpha
  candidates.
- The fixed RFF representation, contexts, horizon, ridge grid, and residual
  aggregates are frozen here and must not be changed after TEST evaluation.

## Second Review Correction

The second review identified two P1 issues: the optimized path had replaced
the original standardization references with nested OOF features, and exact
TRAIN residual p95 calculation could retain an unbounded in-memory residual
array. Both are corrected before any TEST evaluation.

### Restored Score Semantics

- For every OOF outer query fold, aggregate center and scale now come from
  trace features scored by that fold's selected outer model on its
  `outer_fit_rows`. The query fold is excluded from both the model and this
  reference.
- For TEST scoring, aggregate center and scale now come from trace features
  scored by the full model on all TRAIN-normal rows.
- Provenance names are explicit: `oof_standardization_reference`,
  `test_standardization_reference`,
  `test_full_model_train_aggregate_centers`, and
  `test_full_model_train_aggregate_scales`.

### Exact Bounded-RAM Residual p95

`RFF_MAX_TRAIN_RESIDUAL_VALUES = 10_000_000` is frozen in `config.py` and
locked by the contract tests. Before TEST input is read or any RFF transform is
started, the source estimates the full-model residual count from only
`n_train`, the actual TRAIN target length, contexts, and horizon. An estimate
above the limit returns the label-free abstention
`train_residual_work_exceeds_10000000`.

For eligible runs, residuals are streamed to a temporary float64 memmap under
the supplied artifact directory. The implementation uses in-place partition at
the two linear-interpolation ranks, matching `np.quantile(...,
method="linear")` exactly, then closes and deletes the spool on both success
and failure. The 10,000,000-value ceiling represents about 80 MB of raw
float64 spool capacity. It is a preregistered operational resource boundary,
not a TEST-performance decision. The TCN source remains available for larger
eligible datasets.

### Verification

Focused RFF and contract suites:

```text
Ran 26 tests in 5.858s
OK
```

Package suite:

```text
Ran 56 tests in 14.583s
OK
```

New tests verify exact interpolated quantiles, temporary-file cleanup after
success and a generator failure, pre-transform safety abstention, restored
outer/full standardization references, and an end-to-end smoke path with a
bounded spool estimate. Existing query-exclusion, deterministic, short-trace,
and constant-trace tests continue to pass.

### End-to-End Resource Smoke

The deterministic local smoke benchmark used `n_train=100`, TRAIN length 256,
four TEST traces, and seed `20260721`. It is observational only; the tests do
not assert a wall-clock threshold.

```text
elapsed seconds:          4.198668
Python traced peak bytes: 13,544,783
process ru_maxrss:        203,456,512
TRAIN residual estimate:  69,100 values
temporary spools left:    0
finite scores:            true
```

`ru_maxrss` is process-wide and platform-dependent. The meaningful safety
contract is the frozen 10,000,000-value spool cap plus verified cleanup, not a
peak-RSS deployment guarantee.
