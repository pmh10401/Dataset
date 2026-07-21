# Task 6 Report: Exp270-B Causal TCN Forecasting

## Scope

- Added `forecast_tcn_source.py` with the frozen TRAIN-normal causal TCN source.
- Added focused causality, eligibility, fold, seed, device-decision, and bounded-batch tests.
- No DB rows, TEST labels, family names, dataset names, TEST length policy, alert budget, or routing code is read by this task.

## RED Evidence

Before implementation:

```text
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_forecast_tcn_source -v
```

failed with nine `ModuleNotFoundError` / import failures because
`forecast_tcn_source.py` did not yet exist.

## Frozen Implementation

- Four causal residual `Conv1d` blocks; 32 channels; kernel size 3; dilations
  `(1, 2, 4, 8)`; GELU; dropout `0.1`.
- An eight-point future patch is produced from every causal hidden location;
  the final location is therefore a forecast from the final causal hidden state.
- Training uses AdamW (`lr=0.001`, `weight_decay=0.0001`), batch size 32,
  Huber loss, at most 20 epochs, and patience 3.
- The two frozen seeds are `20260721` and `20261721`.
- Every eligible source makes deterministic five-fold TRAIN-only OOF scores,
  then refits on all TRAIN-normal rows for TEST-value scoring.
- `seed_median` is the emitted scalar score. Per-seed vectors are retained in
  label-free provenance so Task 7 can freeze `seed_intersection` as the
  conjunction of both independently calibrated seed p-values. Both variants
  are registered as one `normal_forecast` evidence family.
- `n_train < 100` returns explicit `n_train_below_100` abstention. Trace width
  below nine returns explicit `target_length_below_9` abstention.

## Device Decision Contract

`benchmark_device()` runs exactly one fixed synthetic 100-row benchmark with
three warm iterations and five timed iterations. MPS can be selected only when
it is available, finite, numerically close to CPU (`rtol=1e-4`, `atol=1e-5`),
and has a lower median runtime. The JSON decision is cacheable at
`tcn_device_benchmark.json`; source scoring never reruns that benchmark per
dataset. The later pipeline must run and freeze this decision before full TCN
prediction.

Local smoke benchmark:

```text
CPU median: 0.02784554206300527 s
MPS median: 0.0015127919614315033 s
decision: MPS
```

The test data and model parameters are identical across the CPU and MPS runs;
the MPS decision was made before prediction and does not use labels.

## Runtime And Memory Assessment

The model has 12,744 parameters. A full-width no-grad forward pass with the
largest normal batch size tested (`32 x 3,992` causal inputs, equivalent to
target length 4,000) produced finite `(32, 8, 3,992)` outputs:

```text
CPU: 0.2535470830043778 s
MPS: 0.2864375408971682 s
```

The TRAIN-only provenance estimates 86,354,944 activation bytes for a
32-row, length-4,000 batch. The implementation does not sample TRAIN rows,
resample trace length, or create a residual spool. It keeps device work bounded
to 32 rows at a time and releases model/device cache after every OOF fold and
final fit.

This does **not** make the total full-population training time constant. A
large dataset such as 18,378 TRAIN rows by length 4,000 still requires two
seeds times five OOF fits plus two final fits, each capped at 20 epochs. No
arbitrary runtime cap was added because none is frozen in the plan. Before a
full queue run, Task 7/12 must use the one-active-run policy, record elapsed
time per dataset, and treat a real measured long-run stall as a blocker rather
than silently sampling or changing the TCN.

## GREEN Evidence

Focused test command:

```text
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_forecast_tcn_source -v
```

Result: 10 tests passed.

Package verification:

```text
python3 -m py_compile experiments/exp267_273_zero_support_fn_recovery/*.py
python3 -m unittest discover -s experiments/exp267_273_zero_support_fn_recovery/tests -v
git diff --check
```

Result: 66 tests passed; compilation and whitespace checks passed.

An additional real TRAIN-only smoke fit with 100 rows, target length 16, and
two TEST-value rows completed with 100 finite OOF scores, two finite TEST
scores, and zero fit/query overlap for both seeds. No evaluation metric or
label was produced or read.

## Limitations

- This task implements a source only. It does not select alpha, calibrate
  candidates, route alerts, read TEST labels, or evaluate performance.
- The output is retrospective research infrastructure. It is neither an
  autonomous policy activation nor prospective/deployment validation.
- Full-population TCN cost must be measured by the later sequential pipeline;
  this task intentionally does not introduce a post hoc compute eligibility
  rule.

## P1 Follow-up: Frozen MPS Wiring And TRAIN-only Compute Guard

### MPS Wiring

`CausalTCNNormalForecast` now accepts `benchmark_record` at construction and
calls `choose_device(requested, benchmark_record)`. A direct `device="mps"`
request with no record, incomplete record, failed parity, non-finite result, or
non-faster result stays on CPU. The complete record, its canonical SHA-256, and
the final selected device are stored in every source provenance row. The source
does not call `benchmark_device()`; Task 7 must freeze the single benchmark
record once and pass it to all eligible dataset jobs.

Real smoke evidence used a passing frozen record:

```text
benchmark selected device: mps
source selected device: mps
TRAIN OOF scores: 100 finite values
TEST-value scores: 2 finite values
```

### Pre-registered Compute Guard

`TCN_MAX_TRAIN_ELEMENTS = 5_000_000` is frozen in `config.py`. The workload is
exactly `n_train_normal * TRAIN target_length`; it is checked from TRAIN rows
before any TEST-value array conversion, validation, model fitting, or score
calibration. Exceeding it returns:

```text
abstention_reason = train_work_exceeds_5000000
```

with both the observed TRAIN element count and the frozen threshold in
provenance. It does not inspect a dataset name, family, TEST size, TEST label,
or TEST metric. The frozen metadata audit rationale is that the eligible
population has 252 of 253 `n_train >= 100` datasets at or below this boundary;
the one larger workload is recorded as an explicit abstention rather than a
silent omission.

### Deterministic Runtime Estimate

The frozen benchmark now supplies a linear, inference-only lower-bound report
for each eligible source record. It includes per-batch timing, 12 maximum fits
(two seeds times five OOF fits plus final fits), 20 maximum epochs, batch count,
and the resulting forward-only lower bound. It explicitly excludes backward
pass and IO, so it is a scheduling diagnostic rather than a completion-time
promise.

For the large `18,378 x 4,000` TRAIN shape with an MPS benchmark median of
0.0015 seconds per synthetic 100-row, length-128 batch, the deterministic
forward-only lower bound is 2,203.584 seconds for 138,000 maximum update
steps. Actual full fit time will be higher because backward and data transfer
are excluded.

No checkpoint/resume hooks were added inside the source: a model checkpoint
without the future cache/hash contract would be unsafe. Task 7's atomic
per-dataset cache is the correct layer for restartability.

### Follow-up Verification

```text
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_contracts \
  experiments.exp267_273_zero_support_fn_recovery.tests.test_forecast_tcn_source -v
```

Result: 24 tests passed, including passing/unverified MPS selection and a
forbidden TEST-value object proving the compute guard exits before TEST access.
