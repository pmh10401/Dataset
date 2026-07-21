# Task 4 Report: Exp269 Ordered Path-Signature Source

## Status

Implemented Exp269's label-free ordered path-signature score source and its
focused tests. The source uses the exact channel order `(value,
first_difference, normalized_time)`, depth `3`, and signatures over the full,
first-half, and second-half trace windows.

## Files

- `experiments/exp267_273_zero_support_fn_recovery/signature_source.py`
- `experiments/exp267_273_zero_support_fn_recovery/tests/test_signature_source.py`
- `.superpowers/sdd/task-4-report-exp267.md`

No existing dirty `.superpowers` files or other experiment sources were edited.

## RED Evidence

Before production implementation:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_signature_source -v
```

Result: exit `1`, as expected. The test import failed with:

```text
ModuleNotFoundError: No module named 'experiments.exp267_273_zero_support_fn_recovery.signature_source'
```

The additional provenance test then failed before its corresponding
implementation change with:

```text
KeyError: 'window_definitions'
```

The alignment regression test also failed before the TRAIN-derived target-length
guard was added:

```text
AssertionError: ValueError not raised
```

## GREEN Evidence

Focused Task 4 suite:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_signature_source -v
```

Result:

```text
Ran 6 tests in 0.046s
OK
```

The tests verify batched feature shape `(batch, iisignature.siglength(3, 3) *
3)`, finite output, reversal/order sensitivity, deterministic extraction,
query-excluded TRAIN OOF scores, `minimum_train=5` abstention, TRAIN/TEST
alignment, and label-free provenance.

Package regression suite:

```bash
python3 -m unittest discover -s experiments/exp267_273_zero_support_fn_recovery/tests -v
```

Result:

```text
Ran 38 tests in 7.656s
OK
```

Additional verification:

```bash
python3 -m py_compile \
  experiments/exp267_273_zero_support_fn_recovery/signature_source.py \
  experiments/exp267_273_zero_support_fn_recovery/tests/test_signature_source.py
git diff --check -- \
  experiments/exp267_273_zero_support_fn_recovery/signature_source.py \
  experiments/exp267_273_zero_support_fn_recovery/tests/test_signature_source.py
```

Both commands completed successfully.

## Implementation Notes

- The feature path is batched as `(batch, time, 3)` and invokes
  `iisignature.sig(path, 3)` once per frozen window.
- `feature_distance_result(..., minimum_train=5)` provides robust distance
  scoring and query-excluded cross-fitted TRAIN OOF scores.
- Provenance records signature depth, channel count and exact order, normalized
  window definitions, expected per-window signature length, total feature
  dimension, `iisignature` version, and `contains_test_labels=False`.

## Self-Review

- No production or test code imports TEST labels or evaluation data.
- All input and output feature values are checked for finiteness.
- A trace must provide at least two points so both half windows are valid.
- Score abstention uses the shared `n_train_below_5` contract and preserves
  label-free provenance.
- The staged commit is restricted to the two Task 4 code/test files and this
  requested report.

## Concerns

- The representation is intentionally fixed at depth 3 and does not tune
  channel, depth, or window selection on TEST outcomes.
- `iisignature.sig` returns `float64`; this prioritizes stable, finite feature
  calculations over reduced feature-memory use.
