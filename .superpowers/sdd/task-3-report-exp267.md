# Task 3 Report: Exp268 Multiscale Scattering Source

## Scope

- Added `MultiscaleScatteringDeviation` backed by `kymatio.numpy.Scattering1D`.
- Added focused tests for batch shape, finiteness, determinism, OOF scoring,
  minimum-train abstention, train-derived configuration, and test-width rejection.

## RED Evidence

Command:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_scattering_source -v
```

Before implementation, the test module failed to import with:

```text
ModuleNotFoundError: No module named 'experiments.exp267_273_zero_support_fn_recovery.scattering_source'
```

During self-review, the train-width regression initially failed because the
source used `DatasetBundle.target_length`; the extractor rejected the 128-wide
TRAIN rows when metadata said 64.

## GREEN Evidence

Command:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_scattering_source -v
python3 -m py_compile experiments/exp267_273_zero_support_fn_recovery/scattering_source.py experiments/exp267_273_zero_support_fn_recovery/tests/test_scattering_source.py
git diff --check
```

Result: 6 tests passed; compilation and whitespace validation succeeded.

## Implementation Notes

- The source derives `J` and all frozen transform configuration from
  `train_normal.shape[1]` only. Test rows with a different width are rejected.
- Kymatio receives batches shaped `(batch, time)` and its coefficients are
  reduced over the residual time axis using log-absolute mean, standard
  deviation, p90, and maximum. Output is finite and fixed-width per target
  length.
- `feature_distance_result(..., minimum_train=5)` provides query-excluded TRAIN
  OOF robust-distance scores. The adapter emits the local immutable
  `SourceScore` and never accepts or records labels.
- Provenance includes the Kymatio configuration, computed scattering feature
  width, cross-fit status, and label-free marker.

## Concerns

- Kymatio 0.3.0's NumPy namespace imports the removed `scipy.special.sph_harm`
  name under installed SciPy 1.17.1. The source now installs a temporary
  legacy-signature wrapper only while importing Kymatio, then restores or
  removes the attribute in `finally`; the Exp268 path itself uses only 1D
  scattering. This should be removable after a Kymatio/SciPy compatibility
  upgrade.
- Kymatio warns about possible border effects at the required short-series
  configuration. The known construction warning is suppressed locally because
  the mandated `J` formula is retained; no signal padding or data-dependent
  parameter selection is introduced.

## Reviewer-Fix Evidence

### RED

After adding the SciPy regression tests, the focused suite failed with three
expected `AttributeError` results because `_legacy_sph_harm` and
`_load_scattering1d` did not yet exist.

### GREEN

Command:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_scattering_source -v
python3 -m py_compile experiments/exp267_273_zero_support_fn_recovery/scattering_source.py experiments/exp267_273_zero_support_fn_recovery/tests/test_scattering_source.py
git diff --check
```

Result: 9 tests passed; compilation and whitespace validation succeeded.

The wrapper is verified against SciPy's current `sph_harm_y(n, m, theta, phi)`
semantics using legacy input order `(m, n, theta, phi)`. Regression tests prove
the temporary attribute is visible during the Kymatio import, removed when
absent beforehand, and restored unchanged when it already existed.

## Final Angle-Semantics Correction

### RED

The legacy wrapper test was changed to independently expect
`sph_harm_y(n, m, phi, theta)` using non-symmetric values
`theta=1.23` and `phi=0.47`. The previous no-swap implementation failed with a
complex-value difference of approximately `0.5408`.

### GREEN

The wrapper now maps legacy `(m, n, theta_azimuth, phi_polar)` to SciPy's
current `(n, m, theta_polar, phi_azimuth)` as
`sph_harm_y(n, m, phi, theta)`. The focused suite passed all 9 tests after this
correction, and the source and test modules compiled successfully.
