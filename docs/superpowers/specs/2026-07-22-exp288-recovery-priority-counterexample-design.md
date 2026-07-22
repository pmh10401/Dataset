# Exp288 Recovery Priority Counterexample Diagnostic

## Purpose

Explain the 11 TP and 32 FP among Exp287 alpha 0.05 Recovery Priority
candidates by comparing each TEST candidate with its nearest TRAIN-normal trace.
This is a retrospective diagnostic that reads TEST labels; it cannot generate,
route, threshold, or promote alerts.

## Fixed comparisons

- nearest TRAIN normal by correlation distance after TRAIN-length alignment and
  per-series normalization;
- raw correlation distance;
- absolute log standard-deviation ratio;
- absolute mean shift normalized by the nearest normal robust scale;
- spectral Wasserstein distance;
- fixed 16-dimensional feature distance.

Descriptive categories are frozen before the diagnostic:

- `normal_shape_match`: correlation distance <= 0.05;
- `amplitude_or_level_only`: correlation <= 0.15 and amplitude log-ratio >= 0.5
  or normalized mean shift >= 2;
- `spectral_shift`: spectral distance >= 0.05 and correlation > 0.15;
- `complex_difference`: everything else.

Category thresholds are explanatory only and cannot become policy without a
new preregistered prediction experiment.

## Outputs

- all 43 candidate comparisons with TP/FP labels;
- TP/FP category summaries;
- five diverse TP and five diverse FP representative cases;
- candidate and nearest TRAIN-normal waveforms as CSV;
- a Korean result report with next-step implications.
