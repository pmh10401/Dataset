# Exp305 Wavelet-Reconstruction Cross-confirmation Design

## Goal

Evaluate whether the full-coverage Exp239 wavelet score can confirm the
research-only reconstruction residuals observed in Exp304. The operational Hard
alert set remains frozen; all expansions are retrospective counterfactuals.

## Frozen inputs

- Exp304 frozen predictions: 251 datasets with reconstruction support.
- Exp239 frozen predictions: 1,117 datasets, variant `default`, alpha `0.005`.
- TEST labels are loaded only after the Exp305 prediction file is frozen.

## Policies

- `baseline_no_expansion`: existing automatic Hard alerts.
- `verified_1of4`: at least one existing expert supports a Standard candidate.
- `reconstruction_only_residual`: reconstruction supports the candidate while
  all four existing experts do not.
- `wavelet_only_residual`: wavelet supports the candidate while reconstruction
  and all four existing experts do not.
- `reconstruction_wavelet_cross_confirmed`: reconstruction and wavelet both
  support the candidate while all four existing experts do not.
- `existing_1of4_or_reconstruction_wavelet`: existing 1-of-4 support, or the
  reconstruction-wavelet cross-confirmation above.
- `six_source_2of6`: at least two votes among four existing experts,
  reconstruction, and wavelet.

No policy is selected from TEST performance. Wavelet and reconstruction remain
research evidence until provenance and prospective validation are complete.
