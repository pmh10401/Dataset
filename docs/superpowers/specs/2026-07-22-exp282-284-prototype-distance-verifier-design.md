# Exp282-284 Prototype Distance Verifier Design

## Objective

Build an independent second-stage verifier for the frozen Exp272 Recovery
Shadow and Recovery Priority candidates. Exp253 autonomous Hard alerts remain
unchanged. Every new result is retrospective and review-only.

## Inputs and forbidden information

- Candidate universe: frozen Exp272 Recovery Shadow plus Recovery Priority.
- Fit/calibration input: TRAIN normal values only.
- TEST values are scored only after all TRAIN fit objects and calibration
  references are fixed.
- Prediction cannot use TEST labels, anomaly positions, family, dataset-name
  policy branches, TEST length budgets, TEST order, or previous TEST metrics.

## Distance evidence groups

1. Feature geometry
   - Robust Diagonal Mahalanobis on fixed statistical/dynamic features.
   - Ledoit-Wolf shrinkage Mahalanobis on the same features.
   - These correlated distances form one evidence group, not two votes.
2. Shape
   - Correlation distance on TRAIN-length-aligned, per-series normalized traces.
3. Spectrum
   - Wasserstein distance between normalized one-sided power spectra.

Each dataset uses at most three deterministic TRAIN medoids. Medoid count is
fixed from TRAIN count: one for `n < 10`, two for `10 <= n < 30`, and three for
`n >= 30`. Distance is the minimum distance to any medoid.

## TRAIN-only calibration

For every TRAIN normal instance, leave it out, fit scaling/covariance/prototypes
on the remaining TRAIN normals, and score the held-out instance. TEST candidate
distances are converted to empirical upper-tail p-values against these
leave-one-out normal distances. `alpha=0.05` is frozen before evaluation.
Datasets with fewer than 20 TRAIN normals abstain because `alpha=0.05` is not
reliably expressible after leave-one-out calibration.

Shrinkage Mahalanobis uses Ledoit-Wolf precision. Non-finite precision,
condition number above `1e8`, or fit failure triggers a recorded TRAIN-only
fallback to Robust Diagonal Mahalanobis; it never silently drops a dataset.

## Frozen variants

- `strict_geometry_shape_spectrum`: both Mahalanobis distances anomalous, plus
  shape and spectrum anomalous.
- `balanced_consensus_2of3`: both Mahalanobis distances form geometry evidence,
  and at least two of geometry/shape/spectrum agree.
- `sensitivity_any_mahal_2of3`: either Mahalanobis distance forms geometry
  evidence, and at least two of the three evidence groups agree.

No variant is selected using TEST performance. Exp284 reports all variants.
Precision at least 80% is only a future review-policy qualification condition,
not automatic deployment or Hard promotion.
