# Exp274-275 Large-TRAIN FN Recovery Design

## Objective

Exp274-275 tests whether two frozen structural sources can improve the review
quality and coverage of Exp272 without changing the Exp253 autonomous Hard
policy.  All results are retrospective counterfactual evidence.

## Frozen Inputs

- Exp272 recovery prediction: primary alpha `0.005`.
- Exp239 multiscale wavelet energy prediction: variant `default`, alpha `0.005`.
- Exp254 structural prediction: variant `matrix_profile_16`, alpha `0.005`.
- `n_train` from the label-free Exp272 prediction row.

Every input prediction SHA256 must match its manifest before candidate routing.
Prediction does not open TEST labels.

## Exp274: Large-TRAIN Multisource Recovery

The eligibility rule is `n_train >= 100`.  TEST length, TEST labels, family,
and dataset name are not policy inputs.

Two pre-registered review-only variants are emitted:

1. `forecast_wavelet_matrix`: an outside-lane one-source candidate whose
   existing support is the normal-forecast group and which is also selected by
   Wavelet and Matrix Profile.
2. `any_source_wavelet_matrix`: an outside-lane one-source candidate selected
   by both Wavelet and Matrix Profile, regardless of the identity of that one
   existing support source.

The second variant is a sensitivity result.  TEST performance cannot select
between the variants.

## Exp275: Recovery Shadow Precision Guard

Two pre-registered filtered views of the current Recovery Shadow are emitted:

1. `shadow_either_structure`: current Recovery Shadow confirmed by Wavelet or
   Matrix Profile.
2. `shadow_both_structures`: current Recovery Shadow confirmed by both sources.

These are review-only precision diagnostics.  Neither variant is promoted to
Hard, even if retrospective precision exceeds 80%.

## Metric Boundary

- Autonomous metrics always refer to unchanged Exp253 Hard alerts.
- New candidates are human-assisted review candidates.
- Combined F1 is diagnostic only.
- Actual TEST labels are read only after the new prediction hash is frozen and
  verified.

## Prohibited Changes

- no TEST-length gate or top-N budget;
- no family or dataset-name routing;
- no TEST-selected alpha, source, threshold, or variant;
- no tail replacement;
- no Priority-to-Hard promotion;
- no mutation of Exp239, Exp254, Exp272, or Exp253 artifacts.

