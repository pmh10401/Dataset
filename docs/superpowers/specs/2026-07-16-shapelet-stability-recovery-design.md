# Shapelet Stability Recovery Design

## Goal

Determine whether the calibration-transport failure of the normal-pattern
prototype score is caused mainly by its random dictionary, then test a
deterministic normal-pattern replacement and certify its normal-only alert
risk without changing Exp137 autonomous alerts.

## Fixed Scope

- Exp215 is a diagnostic control: a fixed, data-independent dictionary is used
  by every fold and by the full fit.
- Exp216 is the representation candidate: deterministic medoids are built
  only from normal TRAIN windows and re-fit inside each fold.
- Exp217 is a normal-only risk audit of the frozen Exp215/216 prediction
  variants using held-out TRAIN-normal folds and one-sided Wilson bounds.
- All TEST labels and anomaly positions remain evaluation-only.
- TEST length, dataset name, family, and previous TEST performance must not
  alter feature fitting, calibration, thresholding, or candidate selection.
- ALPHAS remain `(0.005, 0.01, 0.02, 0.05)` and no result selects an alpha.
- New candidates remain research-only; Hard alert, review lane, and Exp137
  outputs remain unchanged.

## Design

### Exp215: fixed-dictionary control

Generate 32 normalized analytic kernels from only a frozen seed and the four
fixed lengths `(8, 16, 32, 64)`. They do not use any dataset values. Use the
same dictionary in all folds and full fit, then score kernel-distance features
with the existing train-fitted robust score. This isolates random
normal-subsequence sampling from calibration transport.

Pass gate: at least 201 of the 251 cohort datasets pass the existing stability
contract. Failure is still valuable: it means dictionary randomness is not the
main explanation.

### Exp216: deterministic normal-consensus medoids

For each fit fold, form a deterministic candidate window pool from normal TRAIN
series using four fixed lengths and eight evenly spaced starts. Standardize each
window, rank candidates by their median nearest-neighbour distance, retain the
32 most central windows with deterministic tie breaking, and use them as the
dictionary. The full fit uses the same rule on all normal TRAIN rows.

This is a lightweight normal-prototype approximation: it chooses recurrent,
central normal shapes instead of random shapes. It does not use labels or
synthetic anomalies. The dictionary is rebuilt inside each fit fold; no heldout
normal rows enter a fold dictionary.

Pass gate: at least 201 datasets pass the stability contract and the Exp217
normal-risk audit has no source variant whose Wilson upper bound is silently
reported as an operational guarantee.

### Exp217: normal-only risk audit

For every Exp215/216 fold model, use only its held-out normal fold as a
pseudo-query. The fit-fold normal scores calibrate the candidate decision.
Aggregate false-alert counts across held-out normal rows and report empirical
rate and one-sided 95% Wilson upper bound for every fixed alpha. This is an
empirical TRAIN-only diagnostic, not a formal conformal coverage guarantee.

## Decision Rules

- Exp215 determines whether random dictionary mismatch explains the old audit.
- Exp216 is considered only if its stability gate passes.
- Exp217 never selects alpha or promotes an alert policy. It records risk for
  later human policy review.
- TEST metrics are computed only after frozen predictions and are labelled
  retrospective research metrics.

## Tests

- Fixed dictionary is identical across repeated calls and ignores data values.
- Medoid dictionary is deterministic and changes only with fit-fold values.
- Held-out rows do not affect a fold dictionary.
- Prediction payload has no evaluation fields or test labels.
- Every candidate index is unique and in range.
- Exp217 receives only normal TRAIN folds and no TEST array.

