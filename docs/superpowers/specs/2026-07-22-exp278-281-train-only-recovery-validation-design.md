# Exp278-281 TRAIN-only Recovery Validation Design

## Goal

Test four follow-up ideas without changing Exp253 autonomous Hard alerts:

1. audit whether TRAIN sample size can support fixed false-alert risk bounds;
2. measure candidate persistence under deterministic TRAIN calibration resampling;
3. audit whether event merging is semantically valid for this database;
4. freeze and retrospectively evaluate a two-stage TRAIN-only review verifier.

## Non-negotiable boundaries

- All 1,117 datasets remain in the population.
- Exp253 Hard alerts remain immutable.
- Prediction code cannot read TEST labels, true anomaly positions, family names,
  past family TEST performance, or TEST order as a policy input.
- TEST length cannot determine thresholds, candidate budgets, or routes.
- Priority and Shadow candidates remain human-review candidates.
- All thresholds and variants are frozen before Exp281 evaluation.
- Results are retrospective and cannot establish prospective deployment quality.
- Tail replacement and adjacent-index promotion are prohibited.

## Exp278: TRAIN risk-resolution audit

For every Exp272 Recovery Shadow and Recovery Priority candidate with continuous
new-source evidence, calculate:

- TRAIN reference count;
- empirical attainable p-value `1 / (n + 1)`;
- one-sided 95% Wilson upper bound already frozen by Exp270;
- whether each source can express risk targets `0.005`, `0.01`, and `0.02`;
- dataset-level maximum source risk for each candidate.

This stage is label-free and diagnostic. It does not select a risk target.

## Exp279: deterministic calibration stability

Model fold outputs are not available for every source, so this experiment must
not claim model-fold stability. It measures calibration-reference stability.
For each source, create five deterministic 80% subsamples of TRAIN OOF scores
using seeds `20260722..20260726`. Recalculate empirical p-values and record the
fraction of replicates in which each candidate remains below `alpha=0.005`.

For TCN, the existing two independently trained seed score vectors are also
checked. The final stability score is the mean available source-group
calibration stability. No TEST label is used.

## Exp280: event-merging applicability audit

One TEST index denotes one complete virtual wafer-run trace, not a time point
inside a trace. Therefore neighboring indices cannot be merged into one event.
Exp280 verifies the sample grain from the database and confirms whether the
same `(dataset, candidate_index)` can appear more than once across final lanes.
If the grain remains instance-level and lane overlap is zero, event merging is
declared inapplicable and no prediction is changed.

## Exp281: two-stage TRAIN-only review verifier

Candidate generation is the frozen union of Exp272 Recovery Shadow and Recovery
Priority. Stage one keeps the existing independent source agreement. Stage two
uses only Exp278 risk and Exp279 stability.

All variants are sensitivity results; none is selected using TEST performance:

- `risk_001_stability_080`: maximum TRAIN risk <= 0.01 and stability >= 0.80;
- `risk_002_stability_080`: maximum TRAIN risk <= 0.02 and stability >= 0.80;
- `risk_002_stability_060`: maximum TRAIN risk <= 0.02 and stability >= 0.60.

Candidates must have at least two continuous independent evidence groups.
Prediction is frozen before labels are read. Evaluation reports candidate TP,
FP, precision, recall, macro F1, workload, and overlap with existing lanes.
Autonomous Hard metrics are reported separately and remain unchanged.

## Success interpretation

A variant may only become a future prospective review-policy candidate if its
retrospective precision is at least 0.80 and it adds true positives. It is not
automatically promoted. Any autonomous promotion requires a new preregistered
experiment and prospective validation.
