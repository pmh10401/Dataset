# Exp236 Detection Coverage Expansion Preregistration

## Purpose

Exp236 packages the strongest retrospective Exp234 expansion signal into one
frozen operating-policy candidate without changing Exp137's existing Hard
alerts.

## Frozen Inputs

- Exp137 Hard, Standard review, and Priority review lanes.
- Exp234 frozen prediction SHA-256:
  `6a7ebfd2f6a90982ff9d1103e4dcc21ecfb9d07ee3afd67e8c2225ac3fe94d60`.
- Exp234 candidates are outside every Exp137 lane by construction and are
  generated from TRAIN-calibrated score agreement without TEST labels.

## Frozen Rule

- Automatic alert: Exp137 Hard union candidates supported by at least three
  independent score sources.
- General review: Exp137 Standard review union candidates supported by exactly
  two score sources.
- Priority review: unchanged Exp137 Priority review.
- No alert: every other index.

The constants `three` and `two` are frozen before Exp236 execution. They must
not be changed after observing Exp236 metrics.

## Information Boundary

Prediction receives frozen lane files only. It does not receive TEST labels,
anomaly positions, family names, past family performance, or TEST-derived
thresholds. Evaluation loads TEST labels only after prediction JSONL is frozen
and its SHA-256 is verified.

## Interpretation Restriction

The three-source rule was nominated after reviewing prior retrospective TEST
results. Therefore Exp236 is a retrospective candidate-packaging and
consistency experiment, not independent confirmation. It cannot be promoted to
deployment until it passes a newly locked holdout or prospective shadow
validation. Priority review remains review-only.

## Success Checks

- 1,117/1,117 dataset coverage and zero duplicate datasets.
- Hard, Standard review, and Priority review lanes are mutually exclusive.
- Exp137 Hard is a subset of Exp236 Hard for every dataset.
- New Hard equals the Exp234 minimum-three set exactly.
- New Standard review equals Exp234 minimum-two minus minimum-three exactly.
- Prediction contains no labels or evaluation metrics.
- Independent reaggregation matches the stored summary.
