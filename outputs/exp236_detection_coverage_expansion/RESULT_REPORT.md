# Exp236 Detection Coverage Expansion Result

## Status

- Experiment: `experiment_236_three_source_hard_two_source_review`
- Datasets: 1,117/1,117
- Prediction SHA-256: `a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524`
- Lane overlap: 0 datasets
- Independent reaggregation: passed
- Scope: retrospective candidate packaging
- Deployment eligible: no

## Frozen Policy

- Automatic alert: Exp137 Hard plus candidates supported by at least three independent score sources.
- General review: Exp137 Standard review plus candidates supported by exactly two sources.
- Priority review: unchanged Exp137 Priority review.

## Autonomous Hard Comparison

| Metric | Exp137 | Exp236 | Change |
|---|---:|---:|---:|
| Hard alerts | 2,005 | 2,116 | +111 |
| TP | 1,691 | 1,789 | +98 |
| FP | 314 | 327 | +13 |
| Precision | 84.339% | 84.546% | +0.207%p |
| Recall | 43.16% | 45.66% | +2.50%p |
| Mean dataset F1 | 0.600772 | 0.604133 | +0.003361 |

The new three-source Hard candidates contain 98 TP and 13 FP, for a
retrospective precision of 88.29%.

## Review Coverage

- New exactly-two-source review candidates: 328
- New review TP: 219
- New review FP: 109
- New review precision: 66.77%
- Total Standard review after union: 967, containing 511 TP and 456 FP
- Priority review remains 9 candidates, containing 8 TP and 1 FP

## Interpretation

Exp236 increases autonomous detection coverage without reducing aggregate Hard
precision on the existing benchmark. It is therefore a stronger structural
candidate than replacing Exp137 with the strict Exp227-233 filters.

However, the minimum-three rule was nominated after examining earlier TEST
results. Exp236 does not provide independent confirmation and must not be
presented as deployment validation. The next valid step is to freeze this exact
policy and evaluate it on newly locked data or prospective shadow feedback.
No threshold or source-count constant was changed after Exp236 evaluation.
