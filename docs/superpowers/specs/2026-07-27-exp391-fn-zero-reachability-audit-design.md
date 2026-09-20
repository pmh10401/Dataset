# Exp391 FN/Zero-F1 Reachability Audit Design

## Purpose

Exp391 measures why the frozen Exp389/390 automatic Hard lane misses 2,003
known anomalies. It does not change automatic alerts, review lanes, Alpha,
thresholds, or score sources.

## Inputs

- Frozen Exp389 predictions and their SHA256 manifest.
- Frozen Exp357 evidence for Alpha values
  `0.005, 0.010, 0.020, 0.050, 0.100, 0.200`.
- Exp381 TRAIN-normal coverage metadata.
- TEST labels loaded read-only from `univariate_ts.db` only after every frozen
  prediction/evidence hash has been recorded.

## Dataset Categories

- `hard_has_tp`: the Hard lane contains at least one true anomaly.
- `zero_no_hard`: F1 is zero because the Hard lane is empty.
- `zero_wrong_hard`: F1 is zero because Hard alerts exist but all are false.

Results are also split by TEST size, with `test_ge_100` reported explicitly,
and by TRAIN-normal count.

## Reachability Rules

For each preregistered Alpha, the nine frozen evidence sources cast one vote
per TEST index. Exp391 reports fixed vote thresholds `1`, `2`, and `3`.

Candidate indices already in the Hard lane or exact-normal lane are excluded.
Existing Standard/Priority review overlap is reported separately. TEST labels
are used only after candidates are frozen in memory to calculate:

- recovered FN;
- added FP;
- retrospective candidate precision;
- novel recovered FN outside existing review lanes.

No Alpha, vote threshold, source, size bin, or family is selected from TEST
performance.

## Outputs

- Dataset-level baseline and evidence-coverage audit.
- Alpha/vote/population reachability summary.
- Source-level FN reachability summary.
- High-FN dataset diagnostic table.
- Input/output manifest with SHA256 values.
- Korean result report.

## Interpretation Boundary

Exp391 is a retrospective counterfactual diagnostic. It may justify a
separately preregistered Exp392, but it cannot promote candidates to automatic
Hard alerts and cannot establish prospective equipment performance.
