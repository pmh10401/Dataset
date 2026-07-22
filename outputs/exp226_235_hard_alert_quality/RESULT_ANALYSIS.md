# Exp226-235 Hard Alert Quality Result Analysis

## Evaluation Contract

- Evaluation datasets: 1,117
- Total labelled anomalies: 3,918
- Scope: retrospective counterfactual ablation
- TEST metrics selected no alpha, threshold, margin, weight, cluster count, or FDR level.
- Hard metrics are autonomous. Review and research metrics are reported separately.
- These results are not prospective or deployment validation.

## Exp137 Reference

| Hard alerts | TP | FP | Precision | Recall | Mean dataset F1 |
|---:|---:|---:|---:|---:|---:|
| 2,005 | 1,691 | 314 | 84.339% | 43.16% | 0.600772 |

## Autonomous Sensitivities

The table includes explicit calibration abstention and unavailable-source rows
as zero-Hard outcomes, so every line represents all 1,117 datasets.

| Experiment / frozen variant | Hard TP | Hard FP | Precision | Recall | Mean dataset F1 | Review workload |
|---|---:|---:|---:|---:|---:|---:|
| Exp227 dual confirmation | 583 | 56 | 91.236% | 14.88% | 0.052896 | 1,366 |
| Exp227 at least one confirmation | 997 | 138 | 87.841% | 25.45% | 0.091484 | 870 |
| Exp228 margin 0.0 | 583 | 56 | 91.236% | 14.88% | 0.052896 | 1,366 |
| Exp228 margin 0.5 | 417 | 27 | 93.919% | 10.64% | 0.031045 | 1,561 |
| Exp228 margin 0.8 | 161 | 8 | 95.266% | 4.11% | 0.008976 | 1,836 |
| Exp229 correlation weight 0.50 | 791 | 70 | 91.870% | 20.19% | 0.070171 | 1,144 |
| Exp229 correlation weight 0.67 | 498 | 22 | 95.769% | 12.71% | 0.044330 | 1,485 |
| Exp229 correlation weight 0.75 | 435 | 21 | 95.395% | 11.10% | 0.036791 | 1,549 |
| Exp231 normal modes 1 | 812 | 96 | 89.427% | 20.72% | 0.071219 | 1,097 |
| Exp231 normal modes 2 | 516 | 51 | 91.005% | 13.17% | 0.034623 | 1,438 |
| Exp231 normal modes 4 | 277 | 12 | 95.848% | 7.07% | 0.012858 | 1,716 |
| Exp233 FDR 0.005/0.01/0.02 | 0 | 0 | n/a | 0.00% | 0.000000 | 2,005 |
| Exp233 FDR 0.05 | 82 | 0 | 100.000% | 2.09% | 0.001224 | 1,923 |

## TRAIN-only Risk Audit

Exp226 reproduced the frozen Exp219/Exp224 TRAIN-normal alignment for the
preregistered 251-dataset population.

- Normal pseudo-run shape candidates: 412
- Dual-confirmed normal candidates: 25
- Single-confirmed normal candidates: 48
- Unconfirmed normal candidates: 339
- Outside the frozen population: 704 datasets, explicitly recorded
- `n_train < 5` calibration abstention: 162 datasets, explicitly recorded

The 73 confirmed normal candidates show that confirmation is not a zero-risk
operation. This is TRAIN-normal workload evidence, not a TEST-performance
threshold-selection result.

## Other Experiments

- Exp230 is blocked because valid frozen multi-seed/kernel repetitions do not exist. No synthetic stability result was substituted.
- Exp232 classified all 955 score-available datasets as review-only under its frozen reliability contract; 162 additional datasets abstained. It produced no autonomous Hard alerts.
- Exp234 remained research-review only. Minimum-source variants produced 439 candidates at 72.21% precision, 111 at 88.29%, and 3 at 100%. These must not be added to autonomous Hard performance.
- Exp235 exported 33,143 label-free shadow rows with SHA-256 `716870cf2a177063be8d93315985a12b5a713330b871e3e48c2993b9732fc9dd`.

## Conclusion

The experiments found strong precision filters, but none is a drop-in
replacement for Exp137. Precision rises because many true anomalies move from
Hard alert to review, causing large recall and macro-F1 losses. The most useful
current outcome is therefore evidence stratification: keep Exp137 as the
reference, use confirmation strength as confidence/review information, and
collect prospective shadow feedback before promoting any stricter rule.

No variant is declared primary, optimal, or recommended from these TEST
results. Exp230 requires new preregistered multi-seed artifacts, and the full
configuration provenance remains a separate D2 limitation.
