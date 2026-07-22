# Size-Stratified Performance Analysis

## Scope

Dataset size is defined as the number of TRAIN-normal instances. Exp137 is
compared with retrospective automatic-promotion counterfactuals. Exp190 means
the disjoint high and standard tiers combined. No result is a deployment
selection.

## Exp137 Baseline By TRAIN Size

| TRAIN normal | Datasets | Anomalies | Precision | Recall | Mean F1 | Zero-F1 | Zero rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| <5 | 162 | 162 | 80.00% | 76.54% | 0.7654 | 38 | 23.5% |
| 5-9 | 74 | 74 | 91.43% | 86.49% | 0.8649 | 10 | 13.5% |
| 10-19 | 242 | 242 | 91.89% | 70.25% | 0.7025 | 72 | 29.8% |
| 20-49 | 138 | 138 | 74.70% | 44.93% | 0.4493 | 76 | 55.1% |
| 50-99 | 250 | 328 | 81.68% | 47.56% | 0.5200 | 109 | 43.6% |
| 100-199 | 74 | 232 | 88.52% | 46.55% | 0.5418 | 16 | 21.6% |
| 200+ | 177 | 2,742 | 83.99% | 36.73% | 0.4575 | 41 | 23.2% |

The largest structural gap is TRAIN 20-49, followed by 50-99. The 200+ group
contains 70% of all anomaly instances, so micro totals are dominated by large
datasets even though the majority of datasets are smaller.

## Exp190 High Plus Standard Promotion

Exp190 changes only datasets with at least 100 TRAIN-normal instances because
the empirical conformal p-value resolution prevents the frozen alpha rules from
activating below that range.

| TRAIN normal | Added TP | Added FP | Added precision | New recall | New Mean F1 | Zero-F1 change |
|---|---:|---:|---:|---:|---:|---:|
| <100 | 0 | 0 | not active | unchanged | unchanged | unchanged |
| 100-199 | 6 | 2 | 75.00% | 49.14% | 0.5594 | 16 -> 15 |
| 200+ | 216 | 39 | 84.71% | 44.60% | 0.5013 | 41 -> 36 |

Exp190 satisfies the 80% incremental precision criterion only in the 200+
group. Applying it universally would hide the weaker 100-199 increment inside
the stronger large-dataset total.

## Exp196 Expanded Promotion

| TRAIN normal | Added TP | Added FP | Added precision | New precision | New recall | Zero-F1 change |
|---|---:|---:|---:|---:|---:|---:|
| <50 | 0 | 0 | not active | unchanged | unchanged | unchanged |
| 50-99 | 18 | 17 | 51.43% | 76.99% | 53.05% | 109 -> 96 |
| 100-199 | 16 | 6 | 72.73% | 86.11% | 53.45% | 16 -> 13 |
| 200+ | 392 | 144 | 73.13% | 80.63% | 51.02% | 41 -> 33 |

Exp196 reduces Zero-F1 more broadly, but its newly promoted candidates fail the
80% precision criterion in every active size group. The overall system remains
above 80% only because Exp137's existing alerts dilute the lower-quality
increment.

## Interpretation

- TRAIN 200+: Exp190 high plus standard is the only current balanced automatic
  promotion hypothesis, with 84.71% incremental precision.
- TRAIN 100-199: current incremental evidence is below 80%; retain review-only.
- TRAIN 50-99: Exp196 improves coverage but produces nearly one FP per TP;
  automatic promotion is not justified.
- TRAIN 20-49: the current empirical p-value thresholds cannot generate new
  candidates at all. This is the highest-priority calibration gap.
- TRAIN <20: Exp137 is already relatively strong on this benchmark. New pooled
  methods should be evaluated separately rather than mixed with 20-99.

## Next Validation

Preregister a size-aware research policy using TRAIN count only:

1. Retain Exp137 everywhere.
2. Test Exp190 promotion only for TRAIN 200+ in prospective shadow mode.
3. Develop leave-target-out TRAIN-only similarity-pooled calibration for TRAIN
   20-99, with a minimum pooled reference size fixed before evaluation.
4. Require incremental candidate precision of at least 80%, not merely total
   system precision above 80%.
5. Report every size stratum separately so large datasets cannot mask weak
   small-dataset behavior.
