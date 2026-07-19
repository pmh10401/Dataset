# Exp237 New Score Source Qualification Result

## Contract

- Experiment: `experiment_237_new_score_source_qualification`
- Population: 1,117 datasets
- TRAIN calibration eligible: 955 datasets
- Calibration abstention (`n_train < 5`): 162 datasets
- Score sources: five frozen, label-free sources
- Alpha grid: `0.005`, `0.01`, `0.02`, `0.05`; no alpha or source was selected from TEST metrics
- Prediction SHA256: `11f1d95f041e024c40eaf18911f1920dbb651080c7c7f730e46f877e1872517d`
- Scope: retrospective research qualification only; no Exp236 lane was changed

## Strict-grid diagnostic at alpha 0.005

This table is a fixed-grid diagnostic, not a selected operating point.

| Source | Candidates | TP | FP | Precision | Recall | Mean F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pooled AR residual | 687 | 237 | 450 | 34.50% | 6.05% | 0.0148 |
| Robust change statistics | 769 | 244 | 525 | 31.73% | 6.23% | 0.0174 |
| Shrinkage normal-space | 739 | 383 | 356 | 51.83% | 9.78% | 0.0275 |
| Elastic normal prototype | 961 | 290 | 671 | 30.18% | 7.40% | 0.0207 |
| Subsequence discord | 534 | 100 | 434 | 18.73% | 2.55% | 0.0124 |

All five sources are too imprecise to be unioned directly into Exp236 Hard alerts. The strongest standalone diagnostic was shrinkage normal-space, but its 51.83% precision is far below Exp236 Hard precision (84.55%).

## Incremental coverage outside Exp236 Hard

At alpha 0.005, the new candidates not already present in Exp236 Hard were:

| Source | New candidates | New TP | New FP | Precision |
| --- | ---: | ---: | ---: | ---: |
| Pooled AR residual | 468 | 51 | 417 | 10.90% |
| Robust change statistics | 522 | 38 | 484 | 7.28% |
| Shrinkage normal-space | 392 | 67 | 325 | 17.09% |
| Elastic normal prototype | 707 | 59 | 648 | 8.35% |
| Subsequence discord | 433 | 12 | 421 | 2.77% |

Requiring agreement among the five new sources did not solve this expansion problem. At alpha 0.005, three-source agreement outside Exp236 produced 203 candidates with 27 TP and 176 FP (13.30% precision); four-source agreement produced 78 candidates with 16 TP and 62 FP (20.51% precision).

## Confidence evidence inside Exp236 Hard

The same scores are more useful as confidence tags on existing Exp236 Hard alerts. At alpha 0.005:

| Confirming source | Confirmed Exp236 Hard | TP | FP | Precision |
| --- | ---: | ---: | ---: | ---: |
| Pooled AR residual | 219 | 186 | 33 | 84.93% |
| Robust change statistics | 247 | 206 | 41 | 83.40% |
| Shrinkage normal-space | 347 | 316 | 31 | 91.07% |
| Elastic normal prototype | 254 | 231 | 23 | 90.94% |
| Subsequence discord | 101 | 88 | 13 | 87.13% |

This supports a confidence/review-priority use case, not an automatic expansion use case. Removing unconfirmed Exp236 alerts was not preregistered and is not evaluated as a new policy here.

## TRAIN-only qualification findings

- Median absolute TRAIN OOF correlations with the frozen existing sources were low to moderate: 0.187 to 0.250 depending on the new source. The sources therefore add different information, but different information was not automatically useful information.
- TRAIN OOF alert rates for pooled AR, robust change, elastic prototype, and subsequence discord were close to their nominal alpha values. Shrinkage normal-space was higher than nominal at every alpha, so its calibration needs extra scrutiny before policy use.
- P-value resolution is a major limitation. At alpha 0.005, 813 of the 955 calibration-eligible datasets could not attain that p-value with their available TRAIN fold reference size. The 162 `n_train < 5` datasets abstained separately.

## Conclusion

Adding score sources is technically feasible and the full 1,117-dataset pipeline is reproducible. However, none of the five sources should be added as an independent Hard-alert union. The useful signal is that shrinkage normal-space and elastic normal prototype can mark a small, higher-confidence subset of existing Exp236 alerts.

The next experiment should therefore preregister a confidence-tier or review-priority audit. It should not increase Hard alerts, should keep Exp236 Hard unchanged, and should use TRAIN-only calibration and source-group correlation controls. A separate coverage-expansion experiment would first need a stronger source with substantially better precision on candidates outside Exp236.

## Verification

- Prediction rows: 22,340 (`1,117 x 5 x 4`)
- Evaluation rows: 22,340
- Qualification rows: 22,340
- Missing datasets: 0
- Cache errors: 0
- Prediction fields containing TEST labels or evaluation metrics: 0
- Independent reaggregation: passed, zero mismatches
- Existing Exp236 outputs were not changed

These are retrospective benchmark results. They do not establish prospective equipment performance or select a deployment threshold.
