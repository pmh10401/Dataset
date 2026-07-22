# Exp266 Post-Freeze Retrospective Result

This report evaluates the frozen label-free checkpoint after SHA-256 verification.
It is retrospective, is not deployment evidence, and does not change any prediction.
No threshold, policy, or source was selected from TEST results.

## Autonomous Hard

Human-assisted review candidates are excluded from every metric in this table.

| Alerts | TP | FP | FN | Precision | Recall | Mean dataset F1 | Zero-F1 datasets | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2253 | 1915 | 338 | 2003 | 0.849978 | 0.488770 | 0.608659 | 356 | 0 |

## Human-Assisted Review Only

| Lane | Requests | TP | FP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Shadow | 2 | 2 | 0 | 1.000000 | 0.000510 | 0.001020 |
| Priority | 31 | 26 | 5 | 0.838710 | 0.006636 | 0.013168 |
| Standard | 806 | 365 | 441 | 0.452854 | 0.093160 | 0.154530 |
| RecoveryShadow | 15 | 13 | 2 | 0.866667 | 0.003318 | 0.006611 |
| RecoveryPriority | 76 | 44 | 32 | 0.578947 | 0.011230 | 0.022033 |
| DataQualityReview | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |

Unique recovered false negatives: **57**
Datasets with added review candidates: **48**
Total existing plus Exp266 review-request rate per TEST run: **0.004747**
Exp266 incremental review-request rate per TEST run: **0.000465**

## Human-Assisted Combined Diagnostic

user reviews Shadow, Priority, Standard, RecoveryShadow, RecoveryPriority, and DataQualityReview; confirms TP and rejects review FP; combined TP is Hard TP plus confirmed unique review TP, combined FP remains Hard FP.
This combined view is human-assisted and never replaces Autonomous Hard.

| Combined accepted | TP | FP | FN | Precision | Recall | Mean combined F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2703 | 2365 | 338 | 1553 | 0.874954 | 0.603624 | 0.723338 |


## Immutable Lane Parity

| Lane | Pinned source | Matching datasets | Mismatches | Index-map SHA-256 |
|---|---|---:|---:|---|
| Hard | Exp253 | 1117 | 0 | `3af05e308116ef3634a855baf002561e9b3edbea8bcb47d977a4972315c58f2e` |
| Shadow | Exp264 | 1117 | 0 | `47b7448dd95527a670ebeb2c8ff63289a986c83a24a842165dfc3eb6b8b16bd3` |
| Priority | Exp264 | 1117 | 0 | `30f2d5f69d2ff24a075a8503dda0b4f884c6789a1af196a791e541317b3125a7` |
| Standard | Exp264 | 1117 | 0 | `a4778fd0f704a67fba4ac462b973f705f8729ad3fe78c4ad91b45f8a296b1a44` |

## Certified TRAIN-Normal Risk Audit

The certified audit uses exactly four provenance-verified sources. The
five-source reconstruction result remains an unapproved sensitivity diagnostic.
Exact five-source blocker: **Five-source combined TRAIN-risk approval is blocked until Exp257 has a producer-signed/pinned manifest tying the frozen archives to query-excluded fitting and exact seeds, or a separately preregistered regeneration is performed**.
The TRAIN-risk table below applies only to the Exp266 incremental RecoveryShadow and RecoveryPriority lanes, never to existing review lanes.

| Lane | False reviews | Rate | One-sided Wilson U95 | Target <= 0.01 |
|---|---:|---:|---:|---|
| RecoveryShadow | 84 | 0.000572223 | 0.000684505 | PASS |
| RecoveryPriority | 160 | 0.001089948 | 0.001241096 | PASS |
| Combined review | 244 | 0.001662171 | 0.001846477 | PASS |

## Interpretation Boundary

RecoveryShadow, RecoveryPriority, DataQualityReview, and existing review lanes
remain human-assisted and review-only. No review result is added to Autonomous Hard.
All point matches are exact TEST instance-index matches with zero-denominator ratios
recorded as 0.0.
No threshold, policy, or source was selected from TEST results.
Five-source blocker: Five-source combined TRAIN-risk approval is blocked until Exp257 has a producer-signed/pinned manifest tying the frozen archives to query-excluded fitting and exact seeds, or a separately preregistered regeneration is performed.
