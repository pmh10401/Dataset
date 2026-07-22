# Exp329-330 Result Report

## Scope

- Population: 251 datasets with at least 100 normal TRAIN instances
- Baseline: frozen Exp291 routing as audited by Exp328
- Baseline anomalies: 2,974
- Baseline Hard: TP 1,339, FP 198, FN 1,635, micro precision 87.12%, recall 45.02%, mean dataset F1 0.518858
- Both experiments are retrospective research Shadow analyses. The automatic-alert policy was not changed.

## Exp329: frozen candidate verifier

| Frozen rule | Shadow count | TP | FP | Shadow precision | Hard + Shadow recall | Mean dataset F1 |
|---|---:|---:|---:|---:|---:|---:|
| review_independent_2 | 38 | 28 | 10 | 73.68% | 45.97% | 0.523436 |
| review_total_support_3 | 58 | 42 | 16 | 72.41% | 46.44% | 0.525329 |
| outside_strict_4of8 | 3 | 0 | 3 | 0.00% | 45.02% | 0.518676 |
| combined_strict | 61 | 42 | 19 | 68.85% | 46.44% | 0.525149 |
| combined_balanced | 130 | 82 | 48 | 63.08% | 47.78% | 0.530716 |

The useful signal is confined to candidates already in the review lane. Independent evidence can rank a small review subset with roughly 72-74% retrospective precision. The rule that attempted to recover candidates outside the existing review lane selected three false positives and no true positives. Therefore Exp329 does not justify expanding autonomous Hard alerts. It supports review prioritization as a future prospective candidate.

No variant is selected as an operational policy from these TEST results.

## Exp330: learned multiscale normal representation

| Representation | Alpha | All candidates TP / FP | Precision | Novel TP / FP | Novel precision | Outside-top64 FN recovered |
|---|---:|---:|---:|---:|---:|---:|
| PCA32 + KNN5 | 0.005 | 413 / 443 | 48.25% | 17 / 382 | 4.26% | 1 |
| PCA32 + KNN5 | 0.010 | 565 / 1,023 | 35.58% | 38 / 924 | 3.95% | 2 |
| PCA32 + KNN5 | 0.020 | 722 / 2,297 | 23.92% | 65 / 2,141 | 2.95% | 4 |
| PCA64 + KNN5 | 0.005 | 458 / 461 | 49.84% | 20 / 392 | 4.85% | 1 |
| PCA64 + KNN5 | 0.010 | 623 / 1,101 | 36.14% | 44 / 990 | 4.26% | 3 |
| PCA64 + KNN5 | 0.020 | 781 / 2,421 | 24.39% | 70 / 2,252 | 3.01% | 4 |

The new representation overlaps many anomalies already reached by the current system, but its genuinely novel candidates are mostly false positives. It recovers only one to four of the 338 Exp328 outside-top64 false negatives while adding hundreds or thousands of false positives. Increasing PCA dimensions from 32 to 64 does not solve that separation problem.

Exp330 is not suitable for automatic-alert expansion or broad review insertion in its current form. It is evidence that a generic resampled multiscale PCA/KNN representation does not provide the missing independent boundary.

## Decision

1. Keep Exp291 Hard alerts unchanged.
2. Treat Exp329's review-lane evidence as a review-priority research signal only.
3. Reject Exp329 outside-review promotion and Exp330 novel candidate insertion.
4. The next representation experiment should target a specific unresolved failure mechanism, not simply increase generic embedding capacity.

## Integrity checks

- Exp329 frozen rows: 251/251, SHA256 `a0c71c27beceff894ea1c4e91b5d89763edd218f51ef2a2dc9ea23a932260254`
- Exp330 frozen rows: 251/251, SHA256 `a98eefd9211f7998cb13053283b32febbd35bc6550d698be870e9559c10c9a72`
- Missing datasets: 0
- Prediction-time TEST labels or evaluation metrics: none
- Candidate index range errors: 0
- Exp329 selected candidate overlap with existing Hard: 0
- TEST-based post-hoc policy selection: none
