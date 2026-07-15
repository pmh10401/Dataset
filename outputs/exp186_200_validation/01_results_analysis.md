# Exp186-200 Results Analysis

## Executive Summary

- Exp137 autonomous Hard alerts remain unchanged. No Exp186-200 result is an
  operating-policy selection.
- Exp186 is the strongest narrow high-confidence review result: 68 candidates,
  62 TP, 6 FP, and 91.18% precision.
- Exp195 raises retrospective precision to 97.06% (33 TP, 1 FP), but it has only
  34 candidates across 10 active datasets. Its 95% Wilson interval is wide
  (85.08%-99.48%), so it is a prospective shadow hypothesis, not an automatic
  alert rule.
- Exp190 high plus standard provides the best practical review balance among
  the preregistered tiers: 263 candidates, 222 TP, 41 FP, and 84.41% precision.
- Exp196 is the broad-coverage option: 593 candidates recover 426 additional
  anomalies at 71.84% precision. It improves human-assisted coverage but is too
  noisy for autonomous promotion.
- Exp197 and Exp199 are dominated by Exp196 on this retrospective benchmark:
  both recover fewer TP while producing more FP.

## Scope And Definitions

The evaluated population is 639 datasets with at least 20 TRAIN-normal
instances. Those datasets contain 3,440 TEST anomaly instances. Every new
candidate excludes all frozen Exp137 Hard, Standard, and Priority indices.
Consequently, candidate TP is an additional anomaly found after the frozen
Exp137 routing.

Precision is micro precision across candidate indices. Mean F1 is the mean of
dataset-level candidate-only F1 and is expected to be small because most
datasets have no new candidate. Review metrics are human-assisted diagnostics;
shadow metrics are not autonomous alerts.

## Review Quality: Exp186-190

| Experiment | Candidates | TP | FP | Precision | Active datasets | P95 workload |
|---|---:|---:|---:|---:|---:|---:|
| Exp186 high-confidence agreement | 68 | 62 | 6 | 91.18% | 23 | 0 |
| Exp187 dual canonical agreement | 20 | 16 | 4 | 80.00% | 14 | 0 |
| Exp188 4-of-5 stability guard | 42 | 32 | 10 | 76.19% | 27 | 0 |
| Exp189 TRAIN count guard | 120 | 95 | 25 | 79.17% | 67 | 1 |
| Exp190 high tier | 68 | 62 | 6 | 91.18% | 23 | 0 |
| Exp190 standard tier | 195 | 160 | 35 | 82.05% | 69 | 1.1 |
| Exp190 exploratory tier | 293 | 100 | 193 | 34.13% | 78 | 3 |

Exp187's extra robust-backend requirement removes 46 TP but only 2 FP relative
to Exp186. It is therefore not a useful additional confirmation gate. Exp190's
high and standard tiers are disjoint; together they produce 263 candidates,
222 TP, 41 FP, and 84.41% precision. The exploratory tier has materially lower
quality and should remain research-only.

## Shadow Promotion Safety: Exp191-195

| Experiment | Candidates | TP | FP | Precision | 95% Wilson CI |
|---|---:|---:|---:|---:|---:|
| Exp191, same rule as Exp186 | 68 | 62 | 6 | 91.18% | 82.06%-95.89% |
| Exp192, dual backend | 20 | 16 | 4 | 80.00% | not used for selection |
| Exp193, three existing plus canonical | 2 | 2 | 0 | 100.00% | too few observations |
| Exp194, fold-stable | 6 | 5 | 1 | 83.33% | too few observations |
| Exp195, zero normal candidate guard | 34 | 33 | 1 | 97.06% | 85.08%-99.48% |

Exp195 is the most promising shadow hypothesis. It retains half of Exp191's
candidates and removes five of six FP, but also removes 29 TP. The removed half
still has 85.29% precision, showing that Exp195 is a confidence split rather
than proof that the remaining Exp191 candidates are poor.

## Coverage And Workload: Exp196-200

| Experiment | Candidates | TP | FP | Precision | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Exp196 expanded TRAIN risk | 593 | 426 | 167 | 71.84% | 4 | 26 |
| Exp197 conservative/canonical union | 556 | 322 | 234 | 57.91% | 5 | 27 |
| Exp198 existing-one plus canonical | 285 | 193 | 92 | 67.72% | 3 | 22 |
| Exp199 expanded 2-of-6 guard | 380 | 168 | 212 | 44.21% | 4 | 14 |

Exp196 dominates Exp197 and Exp199 in retrospective TP/FP space: it recovers
more true anomalies while generating fewer false positives. Exp198 remains a
middle-workload sensitivity point, but Exp190 high plus standard is more
precise with similar candidate volume.

Exp200's `unique candidates` column compares each configuration against every
other configuration, including intentionally duplicated review/shadow rules
(for example Exp186 and Exp191). A zero in that column therefore means the
candidate appears in another reported configuration; it does not mean the
candidate has no diagnostic value.

FP is not confined to one pathological dataset. Exp196's 167 FP span 79
datasets, and 17 datasets account for half. This supports a general calibration
problem rather than a single-family exception and argues against a dataset-name
exception rule.

## Human-Assisted Counterfactual

Across all 1,117 datasets, frozen Exp137 combined routing has 1,991 TP and 662
FP. Adding new review candidates gives the following retrospective diagnostic:

| Added review rule | Added candidates | Combined TP | Combined FP | Combined precision | Combined recall | Mean combined F1 |
|---|---:|---:|---:|---:|---:|---:|
| Exp195 | 34 | 2,024 | 663 | 75.33% | 51.66% | 0.702288 |
| Exp186 | 68 | 2,053 | 668 | 75.45% | 52.40% | 0.703078 |
| Exp190 high plus standard | 263 | 2,213 | 703 | 75.89% | 56.48% | 0.708235 |
| Exp198 | 285 | 2,184 | 754 | 74.34% | 55.74% | 0.707155 |
| Exp196 | 593 | 2,417 | 829 | 74.46% | 61.69% | 0.718796 |

These are human-assisted counterfactuals, not autonomous performance. Exp196
has the highest coverage and mean combined F1 because a user is assumed to
review every added candidate.

## Decision

1. Keep Exp137 Hard alerts unchanged.
2. Treat Exp190 high plus standard as the clearest review-lane structure for a
   locked prospective workflow. Keep high and standard visibly separate.
3. Carry Exp195 forward only as a `shadow_hard_diagnostic` hypothesis. Do not
   promote it until enough new, prospectively collected cases establish its
   false-alert behavior.
4. Keep Exp196 as an optional broad research/review lane where higher workload
   is acceptable. Do not use Exp197 or Exp199 as follow-up priorities.
5. Stop selecting additional retrospective variants on the same TEST set. The
   next confidence-building step is frozen prospective shadow validation with
   user-confirmed outcomes and workload logging.

## Validation Status

The headline totals were independently recomputed from frozen prediction JSONL
and TEST labels. Candidate hashes, 639/639 coverage, lane scope, and stage
summary arithmetic were verified. The analysis is ready to share with caveats:
the benchmark has been repeatedly observed, feature provenance is not fully
end-to-end TRAIN-only, and no result is prospective or deployment validated.
