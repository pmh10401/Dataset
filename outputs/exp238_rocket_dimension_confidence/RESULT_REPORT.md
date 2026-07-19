# Exp238 ROCKET Dimension Confidence Result

## Scope

- Population: 1,117 datasets
- Prediction rows: 4,468 (1,117 datasets x 4 frozen alpha values)
- Calibration abstention: 162 datasets with fewer than 5 TRAIN-normal samples
- Cache and prediction errors: 0
- Purpose: rank the confidence of frozen Exp236 Hard alerts and Standard review candidates
- Exp236 Hard, Standard review, and Priority review lanes were not changed
- This is a retrospective sensitivity audit and is not deployment validation

## Frozen Exp236 Autonomous Baseline

The autonomous Hard alert result is unchanged for every alpha:

| Hard alerts | TP | FP | Micro precision | Recall | Mean dataset F1 |
|---:|---:|---:|---:|---:|---:|
| 2,116 | 1,789 | 327 | 84.546% | 45.661% | 0.604133 |

## Confidence Stratification

The most selective frozen sensitivity, alpha 0.005, produced the clearest confidence separation:

| Hard confidence | Definition | Count | TP | FP | Precision |
|---|---|---:|---:|---:|---:|
| A | Stable across 256/512/1,024-kernel ROCKET and second seed, plus independent auxiliary evidence | 307 | 296 | 11 | 96.417% |
| B | Either ROCKET stability or independent auxiliary evidence | 538 | 477 | 61 | 88.662% |
| C | Neither additional confirmation condition | 1,271 | 1,016 | 255 | 79.937% |
| A+B | At least one additional confirmation family | 845 | 773 | 72 | 91.479% |

A+B contains 43.21% of all Hard true positives but only 22.02% of Hard false positives. A alone contains 16.55% of Hard true positives and 3.36% of Hard false positives.

These strata do not add or remove alerts. They support confidence labels and user review ordering only.

## Standard Review Priority

At alpha 0.005:

| Review priority | Count | TP | FP | Precision |
|---|---:|---:|---:|---:|
| 1 | 14 | 11 | 3 | 78.571% |
| 2 | 144 | 99 | 45 | 68.750% |
| 3 | 809 | 401 | 408 | 49.567% |
| 1+2 | 158 | 110 | 48 | 69.620% |

The ranking is directionally useful: candidates with extra confirmation should be shown before Priority 3. Priority 1 is too small to promote automatically without prospective validation.

## Kernel Dimension Finding

At alpha 0.005, standalone ROCKET candidate precision was:

| Configuration | Candidates | Precision |
|---|---:|---:|
| 256 kernels / 512 features | 1,123 | 82.191% |
| 512 kernels / 1,024 features | 1,148 | 79.791% |
| 1,024 kernels / 2,048 features | 1,162 | 81.583% |
| Stable intersection across all dimensions and second seed | 928 | 86.853% |

Increasing the kernel count alone did not improve precision. The 512-kernel result was 2.40 percentage points below 256 kernels, and 1,024 kernels was 0.61 points below. The stable intersection was 4.66 points above the 256-kernel result.

The useful signal is therefore repeatability across representations, not raw dimensional growth.

## Alpha Sensitivity

Alpha values were frozen before evaluation and none is selected as an operating value from TEST results.

| Alpha | A count | A precision | A+B count | A+B precision |
|---:|---:|---:|---:|---:|
| 0.005 | 307 | 96.417% | 845 | 91.479% |
| 0.010 | 464 | 92.241% | 1,158 | 88.256% |
| 0.020 | 719 | 88.595% | 1,492 | 85.992% |
| 0.050 | 990 | 86.263% | 1,685 | 84.214% |

Larger alpha values attach confirmation to more alerts but weaken the confidence distinction. Choosing an operating alpha requires TRAIN-only calibration or prospective shadow validation.

## Verification

- Cache manifest: 1,117 datasets
- Frozen prediction: 4,468 rows
- Missing datasets: 0
- Cache errors: 0
- Hard/Standard/Priority lane changes: 0
- Confidence and review partitions: exact and disjoint
- Candidate index range violations: 0
- Independent reaggregation mismatches: 0
- Independent reaggregation: PASS

## Conclusion

Exp238 does not improve autonomous Hard F1 because it intentionally leaves Exp236 alerts unchanged. It succeeds at a different operational objective: separating existing alerts into confidence levels. Confidence A is a strong high-trust label, A+B is a useful high-priority group, and Standard review Priority 1/2 improves review ordering. More ROCKET dimensions should not replace the 256-kernel baseline by themselves.
