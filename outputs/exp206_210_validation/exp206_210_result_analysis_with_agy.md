# Exp206-210 Result Analysis With Agy

## Scope

- Retrospective evaluation subset: 251 datasets with `n_train >= 100`
- Baseline: frozen Exp137 autonomous Hard alert
- Total true anomalies: 2,974
- Baseline Hard TP/FP/FN: 1,115 / 206 / 1,859
- Baseline micro precision: 84.41%
- Baseline macro mean Hard F1: 0.482322
- Baseline Zero-F1 datasets: 57

All Exp206-210 jobs completed for 251/251 datasets with no recorded errors. Exp208 and Exp209 frozen-prediction hashes matched their manifests.

## Main conclusion

Exp206 and Exp207 produced useful calibration diagnostics. Exp208 did not provide an operationally useful review source, and Exp209 failed decisively. Neither Exp208 nor Exp209 should alter the frozen Exp137 Hard-alert policy.

## Exp206: OOF reconstruction

Spectral and interval sources transferred reasonably well from out-of-fold TRAIN scoring to full-fit TRAIN scoring. Shapelet did not.

| Source | Median rank correlation | P10 rank correlation | Interpretation |
| --- | ---: | ---: | --- |
| Spectral | 0.9968 | 0.9811 | Stable |
| Interval | 0.9954 | 0.9279 | Mostly stable |
| Shapelet | 0.5177 | 0.1594 | Unstable fold/full representation |

The shapelet source is independent from the other sources, but the evidence indicates instability rather than reliable complementary anomaly information.

## Exp207: TRAIN-normal audit

At alpha 0.005, all three continuous fusion rules remained review diagnostics only:

| Rule | False alerts / pseudo-normal runs | Rate | Wilson upper bound |
| --- | ---: | ---: | ---: |
| Top-2 mean | 504 / 120,575 | 0.004180 | 0.004497 |
| Weighted top-2 | 506 / 120,575 | 0.004197 | 0.004514 |
| Fisher | 495 / 120,575 | 0.004105 | 0.004420 |

No rule met the autonomous/shadow risk target of 0.001. Also, alpha 0.005 is structurally unreachable when `n_train <= 199`, because the minimum empirical p-value is `1 / (n_train + 1)`. This creates zero coverage in 74 datasets and must not be interpreted as safety.

## Exp208: Continuous alternative fusion

At the only review-eligible alpha, 0.005:

| Rule | Candidates | TP | FP | Precision | Mean F1 | Zero-F1 repaired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fisher | 481 | 24 | 457 | 4.99% | 0.467698 | 1 |
| Weighted top-2 | 344 | 16 | 328 | 4.65% | 0.468894 | 1 |
| Top-2 mean | 348 | 16 | 332 | 4.60% | 0.469128 | 1 |

All variants reduced macro Mean F1 relative to 0.482322. The single clean Fisher repair cannot be converted into a dataset/family routing rule using retrospective TEST labels. Exp208 should therefore be rejected as a general Standard-review source and retained only as research evidence.

## Exp209: Seven-source joint novelty

At alpha 0.005:

| Rule | Candidates | TP | FP | Precision | Mean F1 | P95 workload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rank Gaussian | 64,100 | 13 | 64,087 | 0.020% | 0.213293 | 889.5 |
| Shrinkage covariance | 56,121 | 17 | 56,104 | 0.030% | 0.222777 | very high |

The apparent Zero-F1 repairs were not operational repairs: roughly one TP was accompanied by hundreds of FP. Exp209 should be rejected for both automatic alerts and review.

The primary failure is calibration transport. Cross-fitted TRAIN meta-scores and full-fit TEST joint scores were not produced by sufficiently matched scoring mechanisms. Normal-to-TEST distribution shift then amplified weak source shifts, while unstable shapelet features further degraded the joint representation. This is not a top-N or threshold-only problem.

## Agy review and Codex decision

Agy independently recommended stricter source agreement, multiple-testing control, exclusion or down-weighting of unstable shapelet evidence, and non-parametric conformal calibration.

Accepted direction:

- Remove or gate unstable sources using TRAIN-only stability evidence.
- Match calibration and query scoring mechanisms using fold-fitted ensembles.
- Keep non-parametric, TRAIN-only calibration.

Deferred direction:

- Benjamini-Hochberg/FDR control is not the first fix. It assumes sufficiently valid p-values; it cannot repair the observed OOF-to-TEST calibration transport failure by itself.
- A direct 2-of-3 residual rule should not be promoted based only on the TRAIN audit. Earlier retrospective behavior and Exp208 precision show that source agreement can still be mostly FP.

## Recommended next experiments

1. Calibration transport audit: compare OOF TRAIN, full-fit TRAIN, and unlabeled TEST score distributions; quantify location, scale, rank, KS/Wasserstein, and empirical tail-rate drift.
2. Fold-ensemble query scoring: score TEST with the same fold-fitted models used to create OOF references, then aggregate fold p-values or scores. This directly addresses mechanism mismatch.
3. TRAIN-only source stability guard: pre-register rank-correlation and robust scale/location parity criteria. Report spectral, interval, and shapelet ablations without selecting by TEST performance.
4. Only after valid p-values are demonstrated, test conservative 2-of-3 agreement and FDR as separate retrospective sensitivity experiments.

Existing Exp137 autonomous Hard alerts remain unchanged. Any future alternative-source output stays research-only or review-only until prospective shadow validation.
