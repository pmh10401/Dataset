# Exp325-327 Full-source Reliability Result

## Scope

- Retrospective counterfactual audit of the five-source Exp93 rank fusion upstream of Exp291.
- Weights use TRAIN-normal exceedance counts and one-sided Wilson upper bounds.
- TEST labels are not read in Exp325 prediction or Exp326 replay. Exp327 reads labels only after predictions are frozen.
- Existing Exp291 automatic alerts are not changed by these experiments.

## Coverage

- Baseline lineage datasets: 1,117/1,117.
- Primary sources (temporal random kernel, frequency image, texture/shape): 1,117/1,117.
- Auxiliary Exp84 sources (family-guard and count-cap): 339/1,117.
- Missing auxiliary artifacts are declared unavailable with empty rank order and zero reliability. They are not imputed.
- The separate B2 full-coverage source is not substituted because that would confound source replacement with weight replacement.

## Results

| Variant | Changed datasets vs fixed | Selected | TP | FP | Precision | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|
| fixed priors | 0 | 2,644 | 1,983 | 661 | 0.750 | 0.696978 |
| prior x reliability | 0 | 2,644 | 1,983 | 661 | 0.750 | 0.696978 |
| pure reliability | 0 | 2,644 | 1,983 | 661 | 0.750 | 0.696978 |

The fixed-control replay matched the validated Exp119a selected indices for all 1,117 datasets. Reliability changed numeric weights, but it did not change any selected index under the frozen Exp93 replacement gate.

## Interpretation

The current Exp93 decision is dominated by discrete gates: source support, best rank, sparse-base status, and the score-gain threshold. Reweighting alone cannot improve Exp291 while those gates produce the same selection. A future experiment must preregister a reliability-aware decision boundary or a deterministic full downstream replay. This result must not be described as an updated Exp291 automatic-alert policy.
