# Exp303 Expanded TRAIN-only Gates Design

## Purpose

Exp303 tests whether Exp302 missed useful Standard candidates because each existing Gate selected only two of four verified experts. It compares direct all-expert consensus with two additional TRAIN-only expert-selection Gates. It is retrospective and does not update the operational automatic-alert policy.

## Frozen Candidate Evidence

- Exp294 frozen Standard candidate universe and verified expert support
- Exp293 frozen expert audit
- provenance-verified TRAIN OOF score caches used by Exp293

Every artifact is hash-verified. Prediction reads no TEST labels or evaluation metrics.

## Variants

### All verified experts

Use the number of independently implemented verified score families that support a Standard candidate.

- `verified_4of4`
- `verified_3of4`
- `verified_2of4`
- `verified_1of4`

### Diversity Gate

For each dataset, convert every expert's TRAIN OOF variants to rank-normalized representative scores. Choose the pair with the lowest absolute TRAIN Spearman correlation. Ties use lower summed TRAIN risk and then expert name. Report `diversity_2of2` and `diversity_1of2`.

### Calibration-reliability Gate

Rank available experts using only Exp293 TRAIN audit fields: lower risk upper bound, finer attainable p-value, lower TRAIN drift, and higher within-expert stability. Sum those ranks and select the best two with deterministic name tie-breaking. Report `calibration_2of2` and `calibration_1of2`.

## Evaluation

Each rule produces a counterfactual union of unchanged Exp291 Hard alerts and eligible Standard candidates. Evaluation occurs only after prediction freeze and reports added TP/FP, overall precision/recall/F1, zero-F1 changes, and workload. Precision 80% is not a gate and no variant is selected from TEST metrics.

## Interpretation Boundary

Additional Gates are not additional independent score sources. They are alternative TRAIN-only selectors over the same four sources. Direct k-of-4 results measure source consensus; Gate results measure expert-selection coverage. Exp303 is retrospective and requires future shadow validation before any automatic promotion.
