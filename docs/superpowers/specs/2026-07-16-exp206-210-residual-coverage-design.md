# Exp206-210 Residual Coverage Design

## 1. Goal

Reduce Exp137 autonomous Hard-alert Zero-F1 datasets in the frozen
`train_normal_count >= 100` population while preserving the project's false
alarm and user-trust priorities.

The batch investigates whether continuous, TRAIN-normal-calibrated fusion can
recover anomalies that are not already present in any Exp137 lane. It does not
change Exp137 autonomous alerts and does not promote a retrospective result to
operation.

## 2. Evidence That Motivates the Design

Exp202's alternative three-source rule produced 96 raw same-instance
candidates before excluding Exp137 lanes. Those candidates contained 52 TP and
44 FP, but all 52 TP were already present in an Exp137 lane. The 25 candidates
remaining after lane exclusion were all FP.

The corresponding TEST-normal false-alert rate was `0.000302`, below the
Exp164 TRAIN-normal audit rate of `0.000415`. The primary failure is therefore
not treated as global threshold-calibration collapse. It is treated as a lack
of residual anomaly coverage.

Exp204 remains a separate prospective-shadow candidate for
`train_normal_count >= 200`: 255 added candidates, 216 TP, 39 FP, 84.71% point
precision, and five repaired Zero-F1 datasets. It is not an input to policy
selection in Exp206-210.

## 3. Data-Grain Contract

- One dataset represents one virtual equipment/recipe/sensor/step detection
  environment.
- One TRAIN or TEST instance represents one virtual wafer-run trace.
- A candidate index identifies a TEST instance. It is not a timestamp inside a
  trace.
- Adjacent candidate indices therefore have no guaranteed temporal or physical
  relationship.
- Temporal tolerance matching, neighboring-index event clustering, and tail
  replacement are prohibited in this batch.

## 4. Frozen Scientific Contract

- Exp137 Hard, Standard-review, and Priority-review indices remain unchanged.
- New predictions are retrospective review or shadow diagnostics only.
- Prediction reads TRAIN-normal values and TEST values, but never TEST labels
  or anomaly positions.
- Evaluation verifies a frozen prediction SHA256 before loading TEST labels.
- Dataset and family names are lookup keys only and cannot select a source,
  threshold, fusion rule, risk target, or lane.
- TEST length cannot determine a threshold, candidate budget, or routing rule.
- No TEST metric selects a source, fusion method, alpha, margin, or risk target.
- Every preregistered variant is reported, including negative variants.
- Priority review remains human-assisted and is never promoted to autonomous
  Hard alert.
- Results are retrospective counterfactuals, not prospective or deployment
  validation.
- Historical feature configurations retain unresolved end-to-end provenance;
  this batch is policy-level TRAIN-only, not end-to-end strict TRAIN-only.

## 5. Source Names

Current-source scores:

- temporal random-kernel score (`rocket_exp40`)
- spectrogram-image score (`exp55_best`)
- texture-image score (`exp56_best`)
- pruned official random-kernel score (`exp84`)

Alternative-source scores:

- derivative spectral score
- normal shape-prototype score
- interval distribution score

Company-facing reports use the descriptive names. Historical identifiers are
retained in code and manifests only for reproducibility.

## 6. Exp206: Alternative-Source OOF Reconstruction

Rebuild the three alternative sources with deterministic five-fold TRAIN
cross-fitting over the 251 frozen datasets with `train_normal_count >= 100`.

For every fold:

1. fit feature-dependent state and robust score calibration on the fit folds;
2. score only the held-out TRAIN-normal instances;
3. write each held-out score exactly once;
4. after the complete OOF vector exists, fit on all TRAIN-normal instances and
   score TEST values without labels.

The derivative spectral and interval feature transforms are deterministic.
The shape-prototype source must select prototypes from the fit folds only. Its
seed is derived from the fixed experiment seed and fold number, not from the
dataset or family name.

Required audits:

- 251/251 dataset coverage;
- one OOF score per TRAIN-normal instance;
- no held-out instance used in its own fit state;
- fold determinism and reproducible file hashes;
- OOF/full-fit location, scale, and rank-parity diagnostics;
- no evaluation fields in cache files or manifests.

Any missing dataset, non-finite score vector, or fit/query contamination fails
Exp206. Existing Exp162 cache files are read-only and are not overwritten.

## 7. Exp207: TRAIN-OOF Conditional Novelty Audit

Combine the Exp206 alternative OOF cache with the frozen Exp153-157 current
four-source OOF cache. No TEST labels are read.

For the fixed alpha grid `(0.005, 0.01, 0.02, 0.05)`, report:

- source-wise and pairwise Spearman correlation;
- source-wise tail-event rate;
- pairwise phi correlation and Jaccard overlap;
- alternative-source candidate rate conditional on zero current-source
  support at the same pseudo-normal instance;
- current-source candidate rate conditional on zero alternative support;
- two-of-three and three-of-three alternative agreement risk;
- continuous-fusion pseudo-normal risk;
- one-sided Wilson upper bounds.

No arbitrary minimum novelty percentage is used as a success criterion.
Exp207 answers whether the alternative sources provide a measurably distinct
normal-tail space and whether a proposed fusion satisfies the frozen risk
contracts.

Risk contracts:

- shadow diagnostic: pseudo-normal false-alert Wilson upper bound `<= 0.001`;
- review diagnostic: pseudo-normal false-alert Wilson upper bound `<= 0.005`.

Rules that satisfy neither contract are still reported but are not emitted as
Exp208 shadow/review candidates.

## 8. Exp208: Continuous Alternative-Source Fusion

Freeze and report all of the following instance-level fusion methods before
TEST evaluation:

1. robust-z top-two mean;
2. inverse-Spearman-correlation weighted top-two mean;
3. Fisher combined upper-tail p-value.

Each method uses only Exp206 TRAIN OOF scores to fit scaling, correlation
weights, and the empirical/conformal reference. The fixed alpha grid is
reported in full; no alpha is selected from TEST performance.

Prediction emits candidates outside the complete Exp137 lane union. Methods
meeting the Exp207 shadow risk contract are labeled
`shadow_hard_diagnostic`; methods meeting only the review risk contract are
`standard_review_diagnostic`; all others are
`research_sensitivity_only`.

After prediction and hash freeze, evaluation reports added TP, FP, precision,
recall, candidate-only F1, Zero-F1 repair, workload, and exact overlap with
Exp190 and Exp198.

## 9. Exp209: Seven-Source Joint Normal Novelty

Use the four current OOF sources and three Exp206 alternative OOF sources to
model unusual combinations under TRAIN normality.

Freeze two label-free variants:

1. rank-Gaussian robust diagonal distance;
2. shrinkage robust covariance distance, used only when the TRAIN sample count
   supports a finite and stable estimate.

Both variants use TRAIN OOF values for transformation and calibration. The
same fixed alpha grid and Exp207 risk contracts apply. Unsupported covariance
fits abstain rather than falling back to a TEST-informed rule.

Exp209 tests whether joint normal-distribution novelty supplies residual
coverage that exact same-instance AND rules miss. It does not use neighboring
candidate indices or TEST-derived event windows.

## 10. Exp210: Independent Frontier Reaggregation

Independently reaggregate Exp206-209 and compare them with frozen Exp137,
Exp190, Exp198, Exp201, and Exp204 artifacts.

Required views:

- `100-199`, `200+`, and all 251 datasets;
- added candidate count, TP, FP, precision, recall, and mean candidate F1;
- Zero-F1 before, after, fixed, and newly created;
- workload median, p90, p95, and maximum;
- exact candidate overlap and unique contribution;
- paired dataset bootstrap intervals;
- shadow, review, and research-only metrics separated;
- Exp137 autonomous metrics unchanged byte-for-byte.

Exp210 names no best, optimal, or deployment-ready rule. A retrospective point
precision of at least 80% and a net Zero-F1 reduction are research signals
only.

## 11. Exp204 Prospective Shadow Handling

The frozen Exp204 rule is kept in a namespace separate from Exp206-210. On
genuinely new data it may write label-free shadow candidates, but it must not
notify users as an autonomous alert.

Promotion can be discussed only after at least 100 prospectively adjudicated
added candidates and a one-sided 95% Wilson precision lower bound of at least
80%. The criterion is not evaluated on the repeatedly used historical TEST
benchmark.

## 12. Runtime and Queue

Queue order is Exp206, Exp207, Exp208, Exp209, Exp210. One stage runs at a time
with seven CPU workers where dataset-level parallelism is valid.

MPS is used only if a deterministic smoke benchmark for the exact feature
stage demonstrates at least 1.1x speedup over CPU. The current NumPy/scikit-
learn feature path is expected to remain CPU-first.

Each stage writes:

- progress and full logs;
- resumable atomic checkpoints;
- frozen config and input hashes;
- prediction JSONL and prediction SHA256 where applicable;
- result and summary CSV files;
- completion marker and manifest.

A failed dataset is retried once. Remaining omissions fail the stage and are
visible on the dashboard. Dependent stages stop; independent audit outputs are
preserved.

## 13. Required Tests

- prediction functions do not accept `y_test`;
- prediction payloads contain no TP, FP, FN, precision, recall, or F1;
- TEST label and anomaly-position mutation leaves predictions unchanged;
- dataset/family rename after lookup remapping leaves policy results unchanged;
- fold plans and shape-prototype seeds are deterministic and name-independent;
- every TRAIN-normal instance is scored out of fold exactly once;
- TEST length is absent from threshold, risk, and routing functions;
- input row order does not change dataset-level output after canonical sorting;
- candidate indices are in range and diagnostic lanes are mutually exclusive;
- cache and prediction hash tampering is detected;
- Exp137 lane files and indices are unchanged;
- queue order is exactly Exp206 through Exp210.

## 14. Interpretation Boundary

This batch can show whether OOF-calibrated continuous fusion is a better
retrospective residual-coverage mechanism than exact three-source agreement.
It cannot prove prospective precision, user trust, or end-to-end TRAIN-only
model-development provenance. Any later policy change requires a new frozen
benchmark or genuinely prospective shadow evidence.
