# Exp239-248 Diverse Score Source Qualification Design

## 1. Objective

Qualify ten additional anomaly score sources against the frozen Exp236 lanes and the Exp237-238 evidence contract. The experiment asks whether genuinely different representations can:

1. confirm existing Hard alerts with higher precision,
2. prioritize Standard review candidates,
3. recover anomalies outside existing lanes without unacceptable false alarms.

The experiment does not change Hard, Standard review, or Priority review. It is a retrospective qualification study, not a deployment policy.

## 2. Prior Evidence

- Exp237 showed that five new sources were poor as direct Hard-alert unions. Their outside-Hard precision ranged from 2.77% to 17.09% at alpha 0.005.
- Exp237 showed that shrinkage normal-space and elastic normal prototype were useful as confirmation inside Hard, with 91.07% and 90.94% precision.
- Exp238 showed that increasing ROCKET kernels from 256 to 512 or 1,024 did not improve standalone precision.
- Exp238 showed that agreement across ROCKET dimensions and seeds, combined with independent normal-space evidence, produced a 96.42% confidence-A subset.

The next sources must therefore add different information and must be qualified as evidence before they can influence any alert lane.

## 3. Fixed Population And Information Boundary

- Database: `/Users/minho/Documents/Dataset/univariate_ts.db`
- Population: the same 1,117 datasets used by Exp236-238.
- TRAIN data: normal traces only.
- TEST values may be scored after all source configuration and calibration rules are frozen.
- TEST labels and actual anomaly positions are available only in the evaluation phase after prediction hashes are frozen.
- Dataset and family names may be used as lookup keys only, never for policy branching.
- TEST length may not choose a source, threshold, candidate count, or budget.
- `n_train < 5` produces explicit calibration abstention for every source.
- Model-based sources may have stricter preregistered TRAIN-size eligibility, but a dataset may never disappear silently.
- Tail replacement, TEST-position assumptions, TEST-metric source selection, and family performance priors are forbidden.

## 4. Experiment Map

### Phase 1: Lightweight deterministic sources

| Experiment | Operational source name | Frozen score definition | Main anomaly type |
|---|---|---|---|
| Exp239 | Multiscale Wavelet Energy Deviation | Robust distance of fixed wavelet-packet energy and local concentration features from TRAIN-normal | transient, local oscillation, scale-energy shift |
| Exp240 | Ordinal Pattern Distribution Shift | Kernel MMD between each trace's ordinal-pattern distribution and cross-fitted TRAIN-normal reference | ordering and nonlinear dynamic shift |
| Exp241 | Symbolic Motif Rarity | Smoothed negative log probability of fixed-length SAX n-grams under cross-fitted TRAIN-normal motif counts | unseen or rare recipe motifs |
| Exp242 | Complexity Signature Deviation | Shrinkage distance over permutation entropy, spectral entropy, roughness, zero crossing, turning points, and Hjorth parameters | noise, stuck sensor, irregularity shift |
| Exp243 | Recurrence Dynamics Deviation | Shrinkage distance over fixed recurrence rate, determinism, laminarity, trapping time, and recurrence entropy | broken periodicity and nonlinear regime shift |

### Phase 2: Model-based sources

| Experiment | Operational source name | Frozen score definition | Main anomaly type |
|---|---|---|---|
| Exp244 | Robust State-Space Innovation | Robust aggregate of uncertainty-standardized innovations from a TRAIN-normal linear state-space model | trend, level, and predictive-dynamics violation |
| Exp245 | Smooth Fourier Normal Residual | Joint score from residual energy and normal coefficient distance using TRAIN-frozen harmonics and regularization | global period and smooth-shape violation |
| Exp246 | Transformation Consistency | Instability of a small normal-only encoder across fixed crop, mask, jitter, and amplitude transforms | representation-level structural inconsistency |
| Exp247 | Masked Patch Context Error | Cross-fitted masked-patch reconstruction error plus latent nearest-normal distance | local context violation |
| Exp248 | Nonlinear Normal Density | Negative log density from a regularized low-dimensional normalizing flow fitted only to TRAIN-normal embeddings | nonlinear low-density normal state |

The names above are the names used in code, reports, and the dashboard. Experiment numbers are identifiers, not score-source names.

## 5. Frozen Feature Configurations

Every configuration is fixed before full evaluation and included in a hashed config manifest.

### Exp239

- Wavelet family: `db4`.
- Maximum decomposition level: `min(5, pywt.dwt_max_level(target_length, db4))`.
- Features: normalized energy, maximum local energy, energy entropy, and temporal concentration per level.
- Score backend: robust covariance with diagonal shrinkage fallback.

### Exp240

- Ordinal embedding dimensions: 3 and 4.
- Delays: 1, 2, and 4 where target length permits.
- Feature: normalized ordinal-pattern histogram.
- Score: unbiased RBF-kernel MMD to the cross-fitted normal reference; bandwidth from TRAIN-only median heuristic.

### Exp241

- PAA segment counts: 16, 32, and 64 where target length permits.
- SAX alphabet size: 5.
- N-gram lengths: 2 and 3.
- Breakpoints and Laplace smoothing are learned from TRAIN-normal only.
- Score: mean negative log probability of observed n-grams plus unseen-motif ratio.

### Exp242

- Fixed feature vector: permutation entropy, spectral entropy, Hjorth activity/mobility/complexity, robust roughness, zero-crossing rate, turning-point rate, and flat-line ratio.
- Score backend: robust scaler followed by shrinkage Mahalanobis distance.

### Exp243

- Recurrence feature extraction deterministically resamples each TRAIN-aligned trace to at most 256 points to bound quadratic distance cost.
- Delay embedding dimensions: 2 and 3.
- Delay selected from a fixed TRAIN-only autocorrelation rule, clipped to `[1, 16]`.
- Recurrence threshold: TRAIN-only distance quantile fixed at 10%.
- Features: recurrence rate, determinism, laminarity, trapping time, diagonal entropy, and vertical entropy.
- Score backend: robust shrinkage distance.

### Exp244

- Model family: local-level plus damped-trend linear Gaussian state-space model.
- Parameters are fit on TRAIN-normal traces only with robust innovation clipping fixed before evaluation.
- Score: median, p95, maximum, and run-length aggregate of standardized innovation magnitude.
- Eligibility: `n_train >= 10`; otherwise model-specific abstention.

### Exp245

- Harmonic candidates are fixed from the first 16 TRAIN-only FFT frequencies ranked by aggregate normal energy.
- The number of retained harmonics is fixed at 8 or the available maximum.
- Ridge regularization is selected by TRAIN-only cross-fitting over a frozen grid.
- Score combines robust residual energy and coefficient-space normal distance using equal preregistered weights.
- Eligibility: `n_train >= 10`.

### Exp246

- Encoder: small one-dimensional convolutional encoder, no pretrained TEST-derived weights.
- Transform set: fixed crop, contiguous mask, low-amplitude Gaussian jitter, and amplitude scaling.
- Objective: identify or align fixed transforms using TRAIN-normal traces only.
- Score: mean latent inconsistency across transforms relative to TRAIN-normal OOF inconsistency.
- Seeds: two fixed seeds; both are reported and their intersection is reported separately.
- Training epochs: 8 per fold and seed.
- Eligibility: `n_train >= 20`.

### Exp247

- Model: small one-dimensional convolutional masked autoencoder.
- Patch and mask ratios: frozen config, one model size, two fixed seeds.
- Score: 0.5 masked reconstruction error plus 0.5 robust latent nearest-normal distance after TRAIN-only standardization.
- Eligibility: `n_train >= 20`.
- Training epochs: 12 per fold and seed.

### Exp248

- Input: a fixed low-dimensional embedding generated by the Exp247 encoder before TEST evaluation.
- Density model: compact RealNVP with fixed depth and width, two fixed seeds.
- Score: ensemble median negative log density.
- Eligibility: `n_train >= 50`; otherwise model-specific abstention.
- Raw likelihood and typicality-corrected rank are both reported; neither is selected from TEST metrics.
- Training epochs: 40 per fold and seed.

## 6. OOF Calibration Contract

- Every eligible source must produce one OOF TRAIN-normal score per TRAIN trace and one TEST score per TEST trace.
- A pseudo-test TRAIN trace may not participate in its own fitted reference, scaler, motif counts, model, or threshold.
- Fold assignment is deterministic from the frozen seed and TRAIN row position, not dataset name.
- Frozen alpha grid: `0.005`, `0.01`, `0.02`, `0.05`.
- Empirical upper-tail p-values use finite-sample correction.
- Attainable p-value and resolution-blocked status are recorded for every dataset/source/alpha.
- No alpha is selected as an operating value from TEST results.
- Raw score vectors are preserved so later policy experiments do not require recomputing representations.

## 7. Execution Architecture

One package, `experiments/exp239_248_diverse_score_sources/`, owns:

- common DB loading, sanitization, TRAIN-derived target length, and alignment,
- source interfaces and hashed configuration,
- resumable per-dataset cache,
- frozen label-free prediction JSONL,
- evaluation after prediction hash verification,
- independent reaggregation,
- dashboard heartbeat and queue metadata.

Phase 1 uses up to seven CPU workers. Phase 2 executes one model source at a time. MPS is used only where a smoke benchmark confirms a speed benefit and identical numerical contracts; concurrent MPS jobs are forbidden.

The queue order is Exp239 through Exp248. A failure records the dataset and source, retries the missing unit, and does not silently mark an incomplete source successful.

## 8. Required Evaluation Views

Every source and frozen alpha reports four separate views:

1. standalone source candidates,
2. confirmation intersected with frozen Exp236 Hard,
3. review prioritization intersected with frozen Exp236 Standard review,
4. incremental candidates outside frozen Hard, Standard review, and Priority review.

Required metrics include candidate count, TP, FP, micro precision, mean dataset F1, zero-candidate datasets, abstention, resolution blocking, TRAIN OOF false-alert rate, seed stability, and source correlation.

Source correlation is computed from aligned TRAIN OOF scores. Sources with high correlation are treated as one evidence family and may not be counted as independent votes.

## 9. Preregistered Research Gates

These gates classify retrospective evidence; they do not choose deployment policy.

### Confirmation gate

- At least 100 confirmed Hard alerts.
- Micro precision at least 90%.
- Precision at least 5 percentage points above frozen Exp236 Hard.
- Stable direction across both seeds where applicable.

### Review-priority gate

- At least 50 Standard review candidates.
- Micro precision at least 70%.
- No autonomous promotion.

### Coverage-expansion gate

- At least 25 incremental true positives outside all frozen lanes.
- Incremental precision at least 80%, matching the user-approved sufficiency level.
- No family-specific rule and no TEST-derived threshold adjustment.
- TRAIN OOF calibration and seed-stability checks pass.

Any source failing the coverage-expansion gate remains confirmation or diagnostic evidence only.

## 10. Invariance And Integrity Tests

- Prediction functions do not accept `y_test`.
- Label permutation and anomaly-position mutation do not change predictions.
- Dataset and family rename do not change source configuration or prediction.
- Input row order does not change dataset-level prediction after canonical sorting.
- TEST append does not change TRAIN calibration parameters.
- Same input, config, and seed produce the same sorted output hash.
- Prediction JSONL contains no TP, FP, F1, labels, or actual anomaly positions.
- Prediction hash tampering blocks evaluation.
- Every eligible dataset has complete score vectors; every ineligible dataset has an explicit reason.
- Existing Exp236-238 files and lanes remain unchanged.

## 11. Outputs

External result root:

`/Users/minho/Documents/Dataset/exp239_248_diverse_score_sources/`

Repository evidence root:

`outputs/exp239_248_diverse_score_sources/`

Required artifacts:

- source config manifest and SHA256,
- cache manifest and per-source coverage,
- frozen prediction JSONL and SHA256,
- source-level result and summary CSVs,
- confirmation/review/outside-lane qualification CSVs,
- TRAIN OOF correlation matrix and evidence-family report,
- independent reaggregation JSON,
- invariance test log,
- result report and manifest,
- dashboard status and queue records.

## 12. Interpretation Boundaries

- Meeting a retrospective research gate does not establish deployment safety.
- Exp239-248 do not select an operating alpha or alter autonomous Hard alerts.
- `mean_combined_f1` remains human-assisted diagnostic information if reported.
- Heavy model success does not establish end-to-end strict TRAIN-only provenance for earlier frozen feature choices.
- All results use repeatedly observed benchmark TEST data and require future locked or prospective shadow validation.

## 13. Recommended Decision After Completion

The primary comparison is not which source has the largest standalone F1. The decision order is:

1. verify TRAIN calibration and reproducibility,
2. reject highly correlated duplicate evidence,
3. identify sources that improve confidence inside Hard,
4. identify sources that improve Standard review ordering,
5. consider a separate preregistered alert-expansion experiment only for a source that passes the 80% outside-lane precision gate.
