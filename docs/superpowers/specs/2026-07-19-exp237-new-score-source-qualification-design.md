# Exp237 New Score Source Qualification Design

## Purpose

Exp237 qualifies five new, structurally different anomaly score sources before any source is allowed to affect the Exp236 operational policy. The experiment is a retrospective research audit. It does not select a source, alpha, weight, or deployment policy from TEST metrics.

## Frozen score sources

1. `pooled_ar_residual`: pooled TRAIN-normal autoregressive prediction residual.
2. `robust_change_statistics`: robust first/second derivative and level-shift statistics measured against TRAIN-normal references.
3. `shrinkage_normal_space`: compact statistical and spectral features scored by a shrinkage Mahalanobis distance fitted on TRAIN normal traces.
4. `elastic_normal_prototype`: bounded shift-alignment distance to a TRAIN-normal medoid. This is an explicit lightweight elastic approximation, not exact DTW.
5. `subsequence_discord`: distance of local subsequences to a TRAIN-normal subsequence dictionary. This is matrix-profile-inspired, not an exact Matrix Profile implementation.

The first three are primary compute paths. The two bounded approximations are retained as research probes and must be named as approximations in every report.

## Data and calibration contract

- Database: `/Users/minho/Documents/Dataset/univariate_ts.db` in read-only mode.
- TRAIN rows are assumed normal; TEST values are label-free during prediction.
- Target length, preprocessing, model fitting, score calibration, and source qualification use TRAIN normal values only.
- Datasets with fewer than five TRAIN rows abstain from calibration.
- A deterministic five-fold plan with seed `20260719` generates one OOF score per TRAIN row.
- The source-computation length is capped at 256 points by deterministic interpolation after the TRAIN-derived alignment length is fixed. This cap is frozen before TEST evaluation.
- Full TRAIN is then fitted once to score TEST rows.
- Frozen alpha grid: `0.005, 0.01, 0.02, 0.05`. All variants are reported; none is selected from TEST outcomes.
- Prediction JSONL contains score-derived candidate indices and provenance, but no labels or TP/FP/F1 fields.
- TEST labels are loaded only after the prediction file hash is frozen and verified.

## Qualification evidence

For every dataset, source, and alpha, report:

- finite score coverage and abstention reason;
- TRAIN OOF alert count/rate and Wilson upper bound;
- attainable p-value resolution;
- candidate count distribution;
- seed/config/hash provenance;
- TEST retrospective TP, FP, precision, recall, and F1 after prediction freeze;
- TRAIN-only Spearman correlation with the frozen existing score sources where aligned vectors are available.

The report groups evidence conceptually into temporal, dynamics, normal-space, shape, and frequency families. Correlated score sources must not be counted as independent votes merely because they are separate implementations.

## Execution and safety

- Run a 10-dataset smoke test with two workers before the 1,117-dataset run.
- Run the full cache/prediction with seven dataset workers and resumable per-dataset artifacts.
- Atomic writes and per-artifact SHA256 hashes prevent partial results from being accepted.
- Missing datasets or failed score sources stop final aggregation; failures are not treated as empty candidates.
- Dashboard status is updated through the existing heartbeat/status contract without changing Exp236 results.

## Out of scope

- Exp236 policy replacement or score fusion.
- Exact DTW, exact Matrix Profile, deep neural training, or MPS acceleration.
- TEST-driven source selection, alpha selection, threshold tuning, or family routing.
- Prospective or deployment validation claims.

## Follow-up gate

Only after Exp237 is complete may Exp238 preregister a group-aware agreement policy. That policy should treat at least three distinct evidence groups as a hard-alert research candidate and two groups as review-only, with exact constants frozen before TEST evaluation.
