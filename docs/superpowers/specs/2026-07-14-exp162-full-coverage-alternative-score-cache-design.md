# Exp162 Full-Coverage Alternative Score Cache Design

## Purpose

Exp161 showed that the historical 미분·스펙트럼 통계 점수, 부분 형태 일치 점수, and 구간 분포 점수 artifacts were computed on different retrospective target subsets. Exp162 rebuilds all three sources with one frozen configuration over the same 1,117 datasets so a later experiment can evaluate candidate-zero recovery without reusing family- or TEST-performance-selected artifacts.

Exp162 is a score-cache construction experiment, not a candidate-selection or performance experiment.

## Frozen Population And Inputs

- Dataset population: the exact 1,117 names in the frozen Exp160 prediction artifact.
- Database: `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Prediction/cache phase reads `instances.values_blob`, split, index, and non-label dataset metadata only.
- TEST labels, anomaly positions, family names, past TEST performance, and Exp160 candidate-zero status are not feature or score inputs.
- Target length is the median TRAIN-normal series length, clipped to `[8, 2048]`; TEST length does not configure the model.

## Frozen Source Configurations

### 미분·스펙트럼 통계 점수

- Per-series features: raw, first derivative, and second derivative mean/std/median/p90-p10/max-absolute; eight normalized FFT bands; spectral centroid, entropy, and peak.
- Scaling and center/spread fitting use TRAIN-normal features only.
- Score is the mean absolute robust deviation across all feature dimensions.

### 부분 형태 일치 점수

- Resample to TRAIN-derived target length capped at 512.
- Z-normalize each series.
- Select 32 TRAIN-only shapelet prototypes with lengths from `8, 16, 32, 64` that fit the target length.
- Seed is `20260714 + stable_hash(dataset_name)` and is used only for deterministic sampling, never for policy branching.
- Score is the mean absolute robust deviation of the 32 minimum-distance features.

### 구간 분포 점수

- Z-normalized raw signal, first difference, and log-periodogram views.
- Sixteen deterministic intervals plus one full-series interval per view.
- Each interval emits mean, standard deviation, min, max, q05, q25, q50, q75, q95, and slope.
- Scaling and scoring use the same TRAIN-normal robust-deviation rule as the other sources.

## Output Contract

Each dataset produces one compressed NPZ containing six finite one-dimensional arrays:

- `spectral_train`, `spectral_test`
- `shapelet_train`, `shapelet_test`
- `interval_train`, `interval_test`

The cache also stores a JSON metadata payload with dataset, TRAIN-derived target length, source configuration hash, seed, TRAIN/TEST row counts, and `contains_test_labels=false`. A manifest records every cache SHA256, source coverage, errors, execution command, code hash, DB hash, and frozen population hash.

No cache or manifest field may contain TP, FP, FN, F1, anomaly count, TEST label, candidate index, threshold, or lane.

## Execution And Recovery

- Run one experiment process with seven dataset workers.
- Limit BLAS/OpenMP inner threads to one per worker.
- Write each dataset cache atomically and append a checkpoint row only after hash verification.
- On restart, reuse a cache only when its dataset, configuration hash, and file hash match.
- Any missing or failed dataset prevents declaring full coverage. Failures remain visible in `ERRORS.csv`; they are not converted to empty scores.

## Validation

Unit tests cover deterministic features, TRAIN-only fitting, finite score arrays, source shapes, forbidden fields, cache tamper detection, and direct CLI execution. A 10-dataset smoke must produce 10 valid cache files and zero errors before the 1,117-dataset run starts. Independent reaggregation verifies exact population coverage, no duplicate datasets, finite arrays, matching TRAIN/TEST lengths across all three sources, and stable hashes on a replay sample.

## Interpretation Boundary

Successful Exp162 means only that three comparable, family-independent, TEST-label-free score sources are available over the frozen population. It does not prove that any source detects anomalies, that agreement improves F1, or that the pipeline is prospectively validated. Those questions require a separately preregistered Exp163 whose prediction is frozen before TEST evaluation.
