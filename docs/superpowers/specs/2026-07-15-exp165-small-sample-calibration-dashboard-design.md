# Exp165 Small-Sample Calibration And Dashboard Design

## Goal

Exp165 audits whether TRAIN-only score-similarity pooling can repair the
threshold-resolution collapse observed by Exp164 for datasets with
`5 <= n_train < 20`, while the dashboard exposes complete historical records
and reliable live progress for Exp165 and later experiments.

## Fixed Scientific Contract

- Population: the frozen 1,117-dataset Exp164 population.
- Target datasets: exactly the 316 datasets with `5 <= n_train < 20`.
- Abstention: the 162 datasets with `n_train < 5` remain complete abstentions.
- Donor datasets: the 639 frozen datasets with `n_train >= 20`.
- Sources: `spectral`, `shapelet`, and `interval` from Exp164.
- Alphas: `(0.005, 0.01, 0.02, 0.05)`.
- Agreement variants: `2-of-3` and `3-of-3`.
- Minimum pooled reference: 200 normalized TRAIN-normal OOF scores per source.
- Fold seed and source configuration remain inherited from Exp164.
- No TEST values, labels, positions, lengths, family names, dataset-name
  policy branches, or past TEST performance may be used.
- All results are retrospective TRAIN-normal calibration sensitivity results.
- No alpha, agreement variant, pool size, or deployment policy is selected.

## Calibration Design

Each target pseudo-run is excluded from every target-side normalization and
descriptor calculation. For each source, its query score is normalized by the
median and MAD of the other target OOF scores. A 21-dimensional descriptor is
formed from seven statistics for each of the three sources. Donor descriptors
are built only from donor TRAIN-normal OOF scores and robust-scaled with donor
median/IQR statistics.

Nearest donors are added deterministically until every source has at least 200
pooled normalized reference scores. Equal-distance donors at the cutoff are
included together. The target dataset never contributes values to its own
reference. Source flags use strict `query_score > higher_quantile_threshold`;
agreement flags require at least two or all three sources.

Exp164 local flags are retained as a frozen baseline. Exp165 reports local to
pooled transitions, threshold uniqueness, global false-review rates, dataset
Wilson upper bounds, and sample-size stratification. It does not evaluate
anomaly TP, FP, FN, recall, or F1.

## Runtime And Restartability

Exp165 reads the verified Exp164 cache manifest and cache files. Each completed
target writes an atomic target cache and a checkpoint row. Resume is allowed
only when the input hashes and configuration hash match. Missing targets,
duplicate targets, hash mismatches, source failures, or a pool below 200 stop
the run with a blocker report.

The default worker count is seven. BLAS inner threads remain one per worker.
The runner emits canonical progress lines and also publishes atomic external
dashboard state. The dashboard publisher records experiment id, phase,
dataset progress, current dataset, errors, rows, PID, paths, and timestamps.

## Dashboard Design

The dashboard treats process reality as authoritative. A stale state-file PID
must not appear as running. When a stale run has complete artifacts, it is
shown as completed; otherwise it is shown as stale/failed without blocking a
new queue run.

Historical catalog entries are added for Exp153 through Exp165 with explicit
scope labels. Audit-only experiments expose their audit metrics but are not
ranked as model F1 experiments. Multi-alpha experiments retain all variants and
must not display a selected best alpha.

Live status comes from a common atomic progress file plus the sequential
runner heartbeat. The dashboard must show Exp165 in the queue before launch,
then current phase, processed/expected datasets, current dataset, errors,
elapsed time, ETA, and resource use while running.

## Verification

- Unit tests cover robust normalization, query exclusion, deterministic donor
  selection, minimum pooled reference, and agreement flags.
- Structure tests reject TEST access and forbidden policy inputs.
- Dashboard tests reproduce stale state, historical catalog coverage, and live
  external progress.
- A smoke run uses ten targets while retaining the full donor universe.
- Full execution starts only after all tests and smoke checks pass.
