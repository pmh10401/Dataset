# Agy Review: Exp211-214

Model: Gemini 3.5 Flash (High)

## Findings

1. Critical: per-fold empirical p-values have minimum resolution `1 / (n_calibration + 1)`. With five folds, low alpha values are unreachable for many datasets.
2. Medium: evaluation reopened SQLite for every variant row instead of once per dataset.
3. Low: the fold-cache config hash did not include implementation code hashes.
4. Verified: held-out TRAIN calibration and TEST query scores use the same fold-fitted model.
5. Verified: prediction is label-free and frozen before evaluation.
6. Verified: Exp213 source selection uses only TRAIN OOF/full parity and fixed thresholds.
7. Verified: process pool paths are dataset-specific and the sequential runner has an exclusive queue lock.

Initial recommendation: no-go until the p-value resolution and repeated label loading are addressed.

## Codex response

- Added pooled cross-conformal exceedance-count aggregation. Its minimum p-value is `1 / (n_train + 1)` instead of the per-fold floor.
- Retained median and conservative-max variants as explicitly resolution-limited research sensitivities.
- Exp213 now uses the pooled cross-conformal source p-values.
- Added `resolution_blocked` to every frozen prediction row.
- Cached TEST labels once per dataset during delayed evaluation.
- Included implementation file hashes in the fold-cache config hash.
