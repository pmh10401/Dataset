# Exp251 sktime Statistical Features Design

## Purpose

Qualify deterministic, label-free statistical feature blocks from sktime 1.0.1
as independent TRAIN-normal score sources against the frozen Exp236 Hard,
Standard, and Priority lanes. Exp251 does not change autonomous alerts.

## Frozen population and environment

- Population: exactly 1,117 datasets from the Exp236 frozen prediction file.
- Database: `/Users/minho/Documents/Dataset/univariate_ts.db`, read-only.
- Runtime: `/Users/minho/Documents/Dataset/venvs/tsai-sktime-py314/bin/python`.
- Required versions: sktime 1.0.1, scikit-learn 1.7.2, NumPy 2.4.6.
- Datasets with fewer than five TRAIN-normal instances abstain.
- TEST labels are loaded only after the sorted prediction JSONL hash is frozen.
- Family routing, dataset-name policy, TEST-length budget, tail replacement, and
  TEST-driven configuration selection are prohibited.

## Frozen feature blocks

All blocks are deterministic and have `requires_y=False` in sktime 1.0.1.

1. `catch22_22`: `Catch22(catch24=False, replace_nans=True)`, 22 dimensions.
2. `summary_9`: default `SummaryTransformer`, 9 dimensions.
3. `hurst_3`: `HurstExponentTransformer(method="rs")`, 3 dimensions. Its
   maximum lag is bounded by TRAIN-aligned series length, never TEST length.
4. `signature_6`: `SignatureMoments(degree=2, use_index=True)`, 6 dimensions.
5. `combined_40`: concatenation of the four blocks in the order above.

For every block, non-finite TRAIN values are replaced with the finite TRAIN
column median. A column with no finite TRAIN value is replaced with zero. TEST
non-finite values use the corresponding frozen TRAIN replacement. No TEST
statistic is used.

## Score and calibration

- Robust feature scaling uses TRAIN-fit 10th-90th percentile scaling.
- Score: KNN-3 local-gap, matching the Exp249/250 score definition.
- TRAIN calibration: deterministic five-fold OOF distance scoring.
- Frozen fold seed: 20260722.
- Alpha grid: 0.005, 0.01, 0.02, 0.05.
- Candidate generation uses empirical upper-tail p-values from TRAIN OOF only.

Each variant reports standalone candidates, intersections with frozen Exp236
Hard/Standard/Priority lanes, and outside-lane candidates.

## Qualification gates

- Hard confirmation: at least 100 candidates and precision at least 89.546%.
- Standard priority: at least 50 candidates and precision at least 70%.
- Outside-lane coverage: at least 25 TP and precision at least 80%.

These are retrospective research gates. Passing does not select a deployment
feature block or alpha.

## Execution and safety

- Exp251 starts only after Exp250 finishes.
- Seven outer dataset workers; no nested parallel feature execution.
- Per-dataset NPZ/JSON caches are atomic, hash-verified, and resumable.
- A mixed 10-dataset smoke test must complete before full execution.
- Prediction rows contain no labels or evaluation metrics.
- Independent reaggregation must pass before completion is recorded.
- Results remain retrospective and do not establish prospective performance.

