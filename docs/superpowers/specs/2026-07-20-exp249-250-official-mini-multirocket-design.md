# Exp249-250 Official MiniROCKET/MultiROCKET Design

## Purpose

Qualify official aeon MiniROCKET and MultiROCKET transforms as independent
TRAIN-normal score sources against the frozen Exp236 Hard, Standard, and
Priority lanes. The experiments annotate confidence and review priority; they
do not add, remove, or promote alerts.

## Frozen population and safety contract

- Population: exactly 1,117 datasets from the Exp236 frozen prediction file.
- Database: `/Users/minho/Documents/Dataset/univariate_ts.db`, read-only.
- Prediction reads TRAIN-normal and TEST values but never TEST labels.
- TEST labels are loaded only after the sorted prediction JSONL hash is frozen.
- Exp236 Hard, Standard, and Priority indices remain exactly unchanged.
- Datasets with fewer than five TRAIN-normal instances abstain.
- Tail replacement, TEST-length budgets, family routing, and TEST-driven
  configuration selection are prohibited.
- Results are retrospective research evidence, not deployment validation.

## Frozen transformer configurations

Exp249 MiniROCKET:

- requested kernels: 1,024 and 4,096;
- actual aeon dimensions: 1,008 and 4,032;
- maximum dilations per kernel: 32;
- random states: 20260721 and 20261721;
- inner `n_jobs=1`.

Exp250 MultiROCKET:

- requested kernels: 512 and 1,024;
- actual aeon dimensions: 4,032 and 8,064;
- maximum dilations per kernel: 32;
- features per kernel: 4;
- raw and first-difference representations enabled by the official transform;
- `normalise=False` because the shared loader already performs per-instance
  z-normalisation;
- random states: 20260721 and 20261721;
- inner `n_jobs=1`.

Each requested kernel setting reports two frozen variants:

1. `seed_median`: median of the two scalar OOF/query scores;
2. `seed_intersection`: an index must pass the empirical tail threshold under
   both seeds.

Frozen alpha grid: `0.005, 0.01, 0.02, 0.05`. No alpha is selected from TEST
metrics.

## Cross-fitted anomaly score

For each deterministic fold and seed:

1. select at most 1,000 fold-fit TRAIN-normal instances with a frozen seed;
2. fit the transformer on that fold-fit reference only;
3. transform fold-query instances in fixed chunks of 128;
4. robust-scale features using the fold-fit reference median and IQR only;
5. calculate KNN-3 local-gap query scores against the fold-fit reference;
6. place the scores into one OOF vector.

Fit a fresh transformer on a frozen-seed reference of at most 1,000
TRAIN-normal instances and transform TEST in chunks of 128. All TRAIN-normal
instances still receive OOF scores and remain in the empirical calibration
reference. The cap controls memory for datasets with up to 18,378 TRAIN rows
and does not use TEST labels, TEST length, or TEST performance.
The TEST set never affects alignment length, transformer parameters, scaling,
distance reference, threshold, or candidate policy.

## Qualification views and gates

For each source variant and alpha, report standalone candidates and their
intersection with frozen Exp236 Hard, Standard, and Priority lanes, plus
outside-lane candidates.

- Hard confirmation: at least 100 candidates, precision at least 89.546%.
- Standard priority: at least 50 candidates, precision at least 70%.
- Outside-lane coverage: at least 25 TP, precision at least 80%.

These gates are retrospective research gates only. Passing does not select a
deployment setting.

## Execution

- Seven outer dataset workers; one inner aeon thread per worker.
- Maximum transformer/KNN TRAIN reference: 1,000, selected deterministically.
- Query transform chunk size: 128.
- Exp249 runs before Exp250.
- Per-dataset NPZ/JSON caches are hash-verified and resumable.
- A mixed 10-dataset smoke set must pass before the full queue starts.
- The dashboard must show Exp249/250 queue, active dataset, progress, and errors.
