# Exp238 ROCKET Dimension Confidence Design

## Goal

Evaluate whether increasing ROCKET kernels from 256 to 512 and 1,024 improves confidence labeling for the frozen Exp236 Hard and Standard review lanes. Exp238 never adds, removes, or promotes an alert.

## Frozen computation

- Population: the same 1,117 datasets as Exp236.
- Database: `/Users/minho/Documents/Dataset/univariate_ts.db`, read only.
- TRAIN-normal values determine alignment, normalization, reference sampling, score calibration, and thresholds.
- Kernel counts: 256, 512, and 1,024 from one deterministic nested kernel stream with base seed `20260706`.
- Stability probe: an additional independent 256-kernel stream with seed `20261706`.
- Each kernel emits maximum convolution response and PPV, so feature dimensions are 512, 1,024, and 2,048.
- Score: KNN-3 local-gap fitted on the TRAIN-normal reference.
- Alpha grid: `0.005, 0.01, 0.02, 0.05`; every alpha is reported and none is selected from TEST results.
- Datasets with fewer than five TRAIN-normal rows abstain from confidence calibration.

## Confidence display policy

ROCKET dimensions are one correlated evidence family and never count as multiple independent votes.

- `rocket_stable`: candidate is supported by 256, 512, 1,024, and the independent-seed 256 stream.
- `independent_auxiliary`: candidate is supported by either Exp237 shrinkage normal-space or elastic normal prototype at the same alpha.
- Confidence A: frozen Exp236 Hard candidate supported by both `rocket_stable` and `independent_auxiliary`.
- Confidence B: frozen Exp236 Hard candidate supported by either evidence family, but not both.
- Confidence C: remaining frozen Exp236 Hard candidate.
- Standard review priority 1/2/3 uses the same both/either/neither rule within the frozen Standard review lane.

The union of A/B/C must equal Exp236 Hard exactly. The union of review priority 1/2/3 must equal Exp236 Standard exactly. Priority review remains unchanged and is not promoted.

## Evaluation

Prediction is frozen and hashed before labels are loaded. Retrospective evaluation reports:

- each ROCKET dimension's confirmation coverage and precision inside Exp236 Hard;
- confidence A/B/C TP, FP, precision, and coverage;
- Standard review priority strata TP, FP, and precision;
- TRAIN-normal exceed rate and p-value resolution;
- score and candidate correlation/Jaccard between dimensions;
- incremental candidates outside Exp236 only as a diagnostic, never as alerts.

No TEST metric chooses a dimension, alpha, grade rule, or deployment policy. Results are retrospective and not prospective equipment validation.
