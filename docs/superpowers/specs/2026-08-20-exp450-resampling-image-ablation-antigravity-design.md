# Exp450 Resampling and Image Construction Ablation Design

## Status

Written design awaiting user review. No Exp450 implementation or experiment run is authorized by this document alone.

## Purpose

Exp450 measures how much the proposed detector depends on linear interpolation in two places:

1. time-series length alignment;
2. fixed-size image-feature construction.

The experiment produces paper-ready continuous anomaly-score comparisons, AP/AUROC tables, ROC curves, and precision-recall curves. Exp449 remains immutable and is the control. Exp450 is an ablation study, not an automatic replacement or promotion of Exp449.

## Scope

### Included

- Four preregistered preprocessing variants on the same frozen dataset population.
- A 23-dataset values-only smoke run followed, if successful, by the complete 1,117-dataset run.
- One continuous anomaly score per TEST row, with larger values meaning more anomalous.
- Comparison with every public baseline present in the frozen Exp449 official-comparison manifest.
- AP, AUROC, ROC, precision-recall, paired per-dataset differences, and win/tie/loss summaries.
- MPS-first execution for tensor-eligible operations.

### Excluded

- Modification or recomputation of Exp449 official artifacts.
- Replacement of the existing internal missing-value repair rule. Exp450 changes length alignment and image construction only.
- TEST-label use before every A--D score root is complete and frozen.
- Hyperparameter search, dataset-specific method selection, or automatic promotion.
- CPU--MPS numerical-equivalence testing, per the user's explicit instruction.
- Additional security or operational metadata that does not protect result correctness or resumability.

## Experimental Conditions

All conditions use the same raw rows, source definitions, random seeds, folds, thresholds, routes, and score orientation. Only the two declared preprocessing factors vary.

| ID | Length alignment | Image construction |
|---|---|---|
| A | Existing linear interpolation | Existing linear resizing |
| B | Polyphase downsampling + PCHIP upsampling | Existing linear resizing |
| C | Existing linear interpolation | Area/local-mean pooling |
| D | Polyphase downsampling + PCHIP upsampling | Area/local-mean pooling |

Condition A is the Exp449-compatible control. The primary scientific contrasts are B−A, C−A, and D−A. The B/C comparison helps distinguish the two factors but is secondary.

## Design Choice Rationale

The selected hybrid is preferred because polyphase filtering protects downsampling from aliasing, PCHIP preserves local shape during upsampling without cubic-spline overshoot, and adaptive average pooling avoids inventing linear transitions in fixed-size images. Pure FFT resampling is not used because it assumes periodic structure and can ring at boundaries. Nearest-neighbor resizing is not used because it creates block artifacts. A learned resampler is excluded because it would add training, tuning, and a new source of selection bias to what should remain a controlled preprocessing ablation.

## Exact Transform Definitions

### Length alignment

- If source and target lengths are equal, return the input unchanged.
- If the source has one finite value, repeat that value to the target length.
- For downsampling, use `scipy.signal.resample_poly` with integer up/down factors reduced by the greatest common divisor. Keep SciPy's deterministic FIR antialiasing path and require the result to have the exact target length; an unexpected length is an error rather than a silent trim or pad.
- For upsampling, use `scipy.interpolate.PchipInterpolator` over normalized positions `[0, 1]`, with extrapolation disabled. Evaluate only within `[0, 1]`.
- Existing nonfinite-value repair occurs before this transform and remains unchanged.
- Existing target-length rules remain unchanged, including the legacy route's frozen TRAIN+TEST length median where currently used. No TEST labels are used.

### Image construction

- Replace only the fixed-size resizing step. The feature definitions before resizing remain unchanged.
- Use PyTorch adaptive average pooling for one- and two-dimensional fixed-size conversion. When the requested output is larger than the input, the adaptive windows may overlap; no linear interpolation is applied.
- For recurrence plots, first produce the fixed-length one-dimensional representation using adaptive average pooling, then recompute the pairwise-distance matrix. Never resize a completed recurrence matrix.
- Spectrogram and other matrix features use adaptive two-dimensional average pooling to the frozen image size.
- Output shapes, channel order, dtype conversion, and downstream estimators remain frozen.

## Backend Policy

MPS is the canonical tensor backend whenever it is available at readiness time.

- MPS: adaptive pooling, recurrence distance matrices, and other dense tensor image operations.
- CPU: SciPy polyphase filtering, PCHIP, and existing scikit-learn/aeon estimators that do not have an unchanged MPS implementation.
- PyTorch is the tensor runtime. Record its version, macOS version, MPS availability, device name, dtype, and stage timings.
- MLX availability and suitability are assessed and recorded, but MLX is not used because the user selected MPS and no second accelerated implementation is needed.
- No CPU--MPS score or candidate parity gate is performed.
- If MPS is unavailable before a run starts, use and record the CPU fallback for tensor stages. Do not combine MPS-backed and CPU-fallback score roots in one comparison as if they were the same backend.
- The backend choice is frozen before a run. A mid-run MPS failure stops the run; it does not silently switch tensor stages to CPU. Resume uses the same backend.
- Run one MPS-heavy process at a time. Do not combine outer multiprocessing with concurrent MPS jobs.
- CPU-only parallel stages use `max(1, logical_cpu_count - 1)` workers and record both counts. Inner numerical thread counts remain one where the existing pipeline requires it.

## Data and Execution Flow

1. Build a values-only catalog with the same dataset and row identities used by Exp449.
2. Run A--D on the frozen 23-dataset smoke inventory without opening TEST labels.
3. Promote to the 1,117-dataset run only if all four variants complete with finite TEST scores, exact expected row counts, fixed output shapes, and no MPS out-of-memory or backend change.
4. Run all four full variants and persist one checkpoint per dataset × condition so a failure loses at most one condition's work for one dataset.
5. Freeze and independently verify all four complete score roots.
6. Only then open the frozen TEST labels once for evaluation.
7. Reuse the already frozen public-baseline continuous-score artifacts when their manifest proves identical database, dataset, and TEST-row identities. Do not retrain them for Exp450.
8. Write a new Exp450 report and figures; do not modify Exp449 reports or results.

The smoke promotion rule is operational only. It does not inspect AP, AUROC, labels, or whether a variant appears better.

## Continuous Scores and Metrics

Every TEST row receives a finite `float64` anomaly score with the invariant `higher = more anomalous`. Thresholded alerts are retained for compatibility but do not define AP or AUROC.

### Primary summaries

- Macro AP: arithmetic mean of valid per-dataset AP values.
- Macro AUROC: arithmetic mean of valid per-dataset AUROC values.
- Paired per-dataset changes for B−A, C−A, and D−A, with 20,000 bootstrap resamples over datasets using seed `45020260820` and percentile 95% confidence intervals.
- Win/tie/loss counts using absolute metric difference `<= 1e-12` as a tie.

### Secondary summaries

- Micro AP and micro AUROC from all eligible TEST rows pooled together.
- Per-dataset AP/AUROC distributions and dataset-size-stratified summaries.
- Existing thresholded alert metrics, reported separately from continuous-score metrics.

Datasets without both classes are excluded from AUROC and reported in the denominator audit. Datasets without positive TEST examples are excluded from AP and likewise reported. No excluded row or dataset is silently dropped.

### Figures

- ROC: x-axis = false-positive rate, y-axis = true-positive rate.
- Precision-recall: x-axis = recall, y-axis = precision.
- A--D and every eligible frozen public baseline appear in both comparisons.
- The main figure uses pooled micro curves; a companion panel/table reports macro per-dataset metrics so large datasets do not dominate the interpretation.
- Figure data are written as canonical tables before plotting, allowing the plots to be reproduced independently.

## Artifacts

Exp450 creates a new isolated root containing:

- the values-only catalog and backend decision;
- per-dataset × condition score checkpoints;
- four frozen score manifests;
- evaluation tables with per-dataset and aggregate metrics;
- canonical ROC/PR curve-coordinate tables;
- publication PNG, SVG, and HTML figures;
- a Korean paper-results summary and a methods appendix describing the interpolation ablation.

Every new repository filename contains `antigravity`. Runtime artifact names follow the same rule. The implementation creates `docs/codemap/codemap_exp450_antigravity.html`, `.json`, and `.lock` together before the first scientific source edit; Exp449 maps remain unchanged.

## Failure and Resume Rules

- Persist a dataset × condition checkpoint immediately after that output passes finite/shape/identity validation.
- Record a failure and continue with the remaining dataset × condition jobs; retry failed and never-attempted jobs in a second bounded wave.
- Do not run evaluation or create final manifests until all 1,117 datasets succeed.
- A code, transform, seed, dataset, backend, or environment change requires a new root. Same-context transient failures reuse verified successes.
- Progress logs report completed/failed/remaining datasets, current variant and stage, elapsed time, and estimated remaining time.

## Implementation Boundaries

- Add an isolated Exp450 namespace and runner; do not edit Exp449 result artifacts.
- Keep new resampling and pooling functions pure and independently testable.
- Reuse existing frozen data readers, score contracts, and evaluation utilities only where their identities and semantics remain unchanged.
- If an existing Exp449 scientific module cannot accept the new transform without altering its behavior, copy only the minimal pure closure into Exp450 and record the origin path and SHA-256.
- Do not add general-purpose security infrastructure or redesign the Exp449 lifecycle.

## Verification

Before the 23-dataset smoke:

- fixture tests for equal length, singleton input, downsampling, upsampling, short series, exact shapes, finite outputs, and recurrence recomputation;
- tests that A remains byte-compatible with the current linear control on fixed fixtures;
- tests that B--D never call linear resizing in their replaced factor;
- MPS availability and one small tensor-stage completion test, without CPU parity;
- scorer-free artifact and resume tests;
- Exp450 codemap, configuration, and filename-policy checks.

Before full evaluation:

- all four 1,117-dataset score manifests independently verify;
- dataset and TEST-row identities match across A--D and reused baselines;
- no TEST label module was imported before score freeze;
- AP/AUROC and curve axes are independently checked from canonical curve tables.

## Interpretation and Promotion

Exp450 answers whether replacing linear interpolation changes ranking quality and robustness. It does not automatically declare a new production method. Any later replacement of Exp449 requires a separate, explicitly approved experiment based on the preregistered Exp450 evidence.
