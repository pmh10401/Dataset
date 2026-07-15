# Exp177-185 Unattended Batch Design

## Goal

Run a pre-registered sequence of policy and representation experiments without
an agent monitor. The core checkpoint should finish near five hours and all
extensions must stop within twelve hours.

## Scientific Contract

- Exp137 autonomous Hard alerts remain unchanged.
- New candidates are retrospective Standard-review sensitivity only.
- Prediction reads TRAIN-normal series and TEST values, but never TEST labels.
- TEST labels are loaded only after prediction files and their SHA256 hashes are
  frozen.
- Dataset/family names are lookup keys only and never policy inputs.
- No TEST-length budget, tail replacement, or retrospective family routing.
- No alpha, feature, score backend, or fusion rule is selected from TEST
  performance. Every pre-registered variant is reported.

## Queue

1. Exp177 freezes a TRAIN-OOF risk selector over the existing four sources.
2. Exp178 evaluates the frozen selector as review-only sensitivity.
3. Exp179 builds a full-coverage canonical/multiscale statistical feature cache.
4. Exp180 compares KNN local-gap and robust diagonal-distance scores.
5. Exp181 evaluates fixed existing/new-source agreement variants.
6. Exp182 writes the five-hour checkpoint report.
7. Exp183 builds a wavelet/multiscale derivative feature cache.
8. Exp184 evaluates fixed extension fusion variants.
9. Exp185 independently reaggregates the complete batch.

## Runtime Contract

- One experiment runs at a time with seven CPU workers.
- MPS is used only by a PyTorch feature stage and only when a deterministic
  smoke benchmark is faster than CPU. Otherwise the decision is logged as CPU.
- Each stage writes progress lines, a checkpoint, a completion marker, and an
  output hash.
- Missing datasets are retried once. Remaining omissions fail that stage but do
  not silently enter aggregate metrics.
- A failed stage is recorded and independent later stages continue; dependent
  stages are skipped with a reason.
- The core queue has a five-hour soft deadline. The complete batch has a
  twelve-hour hard deadline.

## Interpretation Limit

The inherited four score configurations are not proven end-to-end strict
TRAIN-only. Results are policy-level TRAIN-only retrospective sensitivity, not
prospective or deployment validation.
