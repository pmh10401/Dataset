# Exp421 MPS Batched Acceleration Design

Date: 2026-08-01
Status: approved design; implementation pending

## 1. Purpose

Exp421 validates a runtime-only acceleration path for the frozen Exp417 causal
multi-horizon Shadow workflow. It targets the repeated CPU cost in model
materialization, single-series synthetic inference, and repeated TRAIN OOF
calibration. It does not change the anomaly model, threshold, fold layout,
seed set, qualification rule, frozen lanes, or publication meaning.

The official Exp419 CPU run remains immutable and continues independently.
Exp421 produces only local diagnostic and benchmark evidence until every gate
below passes. A future full-population run would require its own explicit
preregistration and a new no-clobber output root.

## 2. Scope and non-goals

Exp421 may change only execution internals:

- reuse one materialized model per `(length, seed, device)` batch;
- score several TRAIN synthetic inputs in one bounded batch;
- cache deterministic primary perturbations;
- cache normal-OOF calibration values for each `(length, seed, horizon)`;
- select CPU or MPS from a predeclared, TRAIN-only backend policy;
- record backend, device transfer, synchronization, parity, and fallback evidence.

Exp421 must not:

- read TEST labels, anomaly locations, human review, or historical TEST metrics;
- alter Exp417 architecture, horizons `(1, ceil(.02L), ceil(.10L))`, loss,
  optimizer, model seeds, perturbation seeds, alpha, persistence, or gates;
- alter Hard, Standard, Priority, or automatic-normal lanes;
- use tail replacement or retrospective threshold/source selection;
- change the official Exp419 process or reuse its partial checkpoints;
- publish predictions, metrics, or private dataset records to GitHub.

The prototype will not install MLX. The current MLX package is absent, and
implementing the PyTorch causal Conv1d plus AdamW contract in MLX would be an
algorithm rewrite rather than an exact backend substitution. The manifest must
still record the MLX eligibility result, package/version absence, device,
timing status, parity status, and fallback reason.

## 3. Evidence and working hypothesis

The existing frozen Exp417 backend audit retained CPU because its aggregate MPS
speedup was `0.6709759390840965x`, despite acceptable numerical parity. A new
normal-TRAIN diagnostic showed a strong batch-size dependency for the exact
causal CNN training step: batch 1 was approximately `0.43x`, batch 4 about
`1.73x`, batch 8 about `2.76x`, batch 16 about `8.67x`, and batch 32 about
`17--27x` CPU-over-MPS. These smoke measurements were made while official CPU
workers were active and are not adoption evidence.

The hypothesis is therefore conditional:

1. MPS is unsuitable for one-series-at-a-time inference.
2. MPS may accelerate OOF/final training and bounded batched qualification.
3. A hybrid policy must retain CPU for small batches and any failed parity or
   speed gate.

## 4. Architecture

### 4.1 Frozen model adapter

Add a narrow adapter around the existing `CausalMultiHorizonCNN` that accepts a
validated frozen state and materializes it once. The adapter exposes:

- `predict_batch(channels: float32[N, 2, L])` returning the same three horizon
  outputs as the existing single-series path;
- `fit_batch(...)` using the existing Huber loss, AdamW settings, and one-thread
  deterministic execution;
- explicit CPU/MPS device assertions and CPU state export for model hashes.

The existing Exp417 single-series function remains the reference path. Exp421
must compare the adapter against that reference before using it in any runtime.

### 4.2 Calibration cache

For each exact length group and frozen model seed, precompute from normal OOF
residuals only:

- pooled robust center and scale;
- normalized OOF residuals;
- 99th-percentile exceedance threshold;
- normal series scores used by the empirical plus-one p-value.

The cached values must reproduce `score_horizon` exactly within the frozen
tolerance. Query scoring may run on CPU NumPy/Numba after model inference; no
GPU transfer is needed for score aggregation.

### 4.3 Synthetic input cache

For each TRAIN series, create `level_shift`, `gradual_drift`,
`amplitude_scaling`, and `period_change` once. Create the seed-dependent
`secondary_spike` separately for each frozen perturbation seed. Cache entries
are immutable float32 channels and are keyed by exact dataset, length,
instance identity, perturbation name, and seed where applicable.

### 4.4 Hybrid backend policy

The policy is chosen before any future full run from a new TRAIN-only benchmark.
It is not tuned from TEST results. Candidate batch sizes are exactly
`1, 4, 8, 16, 32`; the selected policy is recorded in the manifest.

- CPU is mandatory for batch size 1 and is the fail-closed fallback.
- MPS may be selected only for a batch size whose full stage, including input
  transfer and `torch.mps.synchronize()`, passes all gates.
- A mixed run records the backend for every exact length group and seed stage.
- MPS-heavy work is limited to one process; it is not combined with outer
  multiprocess MPS execution without a separate deterministic benchmark.

## 5. Data flow and boundary

The prototype loads normal TRAIN values only. It may use the already frozen
database digest and population identity, but it must not build a TEST identity
manifest or invoke a TEST query. A future full run preserves the Exp417 order:

```text
normal TRAIN load
  -> exact-length preprocessing
  -> identity-grouped OOF and TRAIN calibration
  -> TRAIN synthetic qualification
  -> freeze TRAIN contract
  -> only then values-only TEST load
  -> score, lane exclusion, workload gate
```

Any context, SQL, or artifact that contains TEST labels or evaluation metrics
causes immediate fail-closed termination.

## 6. Acceptance gates

### Numerical and scientific parity

For each candidate batch size and representative length/count stratum:

- all CPU and MPS outputs and losses are finite;
- CPU and MPS candidate coordinates are identical;
- normalized maximum score error is at most `1e-4`;
- Spearman rank correlation is at least `0.999`;
- model state hash, stop epoch, loss history, OOF residual bytes, calibration
  digest, and qualification decisions match the CPU reference where the same
  operation is compared.

### Runtime gate

Timing includes CPU-to-device transfer, device-to-CPU transfer, and MPS
synchronization. The exact stage speedup must be at least `1.1x` CPU-over-MPS.
The benchmark must run in a fresh isolated attempt without concurrent official
CPU workers. A smoke that passes only because the host is contended is invalid.

### Provenance gate

The manifest records Git commit, database SHA-256, frozen scientific config,
device, PyTorch, MLX and mlx-metal versions (or explicit absence), batch policy,
timings, parity values, fallback reason, and source digests. No partial or
unreviewed artifact is eligible for a future official run.

## 7. Failure and rollback

If any parity, candidate identity, finite-output, provenance, memory, or speed
gate fails, Exp421 retains the CPU reference and records the exact reason. It
does not silently fall back while claiming MPS adoption. The official Exp419
run is never restarted from an Exp421 prototype checkpoint. A future full run
uses a new attempt root and can be approved only after the complete manifest
passes review.

## 8. Verification plan

Tests must cover:

1. batch size 1/4/8/16/32 output and candidate parity;
2. calibration-cache equivalence to `score_horizon`;
3. perturbation-cache identity and seed isolation;
4. CPU fallback for small batches, unavailable MPS, nonfinite output, and speed
   failure;
5. no-TEST SQL and artifact boundary;
6. manifest schema and source-digest validation;
7. bounded normal-TRAIN benchmark with transfer and synchronization timing.

Only after these tests pass will a separate implementation plan and code review
be prepared. No Exp419 production execution is part of Exp421's prototype.
