# Exp297 Bootstrap Confidence Stability and Dashboard Design

## Purpose

Exp297 tests whether the Exp296 cross-Gate confidence tier remains stable when TRAIN-normal OOF runs are resampled. It also exposes the frozen tier and stability rate in the local dashboard without changing any automatic alert or review lane.

## Frozen Inputs

- Exp295 frozen Gate annotations and verified expert support
- Exp296 frozen consensus tiers
- provenance-verified TRAIN OOF score caches used by Exp293

All inputs are SHA256-verified. Masked reconstruction remains excluded from the primary Gate.

## Bootstrap Contract

- bootstrap replicates: 32
- random seed: 20260722
- sampling: with replacement, sample size equal to each dataset's TRAIN-normal count
- the same sampled TRAIN indices are applied to every verified expert and variant for that dataset and replicate
- datasets with fewer than five TRAIN-normal runs remain Gate-abstained and receive no numeric stability claim
- conformal LOO p-values are calculated with an exact rank-equivalent O(n log n) implementation

For each replicate, expert TRAIN risk and cross-variant stability are recomputed, the three Gate variants reselect two experts, and every frozen Hard index receives a replicated consensus tier. Alert stability is the fraction of eligible replicates matching the frozen Exp296 tier.

## Prediction and Evaluation Separation

Prediction contains frozen Hard indices, consensus tier, bootstrap tier counts, eligibility, and stability rate. It reads no TEST labels or evaluation metrics. Evaluation verifies the frozen prediction hash before loading labels and reports diagnostics only.

## Dashboard

The dashboard adds an `자동 경보 신뢰도` panel containing:

- four Exp296 tier counts
- median bootstrap stability for eligible alerts
- eligible and abstained alert counts
- a compact alert table with dataset, index, Korean tier name, and stability percentage
- a tier filter; the complete frozen JSONL remains downloadable from experiment history

The panel explicitly states that all rows remain automatic Hard alerts and no suppression policy is active.

## Invariants and Limits

- exactly 1,117 datasets and 2,218 unchanged Hard alerts
- no TEST label, family name, dataset-name policy, or TEST-length policy in prediction
- no review promotion, no Hard demotion or suppression
- bootstrap configuration is fixed before Exp297 evaluation
- Exp297 is retrospective and cannot establish deployment validity
- the Exp296 tier definition was informed by prior retrospective analysis and must not be described as prospectively preregistered
