# Exp443 Task 10 brief — post-freeze evaluation

## Purpose

Measure Policies A, B, and C only after all three label-free predictions are
frozen and replayed. The evaluator is retrospective and review-only. It must
not change any TRAIN policy, score, candidate, threshold, route, or prediction
byte after seeing TEST labels.

## Fixed boundary

1. Inspect the publication twice before importing or opening any label-capable
   code.
2. Open one pinned database snapshot and issue exactly one ordered TEST-label
   query for each of the 1,117 frozen datasets.
3. Require contiguous instance indices and binary labels, close the database,
   then inspect the publication and frozen source again.
4. Reproduce the fixed baseline confusion `1915 / 303 / 2003 / 191679` and
   micro F1 `0.6241851368970013`; Policy A must be coordinate-identical.
5. Evaluate A, B, and C independently. Never select a winner inside Exp443.

## Metrics and gate

Dataset metrics use `0.0` when a denominator is zero. Macro metrics are simple
means over all 1,117 datasets. A rescued dataset has baseline TP 0 and policy
TP at least 1. A regressed dataset has lower F1; a strong regression also has
baseline F1 at least 0.8.

The paired dataset bootstrap is frozen to NumPy 2.3.5, PCG64 seed
44320260809, 20,000 samples, int64 indices, endpoint false, and linear 2.5/97.5
percentiles. A policy passes only if both point estimates improve, both lower
confidence bounds are positive, FP does not exceed 303, at least one dataset
is rescued, no strong dataset regresses, all 1,098 unaffected datasets remain
coordinate-identical, and pre-freeze label access/errors/nonfinite counts are
zero.

## Implementation and test order

- First add synthetic RED tests for metrics, classifications, transitions,
  family grouping, deterministic paired sampling, label-boundary ordering,
  baseline/Policy-A authority, invariance, gates, and create-only replay.
- Then implement `bootstrap_antigravity.py` and `evaluation_antigravity.py`.
- Do not open the official TEST labels or create an official evaluation during
  this task. The official run belongs to the later CLI/run task.
- Regenerate `codemap.html`, `codemap.json`, and `codemap.lock` together and
  run the focused plus full Exp443 suites before completion.

## Code-map answers

- What calls it? No mapped external caller yet; Task 11 will add the lazy
  `evaluate` CLI branch.
- What does it affect? Only post-freeze evaluation artifacts. Frozen
  publication and prediction bytes remain read-only inputs.
- Which tests cover it? The 16 existing Exp443 suites protect upstream
  contracts; Task 10 adds `test_bootstrap_antigravity.py` and
  `test_evaluation_antigravity.py`.
