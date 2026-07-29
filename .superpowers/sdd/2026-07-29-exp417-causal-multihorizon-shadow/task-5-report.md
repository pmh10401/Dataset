# Exp417 Task 5 report — scoring, persistence, and seed consensus

## Scope

Added the pure-NumPy, synthetic-only scoring reference and its focused tests:

- `experiments/exp417_causal_multihorizon_shadow/scoring.py`
- `experiments/exp417_causal_multihorizon_shadow/tests/test_scoring.py`

No real inputs, labels, lanes, model fitting, MPS, MLX, or full-population run
were used.

## TDD evidence

- RED: `.venv/bin/python -m unittest
  experiments.exp417_causal_multihorizon_shadow.tests.test_scoring -v` failed
  at import time with `ModuleNotFoundError` for the absent `scoring` module.
- GREEN: the same focused command passed all 13 tests after implementation.

The tests use hand-derived pooled OOF median/MAD scaling, linear Q95 and Q99
percentiles, persistence, series score, and plus-one empirical p-value. They
also cover aligned-length, finiteness, zero-scale, bool/type, horizon mapping,
two-of-three persistence, frozen seed, duplicate seed, and consensus failures.

## Interface corrections

The plan's original return records required two pieces of evidence that its
listed signatures did not accept. The coordinator approved these focused,
test-proven keyword-only arguments:

- `score_horizon(..., *, horizon: int)` binds `HorizonScore.horizon` without
  inferring it from residual length or map order.
- `seed_decision(..., *, seed: int)` binds `SeedDecision.seed` without
  inferring it from sequence order; only one of the three frozen model seeds is
  accepted.

At the score boundary, the horizon triple must be possible under the frozen
formula. The exact length binding remains enforced by `ShadowCandidate`, which
is the only existing record that contains the length.

## Verification

All of the following exited successfully:

```bash
.venv/bin/python -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_scoring -v
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
git diff --check
.venv/bin/python -m compileall -q \
  experiments/exp417_causal_multihorizon_shadow
```

## Commit and concerns

Commit message: `feat: score Exp417 persistent horizon errors`.

No implementation concerns remain. The horizon and seed signature corrections
are intentionally documented above for downstream callers.
