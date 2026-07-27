# Task 13 Report: Smoke, Parity, and Sequential Full Validation

## Status

Full sequential validation is live, not complete. The only active Exp403-408
process writes to `/Users/minho/Documents/Dataset/exp403_408_feature_ml_validation`.
At the recorded live checkpoint it is in `audit-features`, `0/1117`, with
`errors=0`, run ID `62221a6ecbee44a49eff518064875050`.

No conclusion is drawn from the unfinished full run.

## Implementation Fixes And Commits

- `f864ac2a27881486a27687fd7747c8a5aba342b9` - smoke CLI, scoped validation,
  serialization, and representative selection hardening.
- `ea6e005d67bf02a76c1b5d69f3f9d28b8d646e37` - Exp408 structured
  all-source-abstention artifact and cross-stage identity metadata/regressions.
- `b6647da019847984bac0128f581010000b5c780c` - defer no-source classification
  until source loading completes.
- `54627c48aabbf56e39e6fb6eaa3a9f558676ddce` - restrict the database order to
  the hash-verified frozen 1,117-dataset universe.

Valid all-source abstentions preserve frozen Hard, Standard, and Priority,
record source status/reason, and emit zero Shadow candidates for all five
variants at every alpha. `stage_error`, missing, and corrupt sources remain
errors.

The interrupted first full launch exposed two database-only aliases outside the
verified universe: `CornellWhaleChallenge` and `Wafer_normal_1`. The final
selector preserves the existing small-to-large order while excluding them; no
frozen constants changed.

## Commands And Tests

Completed after the final implementation fix:

```bash
python3 -m py_compile experiments/exp403_408/*.py scratch/run_exp403_408_feature_ml_validation.py
python3 -m unittest discover -s experiments/exp403_408/tests -p 'test_*.py' -v
python3 -m unittest discover -s experiments/exp400_402 -p 'test_*.py' -v
git diff --check
```

Compilation and diff checks passed. Exp403-408: `144` tests passed in
`26.020s`. Exp400-402: `18` tests passed in `0.050s`.

Representative smoke command:

```bash
python3 scratch/run_exp403_408_feature_ml_validation.py \
  --stage all --dataset-limit 20 --representative-smoke --workers 2 \
  --device cpu --reset-checkpoint \
  --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation_smoke
```

Smoke runtime: `84.606s` from run-bound status timestamps.

Current full command:

```bash
python3 scratch/run_exp403_408_feature_ml_validation.py \
  --stage all --workers 7 --device auto --reset-checkpoint \
  --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation
```

## Representative Smoke Results

- Coverage: `20`; TRAIN tiers: abstain `5`, robust-distance `5`,
  classical-committee `5`, learned-methods `5`.
- Cross-stage `dataset`/`n_train`/eligibility checkpoint parity passed for
  all 20 datasets.
- Stage errors: `0`. Valid all-source abstention artifacts: `5`; every
  variant/alpha candidate list is empty for each.
- Scoped lane totals preserved: Hard `17`, Standard `3`, Priority `0`.
  Report validation and forbidden-prediction-field validation passed.
- Independent reaggregation mismatch count: `0`.
- Exp404 prediction SHA256:
  `f32bec225f85931c1db8f11e666115b8a0239165ce459a46918d5d72325232d9`.
- Exp408 consensus SHA256:
  `1287681b0ae04a1050854b7b9d19435733e7b3c94725aeb563e8ae2bace1a22d`.

## CPU/MPS Exp406 Benchmark

Dataset `CricketX_normal_1`: `n_train=52`, `n_test=50`, same frozen
config and seed. The raw values were captured inside the accepted MPS parity
path.

| Run | Runtime (s) | Published prediction SHA256 | Raw rank match | Max raw difference |
| --- | ---: | --- | --- | ---: |
| CPU 1 | 0.875889 | `22ccb5a12ba1703a6bfbea43de0ec1ca92d431fa7b7d309049b2571d5542eb6c` | yes | 0 |
| CPU 2 | 0.275877 | `22ccb5a12ba1703a6bfbea43de0ec1ca92d431fa7b7d309049b2571d5542eb6c` | yes | 0 |
| MPS 1 | 2.840222 | `22ccb5a12ba1703a6bfbea43de0ec1ca92d431fa7b7d309049b2571d5542eb6c` | yes | `2.384185791015625e-07` |
| MPS 2 | 1.937345 | `22ccb5a12ba1703a6bfbea43de0ec1ca92d431fa7b7d309049b2571d5542eb6c` | yes | `2.384185791015625e-07` |

The rank and `<= 1e-5` gates passed. MPS was not selected for the full
publication run because its accepted path was slower and publication is
CPU-canonical; `--device auto` resolves to CPU.

## Full Verification Pending

The final checks await the live run: exact 1,117 coverage; zero errors or
explicit no-missing-row abstentions; Hard/Standard/Priority totals
`2218/806/31`; zero forbidden fields; independent mismatch `0`; and exact
repeat sorted-prediction hash match.

Only the designated smoke output was reset for representative reruns. The
interrupted designated full output was reset before the corrected 1,117-dataset
run; no prior experiment output directory was modified.
