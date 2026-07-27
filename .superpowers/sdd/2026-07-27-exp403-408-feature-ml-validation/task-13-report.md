# Task 13 Report: Smoke, Parity, and Sequential Full Validation

## Status

Full sequential validation completed with run ID
`164d917e86bb4aeaa04b1f7a6bb5d326`, using the designated full output only.
Every data-bearing stage completed all `1,117` datasets with `errors=0`; the
aggregate report also completed. The process exited normally.

## Implementation Fixes And Commits

- `f864ac2a27881486a27687fd7747c8a5aba342b9` - smoke CLI, scoped validation,
  serialization, and representative selection hardening.
- `ea6e005d67bf02a76c1b5d69f3f9d28b8d646e37` - Exp408 structured
  all-source-abstention artifact and cross-stage identity metadata/regressions.
- `b6647da019847984bac0128f581010000b5c780c` - defer no-source classification
  until source loading completes.
- `54627c48aabbf56e39e6fb6eaa3a9f558676ddce` - restrict the database order to
  the hash-verified frozen 1,117-dataset universe.
- `d302d460abcaa7de3b9c8970a1ce3b8ea93ff83b` - stream per-dataset input hash,
  resume validation, and processing instead of blocking on a global prepass.
- `fca9a1d88445799126ee615a0f7c8bcba3b78628` - record a valid audit abstention
  below two TRAIN rows instead of invoking cross-fit.
- `30e533a86b2cbd208746890af4588996dbc1a635` - support 15-point runs in the
  fixed 16-band Exp406 views without creating an empty-band NaN.

Valid all-source abstentions preserve frozen Hard, Standard, and Priority,
record source status/reason, and emit zero Shadow candidates for all five
variants at every alpha. `stage_error`, missing, and corrupt sources remain
errors.

The interrupted first full launch exposed two database-only aliases outside the
verified universe: `CornellWhaleChallenge` and `Wafer_normal_1`. The final
selector preserves the existing small-to-large order while excluding them; no
frozen constants changed.

The hash-streaming remediation retains the exact content/config/code/upstream
hash binding. Serial and learned stages now hash, resume-validate, and write
one dataset before moving on. CPU parallel stages submit each stale dataset as
soon as its hash is available while the parent remains the sole checkpoint and
status writer. The regression suite proves worker processing begins before all
slow hash calls finish; resume skips and one-dataset invalidation remain exact.

## Commands And Tests

Completed after the final implementation fix:

```bash
python3 -m py_compile experiments/exp403_408/*.py scratch/run_exp403_408_feature_ml_validation.py
python3 -m unittest discover -s experiments/exp403_408/tests -p 'test_*.py' -v
python3 -m unittest discover -s experiments/exp400_402 -p 'test_*.py' -v
git diff --check
```

Compilation, queue registration, and diff checks passed. Final Exp403-408:
`148` tests passed in `28.024s`; Exp400-402: `18` tests passed. The queue test
reported `Exp403-408 queue and dashboard registration: PASS`.

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

## Full Sequential Results

Command:

```bash
python3 scratch/run_exp403_408_feature_ml_validation.py \
  --stage all --workers 7 --device auto --reset-checkpoint \
  --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation
```

Wall runtime from `/usr/bin/time -p`: `6832.25s`. Stage status timing was
approximately audit `580s`, classical `368s`, SSL `1055s`, synthetic `3726s`,
consensus `1077s`, and report `19s`.

| Invariant | Result |
| --- | ---: |
| Dataset/checkpoint coverage per data stage | `1117/1117` |
| Audit JSONL / consensus JSONL rows | `1117 / 1117` |
| Stage errors | `0` |
| SSL abstentions | `616` (`n_train_below_50`) |
| Synthetic abstentions | `616` (`n_train_below_50`) |
| Completed learned datasets per source | `501` |
| Frozen Hard / Standard / Priority | `2218 / 806 / 31` |
| Forbidden prediction fields | `0` |
| Independent reaggregation mismatches | `0` |
| Result CSV rows | `5585` |

The read-only `verify_prediction` regeneration was run twice. Both verified
row sequences and manifests were identical, and the canonical sorted Exp404
prediction SHA256 exactly matched the manifest on both reads:

`437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`.

No test labels or evaluation metrics appeared in the prediction publication.
All frozen lanes were preserved; Exp406-408 candidates remain review-only.

Only the designated smoke output was reset for representative reruns. The
interrupted designated full output was reset before each corrected 1,117-dataset
run; no prior experiment output directory was modified.

## Fix Round 1 Closure

- Final report refresh run ID: `95a8449547124c0f9a97feca9aac725a`; it reused
  verified full upstream publications and did not rerun a learned full stage.
- Final representative smoke run ID: `5c61c5c193604deb94a104e37b40f694`;
  elapsed wall time from run-bound timestamps was `70.028s`. It has exactly
  20 checkpoints in every data stage, 0 errors, and tier coverage abstain /
  robust-distance / classical-committee / learned-methods = `5 / 5 / 5 / 5`.
- The selector uses canonical normal-TRAIN SQLite counts only. It neither
  queries TEST rows nor retains a TEST-derived input ordering. Every smoke
  checkpoint agrees on dataset ID, `n_train`, and eligibility.
- Final smoke hashes: Exp404 prediction
  `b5b55a9cb63fc2d7f557d613d5d2608a7b264cf0f715b59e6f4a31dbc5568ec5`;
  Exp408 consensus
  `f4fb4317d547b0d9162d3687f8fd7680a288aa8acbfb3d57dae80bfef94db633`.
- Full consensus regeneration hash:
  `c4016db7a3c17d130b5bd8796e0139325cb63b0645bfe3f0ca7b96c825339d23`.
  It contains `162` valid all-source-abstention rows, each with every
  FeatureBlock, SSL, and synthetic status/reason recorded.
- Frozen MPS benchmark artifact:
  `task13_exp406_mps_benchmark.json`, SHA256
  `81d606ac6210f3b2edf03ec29a688b65623390d4e036336bc938e5ab8f417249`.
  `CricketX_normal_1` (`n_train=52`, `n_test=50`, seed `20260727`) recorded
  CPU `1.017256s`, `0.315069s` and MPS `2.408724s`, `1.953296s`; all ranks
  matched, maximum raw-score delta was `2.384185791015625e-07`, and all four
  published score hashes were
  `200d69cf62e2a0282d1a7e4099ac79af3dce4d012f230748bad46e3abbcbe6d8`.
  CPU mean was `0.666162s` versus MPS mean `2.181010s`, so CPU was retained
  because parity passed but MPS was slower.
- The manifest records the exact historical full command
  `python3 scratch/run_exp403_408_feature_ml_validation.py --stage all --workers 7 --device auto --reset-checkpoint --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation`
  and the report command
  `python3 scratch/run_exp403_408_feature_ml_validation.py --stage report --workers 7 --device auto --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation`.
- `bbeea9a405a3cb013ba50fb4f8878567279416bf` added the TRAIN-only selector,
  consensus accounting, and worker policy. `41e0b19209fcc1c7a2bc0dbb07548599c226e2f3`
  fixed completed learned-source classification. The reset cleanup wrapper now
  deletes only checkpoint/status state in the explicitly supplied output; it is
  outside the pipeline file frozen by the full prediction contract.
