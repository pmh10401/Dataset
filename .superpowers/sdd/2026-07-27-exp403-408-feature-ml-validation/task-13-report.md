# Task 13 Report: Final Validation Evidence

## Status

Task 13 completed. The frozen full all-stage publication was retained; the
current code reran the designated 20-case representative smoke and republished
only the full classical evaluation/provenance, Exp408 consensus, and report
from verified upstream checkpoints. No audit, classical prediction, SSL, or
synthetic full stage was rerun during the final repair.

The final source contract accounts for exactly 13 expected sources per dataset:
9 registered FeatureBlocks, tri-view SSL, and 3 synthetic models. Explicit
historical-coverage abstentions are recorded but never vote. Integrity failures
(invalid/corrupt payloads, hash or identity mismatch, forbidden/label-derived
content, malformed candidates, or unrecognized states) fail the stage closed.

## Canonical Commands And Tests

```bash
python3 -m py_compile experiments/exp403_408/*.py scratch/run_exp403_408_feature_ml_validation.py
python3 scratch/run_exp403_408_feature_ml_validation.py --stage all --dataset-limit 20 --representative-smoke --workers 2 --device cpu --reset-checkpoint --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation_smoke
python3 scratch/run_exp403_408_feature_ml_validation.py --stage classical-evaluate --workers 7 --device auto --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation
python3 scratch/run_exp403_408_feature_ml_validation.py --stage consensus-evaluate --workers 7 --device auto --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation
python3 scratch/run_exp403_408_feature_ml_validation.py --stage report --workers 7 --device auto --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation
```

The original full all-stage command was:

```bash
python3 scratch/run_exp403_408_feature_ml_validation.py --stage all --workers 7 --device auto --reset-checkpoint --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation
```

The final regression commands, queue check, independent verification, and
`git diff --check` were run after this metadata refresh.

## Final Evidence

| Evidence | Smoke (current code) | Full publication (current consensus/report) |
| --- | ---: | ---: |
| Dataset coverage | 20 | 1,117 |
| Audit/classical/SSL/synthetic/consensus checkpoints | 20 each | 1,117 each |
| Stage errors | 0 | 0 |
| Eligibility tiers (abstain/robust/classical/learned) | 5 / 5 / 5 / 5 | 162 / 316 / 138 / 501 |
| SSL and synthetic learned abstentions | 15 each | 616 each |
| Learned completed datasets | 5 each source | 501 each source |
| Source-status coverage | 13 per dataset | 13 per dataset |
| Valid all-source abstention consensus rows | 5 | 162 |
| Hard / Standard / Priority | 17 / 3 / 0 scoped | 2,218 / 806 / 31 frozen |
| Forbidden prediction fields | 0 | 0 |
| Independent reaggregation mismatches | 0 | 0 |
| Prediction SHA256 | `b5b55a9cb63fc2d7f557d613d5d2608a7b264cf0f715b59e6f4a31dbc5568ec5` | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` |
| Consensus SHA256 | `82e416a31953b3163c12806a78f958ef8791b4a88458dfc06a324afd1448cbae` | `588b246a7aae3a2cc00fd684731657c816f7a680d22e6899f35b43715b7760da` |

The final-code smoke run ID was `7d16ab1d3b4e459db47ca65925647cb4` and completed in
`70.247s` from run-bound timestamps. Its selector read only canonical
normal-TRAIN dataset names and counts; it did not query TEST rows, metadata,
blobs, or a TEST-derived order. Cross-stage dataset identity, `n_train`, and
eligibility matched for every selected dataset.

The full consensus regeneration run ID was `9e98049f3ee84980933e954d5e8b506b`.
It completed its 1,117 checkpointed records in `1104.025s`, with no errors.
Its `561` `source_train_count_mismatch` statuses are explicit historical
coverage abstentions, not available evidence and not integrity exceptions.
The remaining full source-status counts were 2,464 learned
`n_train_below_50` abstentions, 1,426 explicit `source_abstained` FeatureBlock
statuses, and 10,070 available source statuses.

The read-only sorted prediction verification was repeated twice and matched the
full manifest hash exactly both times. Frozen lanes were unchanged; Exp406-408
remain review-only and no prediction publication includes TEST labels or
evaluation metrics.

## MPS Decision

`task13_exp406_mps_benchmark.json` is the frozen four-run evidence artifact,
SHA256 `81d606ac6210f3b2edf03ec29a688b65623390d4e036336bc938e5ab8f417249`.
It covers `CricketX_normal_1` (`n_train=52`, `n_test=50`, seed `20260727`):

| Device runs | Mean runtime (s) | Prediction hash | Rank / max delta |
| --- | ---: | --- | --- |
| CPU 1, CPU 2 | 0.6661622920073569 | `200d69cf62e2a0282d1a7e4099ac79af3dce4d012f230748bad46e3abbcbe6d8` | exact / 0 |
| MPS 1, MPS 2 | 2.1810099170543253 | same | exact / `2.384185791015625e-07` |

The manifest validator requires exactly two finite positive CPU and two MPS
runtimes, one shared prediction hash, exact rank parity, finite delta no larger
than `1e-5`, recomputed means, and the measured-speed-consistent decision.
Parity passed, but MPS was slower; CPU remains selected.

## Implementation And Limits

This closure is committed after the final checks. The changes add the TRAIN-only
representative selection trace test, source-state fail-closed regressions,
cross-stage smoke identity coverage, benchmark semantic validation, and an
aggregate provenance republish before evaluation when code provenance changes.

The result is retrospective only. It does not claim prospective validation,
deployment validation, or an automatic lane-policy change.
