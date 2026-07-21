# Exp266 Final Review Fix Report

## Status

Complete. All six Exp266 final-review findings were fixed in one wave.

- Full population: `1,117/1,117`
- Pipeline terminal state: `completed`
- `first_incomplete_stage`: `None`
- `completed_output_is_valid`: `True`
- Strict TRAIN-risk audit: `PASS`
- Independent reaggregation: `PASS`
- Invariance checks: `6/6 PASS`
- Supabase/Vercel retrospective replay files changed: none
- Source/test commit: `f11db128` (`fix(exp266): enforce final review contracts`)

## 1. Strict TRAIN-Risk Audit Contract

Added `02_train_normal_risk_audit_manifest.json` and
`validate_train_risk_audit()`. The validator is used by risk-stage resume,
report-stage completion, complete-output validation, and report generation.

It requires:

- exactly 1,117 unique rows and the exact pinned dataset population;
- exact `n_train` values from the four pinned certified cache manifests;
- certified source count `4` and exact source set
  `kernel|statistical|structural|change_point`;
- provenance and query-excluded source fitting true on every row;
- policy validation true exactly for eligible rows;
- `n_train < 5` abstention with `normal_runs=0`, the exact abstention reason,
  no passing risk flags, and empty sensitivity metrics;
- eligible rows with `normal_runs=n_train`, internally consistent false-review
  counts, rates, Wilson bounds, and pass flags;
- five-source sensitivity provenance false, policy validation false,
  deployment eligibility false, risk pass null, and one exact blocker;
- the frozen aggregate counts and the CSV SHA-256;
- a sidecar manifest binding the dataset-set hash, CSV hash, aggregate contract,
  certified source set, database hash, and sensitivity blocker.

Name-complete or stale CSV files no longer satisfy resume. A TDD test changes
`source_count` to `5`, rehashes the stale CSV in its sidecar, and confirms the
strict row contract still rejects it.

Final certified aggregate:

```text
datasets=1117
eligible=955
abstained=162
normal_runs=146796
RecoveryShadow false reviews=84, U95=0.000684504543
RecoveryPriority false reviews=160, U95=0.001241096234
combined false reviews=244, U95=0.001846477191
target=0.01, PASS
```

Unapproved five-source sensitivity:

```text
RecoveryShadow false reviews=110
RecoveryPriority false reviews=176
combined false reviews=286
provenance_verified=false
policy_risk_validated=false
risk_target_pass=null
deployment_eligible=false
```

Exact blocker:

```text
Five-source combined TRAIN-risk approval is blocked until Exp257 has a producer-signed/pinned manifest tying the frozen archives to query-excluded fitting and exact seeds, or a separately preregistered regeneration is performed.
```

## 2. Immutable Lane Parity

Evaluation checks every dataset before opening SQLite. Independent
reaggregation separately repeats the same source-file parsing and exact index
comparisons.

```text
Hard     source=Exp253  parity=1117/1117  mismatches=0
Shadow   source=Exp264  parity=1117/1117  mismatches=0
Priority source=Exp264  parity=1117/1117  mismatches=0
Standard source=Exp264  parity=1117/1117  mismatches=0
```

Index-map hashes:

```text
Hard     3af05e308116ef3634a855baf002561e9b3edbea8bcb47d977a4972315c58f2e
Shadow   47b7448dd95527a670ebeb2c8ff63289a986c83a24a842165dfc3eb6b8b16bd3
Priority 30f2d5f69d2ff24a075a8503dda0b4f884c6789a1af196a791e541317b3125a7
Standard a4778fd0f704a67fba4ac462b973f705f8729ad3fe78c4ad91b45f8a296b1a44
```

Tests swap a Shadow index while preserving its count and confirm evaluation
fails before label access and independent reaggregation fails on its own.

## 3. Review-Rate Scope

The ambiguous `review_request_rate` field was removed from dataset rows,
global summary, and independent reaggregation.

```text
total_existing_plus_exp266_review_request_rate=0.004747320061255743
exp266_incremental_review_request_rate=0.00046452271567126084
```

The result report explicitly limits TRAIN-normal risk comparisons to Exp266's
incremental `RecoveryShadow` and `RecoveryPriority` lanes. Existing Shadow,
Priority, and Standard workload is reported separately and is not compared to
the Exp266 TRAIN-risk bound.

## 4. Regenerable Invariance Evidence

Stable logic now lives in
`experiments.exp266_fn_recovery_shadow.invariance`. The scratch verifier is a
thin CLI wrapper over the same package logic used by the pipeline report stage.
The report stage executes and rewrites detailed evidence instead of replacing
it with a generic table.

Final checks:

```text
label-permutation          PASS
anomaly-position-mutation PASS
display-name-remapping     PASS
test-length-mutation       PASS
input-row-permutation      PASS
repeat-prediction-hash     PASS
```

All six outputs match prediction SHA-256:

```text
2f85a4a2c80570be6471b19f27878e392a5e55a6a116ed00372b5c1bb95ad246
```

TEST-length mutation adds one unselected TEST slot to every dataset
(`1,117` added slots) and proves the TRAIN-derived config and prediction hash
remain unchanged. Every command in `08_invariance_report.md` is complete,
shell-quoted, and runnable, including paths containing spaces and Unicode.

## 5. Frozen Database Hash

Frozen SQLite SHA-256:

```text
8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
```

The hash is frozen in config, verified while building the contract audit,
verified before evaluation opens SQLite or reads TEST labels, and verified
again by independent reaggregation. It is published in the contract audit,
prediction manifest, risk manifest, global summary, independent reaggregation,
and every main manifest row.

## 6. SQLite Resource Handling

Exp266 SQLite connections now use explicit `close()` in `finally` blocks while
retaining URI `mode=ro` and `PRAGMA query_only = ON`. The complete suite passes
with `ResourceWarning` promoted to an error.

Baseline before the fix:

```text
70 passed, 21 subtests passed, 2 unclosed-SQLite warnings
```

Final result:

```text
79 passed, 21 subtests passed, 0 ResourceWarnings
```

## TDD Evidence

RED evidence included:

- missing `validate_train_risk_audit` import;
- missing package-level `invariance` module;
- same-count swapped Exp264 index accepted by the prior Hard-only parity gate;
- ambiguous review-rate assertions against the old field;
- database hash mismatch not checked before connection;
- missing TEST-length mutation check;
- JSON-escaped Unicode verifier command rejected by a shell-roundtrip test.

GREEN focused commands:

```bash
python3 -m pytest -q experiments/exp266_fn_recovery_shadow/tests/test_train_risk.py
python3 -m pytest -q experiments/exp266_fn_recovery_shadow/tests/test_evaluation.py experiments/exp266_fn_recovery_shadow/tests/test_invariance.py experiments/exp266_fn_recovery_shadow/tests/test_pipeline.py experiments/exp266_fn_recovery_shadow/tests/test_prediction_boundary.py experiments/exp266_fn_recovery_shadow/tests/test_inputs.py
```

Results: `14 passed`; then `56 passed, 19 subtests passed`.

## Final Verification

```bash
python3 -W error::ResourceWarning -m pytest -q experiments/exp266_fn_recovery_shadow/tests
```

Result: `79 passed, 21 subtests passed`.

```bash
python3 -W error::ResourceWarning -m pytest -q scratch/test_exp266_fn_recovery_shadow_queue.py
```

Result: `6 passed`.

```bash
python3 -m pytest -q scratch/test_rank_dashboard_operational.py -k 'exp266 or dashboard_registers_exp266'
```

Result: `4 passed, 63 deselected`.

```bash
python3 scratch/verify_exp266_invariance.py --check all
```

Result: exit `0`, six JSON records, all `passed=true`.

```bash
python3 -m experiments.exp266_fn_recovery_shadow.independent_reaggregate --output-root /Users/minho/Documents/Dataset/exp266_outside_lane_fn_recovery_shadow
```

Result: exit `0`, all four lane parities exact, independent reaggregation
`PASS`.

```bash
python3 -m py_compile experiments/exp266_fn_recovery_shadow/__init__.py experiments/exp266_fn_recovery_shadow/config.py experiments/exp266_fn_recovery_shadow/contracts.py experiments/exp266_fn_recovery_shadow/database.py experiments/exp266_fn_recovery_shadow/evaluation.py experiments/exp266_fn_recovery_shadow/independent_reaggregate.py experiments/exp266_fn_recovery_shadow/inputs.py experiments/exp266_fn_recovery_shadow/invariance.py experiments/exp266_fn_recovery_shadow/prediction.py experiments/exp266_fn_recovery_shadow/reporting.py experiments/exp266_fn_recovery_shadow/routing.py experiments/exp266_fn_recovery_shadow/run_pipeline.py experiments/exp266_fn_recovery_shadow/train_risk.py scratch/run_exp266_fn_recovery_shadow.py scratch/verify_exp266_invariance.py
git diff --check
git diff --cached --check
```

Result: all exit `0`.

The broad historical dashboard script still stops at its pre-existing Exp149
fixture assertion because four ignored Exp149/150 CSV files are absent from
this isolated worktree. Exp266-focused dashboard checks pass, and no dashboard,
Vercel, Supabase, or retrospective replay source was changed in this fix.

## Final Artifact Hashes

```text
00_contract_audit.json                    bbc9cd7c3ec466464f3eb8e194117f5f687f2a36493e2b5c139823412681fd49
01_source_manifest.csv                    0e5c77f8f245f74d64eb54432a8fe098f986837d30fd927a4c481803ebc82694
02_train_normal_risk_audit.csv            ae96262ce0e6d9f7c2f742c37d259f270487a64e95afe68dc112895547af9ddd
02_train_normal_risk_audit_manifest.json  53dc28c53401222383baa933a11a5056f2e569de27b007c2859ea972e1d7691a
03_frozen_predictions.jsonl               2f85a4a2c80570be6471b19f27878e392a5e55a6a116ed00372b5c1bb95ad246
04_prediction_manifest.json               6cdc2ac153409e4f9dec0dc78c7abe138a36fb1a7a2dded265305118eaaebf4e
05_dataset_results.csv                    695690f59d8ee07865a1138d5ca35d06812a77c5dd99c8a88a6f25b762a7d58c
06_global_summary.csv                     1758fd9425fc8aa2a1105bb1f343868d3a5d0d2241ed33806294e09191afd4c8
07_source_agreement_diagnostics.csv       f83806298cea96c547c759111dec83c29c834b4f2d7734017bfd01d76b021ccd
08_invariance_report.md                   896a88feb5840f37a3e4e381601af79b7f49b0629f92fed039342e03ea2d4b70
09_result_report.md                       3cd2976356e4dea0ab1f28b2b4010f7870058322718bc787c44ee877674cebb3
10_independent_reaggregation.json         1d53cc9c519f72c1014431674cb1e8f568d39da186f9497aacfc9efb91d20d2c
MANIFEST.csv                              4b3f9e3483efd32740928528079ce85aa28bdd05ed454ac8f31a7effc566d7ac
```

Repository summaries include `00`, the new risk manifest, `04`, `06`, `08`,
`09`, `10`, and `MANIFEST.csv`. Raw JSONL, row-level CSVs, DBs, caches, logs,
and checkpoints remain outside Git.
