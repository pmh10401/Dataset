# Exp266 Invariance Report

All checks are retrospective boundary checks. TEST labels were read only after the
frozen prediction SHA-256 was verified, and no prediction artifact was rewritten.

Prediction SHA-256: `c0b99408c70a6ad0ad35af1cea91f83a969c1e4c33b0b42f96e22d7a3411e82c`

| Check | Status |
|---|---|
| Frozen prediction SHA matches manifest | PASS |
| Evaluation rows bind to the verified prediction SHA | PASS |
| Global summary binds to the verified prediction SHA | PASS |
| Frozen prediction contains no TEST labels | PASS |
| Frozen prediction contains no evaluation metrics | PASS |
| Autonomous Hard alert delta remains zero | PASS |
| Exact pinned Exp253 Hard parity covers every dataset | PASS |
| Repeat read preserves the frozen prediction hash | PASS |

Exact matching uses dataset TEST `instance_index`; adjacent indices are not merged.
Undefined precision, recall, and F1 denominators are emitted as `0.0`.

## Task 7 Executed Mutation Checks

All mutation inputs were generated under
`/Users/minho/Documents/Dataset/exp266_task7_invariance`. Frozen upstream files
were opened read-only and were not rewritten. Display names were mapped back to
their canonical dataset identifiers before output comparison, so the check tests
routing invariance rather than a deliberate identifier-byte change.

| Check | Command | Input SHA-256 | Output SHA-256 | Status |
|---|---|---|---|---|
| TEST label permutation | `python3 - <<'PY'` (within-dataset TEST-label rotation, then `prediction.predict_all`) | `4ad85f740467bd00b1af71bf45f8b5e69a6d65061cdaf46787cd458c2d9d2549` | `c0b99408c70a6ad0ad35af1cea91f83a969c1e4c33b0b42f96e22d7a3411e82c` | PASS |
| TEST anomaly-position mutation | `python3 - <<'PY'` (within-dataset TEST-index rotation, then `prediction.predict_all`) | `46e86956e3d43e20a6169d800887e94e8d0ae9d962851fd4cf183e1353dbcc36` | `c0b99408c70a6ad0ad35af1cea91f83a969c1e4c33b0b42f96e22d7a3411e82c` | PASS |
| Dataset display-name remapping | `python3 - <<'PY'` (remap all 1,117 loaded names, route, restore canonical IDs, then `prediction.freeze_rows`) | `581e59f0e798abb6b6266c008861263f33816d6b798e8381d7db4508a6ce263a` | `c0b99408c70a6ad0ad35af1cea91f83a969c1e4c33b0b42f96e22d7a3411e82c` | PASS |
| Input-row permutation | `python3 - <<'PY'` (reverse population and mapping iteration order, route, then `prediction.freeze_rows`) | `4b03ab6f5e4c7ebc5bc38507fed0f911fc939447da7f3eb8ffe90934c87e0148` | `c0b99408c70a6ad0ad35af1cea91f83a969c1e4c33b0b42f96e22d7a3411e82c` | PASS |
| Fresh full-run repeat hash | `python3 scratch/run_exp266_fn_recovery_shadow.py` (two fresh output roots) | `c0b99408c70a6ad0ad35af1cea91f83a969c1e4c33b0b42f96e22d7a3411e82c` | `c0b99408c70a6ad0ad35af1cea91f83a969c1e4c33b0b42f96e22d7a3411e82c` | PASS |

The label permutation changed 2,238 of 225,800 TEST label rows. The anomaly-position
mutation changed the TEST `instance_index` of all 225,800 TEST rows. Every substantive
fresh-run artifact from `00_contract_audit.json` through
`10_independent_reaggregation.json`, plus `MANIFEST.csv`, was byte-identical across
the two fresh runs before this Task 7 evidence section was appended. Runtime-only
`status.json` differed as expected because it contains timestamps and process IDs.

No TEST mutation was used to select or alter alpha, source groups, agreement counts,
the `0.01` TRAIN-normal risk target, or any routing policy.
