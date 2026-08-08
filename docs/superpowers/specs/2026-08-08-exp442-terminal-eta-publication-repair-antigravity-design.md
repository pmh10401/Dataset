# Exp442 terminal ETA publication repair design

## Purpose

Exp442 is an engineering-only successor to Exp441. Exp441 completed the full
1,117-record label-free replay with zero record errors and the frozen 30-candidate
selection, but publication failed because the runtime wrote terminal
`eta_seconds=0.0` while the publication validator required `eta_seconds=null`.

Exp442 changes no scientific input, feature, score, threshold, ranking, route,
candidate budget, bootstrap rule, or governance decision. It repairs only the
terminal progress contract and gives the repaired lifecycle a new immutable
identity and an absent output root.

## Preserved Exp441 failure evidence

The following predecessor root is immutable and may be read only as an audit or
parity oracle:

```text
/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808
```

Its completed attempt has 1,117 accepted records, 0 errors, 840 provisional
candidates, 394 outside-Hard candidates, 73 eligible candidates, 49 dataset
winners, and 30 selected candidates. No publication or evaluation exists, its
PID is inactive, and TEST-label access is zero. Exp442 must not edit, resume,
copy, or publish that attempt.

## Frozen Exp442 identity

```text
experiment_id = Exp442
schema_namespace = exp442
run_identity = exp442-terminal-eta-publication-repair-antigravity-20260808
run_root = /Users/mac/Documents/Dataset/exp442_terminal_eta_publication_repair_antigravity_20260808
dataset_count = 1117
cpu_workers = 11
backend = cpu
```

The new run root must be absent when readiness begins. Readiness is bound to a
clean reviewed commit and must be recreated after every code change.

## Exact engineering change

`ProgressWriterAntigravity` keeps its existing ETA formula while a process is
active. When `terminal_antigravity` writes an inactive `complete`, `failed`, or
`interrupted` state, it writes exact JSON null (`None` in Python) for
`eta_seconds`. The publication validator remains strict and accepts only that
terminal representation.

This normalization is tested at the writer boundary and through an actual
writer-to-publication integration. Hand-written terminal progress fixtures must
match the real writer.

## Thin shared lifecycle

Exp441 remains callable with its original fixed identity. A frozen
`RecoveryLifecycleProfileAntigravity` explicitly carries only:

- experiment id;
- schema namespace;
- run identity;
- fixed run root;
- reviewed design SHA256;
- CLI program name.

Readiness, runtime, reporting, and evaluation receive that profile explicitly.
Their existing Exp441 public functions are fixed wrappers around
`EXP441_PROFILE`. Exp442 has a small four-command CLI wrapper fixed to
`EXP442_PROFILE`. Module-global mutation, environment selection, path flags, and
runtime identity overrides are forbidden.

All scientific fields in the Exp442 profile are inherited from and checked
equal to Exp441: frozen source paths and hashes, selected coordinate identity
`db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048`,
budget 30, one candidate per dataset, bootstrap seed 44120260808, NumPy 2.4.6,
and the nine-field review-only governance block.

## Execution and label boundary

Exp442 performs a fresh label-free readiness replay and a fresh 1,117-record
CPU/11-worker `new` run. It may compare the resulting label-free selection and
source identities with Exp441, but may not reuse predecessor attempt bytes.

Prediction publication must be create-only and replay-verified before the
evaluation module may open SQLite TEST labels. The prediction phase may read
only the existing Exp440 frozen checkpoint/publication and Exp404 Hard
prediction/manifest. Exp440 evaluation artifacts, TEST labels, actual anomaly
locations, and family-specific TEST performance remain forbidden.

## Acceleration decision

The selector performs deterministic parsing, hashing, set filtering, and
sorting. It has no dense tensor stage. MLX/MPS is therefore inapplicable and the
canonical path remains CPU with `max_cpu_count - 1 = 11` workers on this Mac.

## Required verification

Before readiness:

1. RED proves the real writer emits a publication-incompatible terminal ETA.
2. GREEN proves terminal ETA is exact null and active ETA behavior is unchanged.
3. Exp441 public wrappers retain original profile identity.
4. Exp442 CLI exposes exactly `readiness/new/inspect/evaluate` and no flags.
5. Exp442 profile cannot be constructed with altered scientific fields.
6. Exp441 and Exp442 selectors produce identical 30-coordinate identity.
7. Full Exp441 and Exp442 focused suites pass.
8. A real 1,117-record CPU/11-worker replay matches the sequential reference.
9. The code map is synchronized and a fresh Sol High review reports no findings.

## Interpretation

Even if all technical and retrospective gates pass, Exp442 remains adaptive,
descriptive, review-only evidence. It is not independent validation, an
operational alert policy, an external-generalization claim, or a new champion.
