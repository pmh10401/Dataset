# Task 8: Exp270-C TRAIN Risk and Exp271 Qualification

## Scope

Implemented the TRAIN-only source-risk and source-qualification boundary for
the Exp267-273 recovery package. This task freezes membership before any TEST
label access; it does not create new Hard alerts or modify Exp253/Exp266 lanes.

## Files

- `experiments/exp267_273_zero_support_fn_recovery/train_risk.py`
- `experiments/exp267_273_zero_support_fn_recovery/qualification.py`
- `experiments/exp267_273_zero_support_fn_recovery/tests/test_qualification.py`
- `experiments/exp267_273_zero_support_fn_recovery/config.py`
- `docs/superpowers/plans/2026-07-21-exp267-273-zero-support-fn-recovery.md`
- `docs/superpowers/specs/2026-07-21-exp267-273-zero-support-fn-recovery-design.md`

## Frozen Contract

- Absolute aligned TRAIN OOF Spearman collapse threshold: `0.90`.
- Minimum finite aligned TRAIN OOF rows: `20`.
- Variants in one registered family or evidence identity collapse without a
  correlation calculation.
- Distinct families collapse through connected components when absolute
  Spearman is at least `0.90`.
- Low overlap or non-finite correlation leaves independence unverified, so the
  affected source cannot contribute an independent membership or vote.
- TRAIN pseudo-run p-values exclude the queried OOF score and use
  `(1 + count(reference >= score)) / (1 + len(reference))`.
- The one-sided 95% Wilson bound is recorded for every alpha, source, variant,
  dataset, and TRAIN tier; `0.005` remains the primary-alpha qualification
  view.

## Boundary Checks

- `audit_train_risk()` rejects TEST score, label, or evaluation fields.
- `qualify_sources()` has no TEST-label argument and writes the hash-protected
  `exp272_source_membership.json` before evaluation.
- The manifest carries config, input, prediction, and run-contract hash
  material, plus TRAIN-only risk and correlation provenance.
- `evaluate_frozen_qualification()` verifies prediction and membership hashes
  before reading any TEST truth. Autonomous Hard and human-assisted review
  summaries are returned as separate objects and never expose a mixed metric.

## Verification

```text
python3 -m py_compile experiments/exp267_273_zero_support_fn_recovery/train_risk.py experiments/exp267_273_zero_support_fn_recovery/qualification.py experiments/exp267_273_zero_support_fn_recovery/tests/test_qualification.py
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_qualification -v
python3 -m unittest discover -s experiments/exp267_273_zero_support_fn_recovery/tests -q
git diff --check
```

All focused tests (`11`) and the package suite passed. The test coverage proves
label-permutation invariance of membership, family double-vote blocking,
high-correlation collapse, low-overlap abstention from independence, Wilson and
query-exclusion arithmetic, tamper rejection, coverage rejection, and hard /
review metric separation.

## Limits

This is an implementation and qualification boundary only. It has not run the
new score sources across all 1,117 datasets, does not select any source from
TEST performance, and remains retrospective rather than prospective validation.
