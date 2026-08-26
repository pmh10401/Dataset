# Task 5 Report: Exp471 Typed-Abstain Paper Reporting

## Scope

Implemented the typed report renderer and integrated it with the existing
create-only Exp471 atomic publisher. No official result, source data, labels,
secrets, or official process was read or modified.

## Result

- The renderer consumes only the fully validated Exp471 v2 evaluation artifact;
  it does not reconstruct an Exp450 report object.
- It publishes canonical coverage, full-population decision, scored-support,
  and all-ten common-support metrics in CSV/JSON; both ROC/PR figure sets in
  deterministic SVG/PNG; HTML; and a typed digest manifest below
  `exp471_typed_report_artifacts_antigravity/`.
- All ten method blocks are preserved. Coverage includes population, scored,
  abstained, failed, ratio, and reason counts. Continuous tables retain explicit
  AP/AUROC denominators; full-population decisions remain defined for no-alert
  abstentions.
- SVG metadata and labels bind ROC to `false_positive_rate` / `true_positive_rate`
  and PR to `recall` / `precision`. The common-support authority is carried from
  the v2 artifact (production requires 922 datasets).
- HTML states that abstention is a preregistered no-alert outcome, not a runtime
  failure and not a continuous score. It prohibits score completion, filling
  absent values, automatic promotion, and historical-parity claims.
- The outer publisher accepts only `Exp471TypedReportArtifactAntigravity`,
  recomputes and verifies every leaf digest, then writes provenance, artifact
  map, marker, fsyncs staging, and retains atomic no-replace rename/retry.

## TDD Evidence

The typed renderer test was added before production code. Its initial RED run
failed at import because `typed_abstain_reporting_antigravity` did not exist.
After implementation, the same test was GREEN. Lifecycle tests then verified
exact reuse, conflict rejection, semantic rehash rejection for every typed leaf,
and threaded/process publication contention.

## Verification Evidence

```text
Task 5 reporting lifecycle suite: 28 tests in 22.267s — OK
Reporting/CLI suite: 86 tests in 49.091s — OK
Task 4 evaluator regression: 28 tests in 4.571s — OK
```

The first full Exp471 run identified a stale fixed codemap expectation for the
new production renderer. Under the approved minimal scope extension, the
codemap test now adds only the renderer path and its direct effect/import
expectations so the lock fingerprint scope can include that file. No existing
expectation was otherwise changed.

## Fix Round 1

RED tests found the report closure did not bind the Exp450 PNG helper source and
the direct renderer could follow a caller-root symlink. The closure now binds
the Exp450 reporting/evaluation modules actually used to derive raster and
metric/curve values, and test-injected PNG-source bytes change its digest. The
renderer now opens the caller root and destination through descriptor-relative,
no-follow cache-artifact helpers, validates both before and after publication,
and rejects missing, extra, or linked destination inventory. PNG output now
contains a deterministic `tEXt` metadata chunk binding support, axes, methods,
and canonical curve values without changing visible raster semantics.

Focused RED then GREEN evidence: closure dependency and symlink/inventory tests
failed before the hardening change; after it, the focused renderer set passed
3 tests in 1.787s and the synchronized codemap suite passed 5 tests in 19.894s.

Final verification on the repaired tree: the production-shaped 1,117-dataset /
922-common-support oracle, dependency-closure check, and descriptor-safety
check passed 3/3 in 6.177s.  The full reporting/CLI suite passed 89/89 in
52.564s; the Task 4 evaluator regression passed 28/28 in 4.122s; and the
complete Exp471 suite passed 219/219 in 87.820s. `py_compile`, codemap
generation plus `--check`, and `git diff --check` completed successfully.
