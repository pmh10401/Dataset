# Exp443 Task 10 — post-freeze evaluation report

## Outcome

Task 10 implements the separate retrospective evaluator for the three frozen
Exp443 policies. Prediction code remains label-free and does not import this
module. No official TEST labels were opened and no official evaluation was
created during implementation.

## Label boundary

`load_frozen_test_labels_antigravity()` requires the supplied publication to
have zero label access, replays it twice before the label-capable connection is
imported, and then opens exactly one descriptor-bound read-only database
snapshot. It runs one fixed ordered TEST-label query per frozen dataset,
requires contiguous indices and binary labels, closes the connection, and
rehashes the source and publication again.

The label snapshot is detached, immutable, and bound to the publication
manifest SHA256, all three prediction SHA256 values, the frozen database
SHA256, a canonical label-identity SHA256, 1,117 query calls, and zero
pre-freeze label access.

## Metrics and comparison

The evaluator computes, independently for A, B, and C:

- whole-population and per-dataset TP/FP/FN/TN;
- precision, recall, F1, and false-positive rate with zero denominators set to
  `0.0`;
- simple macro means across all 1,117 datasets;
- added and removed coordinates with TP/FP attribution;
- rescued, regressed, and baseline-F1-at-least-0.8 strong regressions;
- explicit 19 affected-dataset and 1,098 unaffected-dataset diagnostics;
- family diagnostics and exact unaffected-coordinate invariance.

The fixed Exp404/Exp291 baseline must reproduce
`1915 / 303 / 2003 / 191679` and F1 `0.6241851368970013`. Policy A must have
the same coordinates and confusion. Any mismatch stops evaluation publication.

## Paired bootstrap and decision

The bootstrap rejects every non-frozen production setting. It requires NumPy
2.3.5, PCG64 seed 44320260809, 20,000 paired dataset samples, int64 indices,
`endpoint=False`, and linear 2.5/97.5 percentiles. A sample-index SHA256 makes
the resampling stream independently replayable.

Each policy receives the complete preregistered gate table. Passing arms are
reported as `retrospective_improvement_signal`; all other arms are
`no_retrospective_improvement_signal`. The decision artifact always keeps
`scientific_winner_selected=false`, `review_only=true`, and
`operational_alert_eligible=false`.

## Immutable artifacts and replay

The evaluator creates exactly ten files through a private staging directory
and one no-clobber rename. A pre-existing evaluation is inspection-only and
does not reopen labels or overwrite any file. A mid-label failure is marked
truthfully as label access started and preserved in a unique failed audit
sibling. Its self-excluding artifact map binds the other nine files.

Replay reconstructs all dataset rows, transitions, aggregates, family facts,
gates, decisions, and all three 20,000-sample bootstraps from frozen evaluation
bytes. It then requires byte identity for the entire ten-file inventory and
reapplies the official baseline, Policy A, database, population, and 1,098
unaffected-authority checks.

## Verification

- Focused Task 10 suite: 15 tests passed.
- Full Exp443 suite: 156 tests passed; two opt-in real-data tests skipped as
  expected.
- Python compileall: passed.
- `git diff --check`: passed.
- Synthetic full-population replay: three policies x 1,117 dataset rows,
  deterministic bootstrap reconstruction, exact inventory byte replay.
- Create-only replay: second evaluation call made zero label-loader and zero
  compute calls and preserved every artifact byte.

## Next boundary

Task 11 may expose this evaluator only through the lazy `evaluate` CLI branch.
Official TEST labels remain closed until the official 57-unit label-free run
finishes, all three predictions are frozen and inspected, and the user signals
completion. Exp444 must not be started from Task 10.
