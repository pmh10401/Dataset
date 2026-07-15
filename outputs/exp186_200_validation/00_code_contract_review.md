# Exp186-200 Code Contract Review

## Scope

This review covers the Gemini first implementation and the Codex corrections
made before the full Exp186-200 run. The batch is a retrospective review and
shadow-diagnostic sensitivity study. It does not modify Exp137 autonomous Hard
alerts and it is not prospective or deployment validation.

## Corrected Before Full Execution

- Replaced the cross-dataset global workload cap with a dataset-local cap from
  that dataset's five cross-fitted TRAIN-normal fold candidate counts.
- Added SHA256 verification before opening Exp153-157 and canonical score NPZ
  files.
- Added exact 639-dataset eligibility and source-coverage checks.
- Added a hash-keyed shared TRAIN precompute cache so the same normal-only fold
  calculations are not repeated for all 15 stages.
- Moved completion markers after successful evaluation; Exp200 now also
  requires successful independent reaggregation before completion.
- Separated review and shadow-diagnostic metric scopes and marked every output
  retrospective, not prospectively validated, and not selected for operation.
- Added candidate indices to evaluation details and corrected recovery to mean
  recovery after all frozen Exp137 lanes, rather than after Hard alone.
- Added smoke-safe output and dataset-limit options plus dashboard-compatible
  per-dataset progress lines.
- Removed placeholder reasoning from Exp200 and wired hash-verified independent
  arithmetic into the stage.

## Frozen Rules

- The inherited Exp177 conservative alpha is loaded from the frozen policy and
  missing policy metadata is a hard failure.
- Exp189 and Exp199 use the raw dataset-local conformal count cap as primary.
  The minimum-one value is metadata only and is not selected using TEST results.
- Exp191-195 remain `shadow_hard_diagnostic` with `autonomous_alert=false`.
- Dataset/family names are lookup keys only. No TEST-length top-N cap, tail
  replacement, family performance routing, or Priority-review promotion exists
  in this implementation.

## Validation Evidence

- Python compilation succeeded for every file in `experiments/exp186_200`.
- Twenty package, structure, queue, and dashboard contract tests passed.
- A 10-dataset smoke run completed all 15 stages, including Exp200 independent
  reaggregation.
- Smoke prediction SHA256 values matched their manifests; prediction JSONL had
  no labels or evaluation metrics; every stage had a post-evaluation completion
  marker.
- The shared TRAIN precompute cache was reused from Exp187 onward.

## Remaining Limits

- These experiments reuse inherited feature/score configurations whose full
  end-to-end development provenance is unresolved.
- TEST labels are used only after prediction freeze, but all reported metrics
  remain retrospective on a repeatedly studied benchmark.
- Five TRAIN folds provide coarse resolution for dataset-local count guards.
- No candidate or shadow rule is promoted to autonomous operation by this
  batch. Any promotion requires a separately preregistered prospective shadow
  validation.
