# Task 12 Fix Round 1 Report: Independent Reaggregation, Reports, and Manifest

## Status

Implemented the first remediation round for Exp403-408 retrospective reporting.
The report path now separates frozen input verification, label-based evaluation
publication, independent reaggregation, and final human/machine manifests.

## Delivered

- Independent reaggregation verifies input/publication SHA256 contracts before
  consuming prediction, audit, consensus, evaluation, or frozen-label files.
- Exp408 TP, FP, precision, FN recovery, baseline Hard TP/FP/FN, and Zero-F1
  diagnostics are recomputed from frozen label rows and candidate indices.
- The report stage writes a frozen label table plus candidate-level evaluation
  rows, and does not read the label DB until the externally pinned prediction,
  provenance, audit, evaluation, consensus, DB, and lane hashes verify.
- The Korean runbook now contains runnable `all` and `report` commands and
  states the retrospective, immutable-Hard, and human-assisted boundaries.
- `MANIFEST.json` and `MANIFEST.md` exclude mutable `status.json` and
  self-referential manifest outputs from their artifact inventory.
- Resumed report execution validates final report artifacts and invalidates the
  report checkpoint when the required report or manifest is missing or tampered.

## Verification

TDD began with the new altered-baseline/FN and frozen-publication failure cases;
the pre-fix behavior accepted the official totals or lacked the required
publication inputs. The focused and regression checks now pass:

```text
python3 -m unittest \
  experiments.exp403_408.tests.test_evaluation.EvaluationTests.test_independent_reaggregation_matches_frozen_official_metrics \
  experiments.exp403_408.tests.test_evaluation.EvaluationTests.test_independent_reaggregation_rejects_altered_baseline_fn_and_consensus_hash \
  experiments.exp403_408.tests.test_evaluation.EvaluationTests.test_independent_reaggregation_rejects_rehashed_exp408_tp_integer \
  experiments.exp403_408.tests.test_pipeline.PipelineTests.test_report_resume_regenerates_when_final_output_is_missing_or_corrupt \
  experiments.exp403_408.tests.test_pipeline.PipelineTests.test_final_manifest_ignores_status_but_rejects_deleted_or_corrupt_report \
  experiments.exp403_408.tests.test_pipeline.PipelineTests.test_runbook_commands_reference_the_real_entrypoint -v
Ran 6 tests - OK

python3 -m py_compile experiments/exp403_408/*.py scratch/run_exp403_408_feature_ml_validation.py
python3 -m unittest discover -s experiments/exp403_408/tests -p 'test_*.py' -v
Ran 130 tests - OK

git diff --check
(no output; exit 0)
```

The adversarial coverage includes a rehashed altered Exp408 TP integer, altered
baseline FN, altered consensus bytes without a matching frozen hash, hash
failure before label DB access, missing/corrupt final reports after a report
checkpoint, status updates after manifest publication, and runbook entrypoint
validation.

## Scope Boundary

This remains retrospective-only. Exp405 does not alter production automatic
Hard, and Exp406-408 remain human-assisted review-only Shadow outputs. No
prospective or deployment validation is claimed.

## Remaining Concerns

- The tests use bounded frozen fixtures; this change does not publish a new
  full-dataset retrospective run.
- The bootstrap is independently reconstructed with the frozen seed and 2,000
  paired dataset resamples. Changing that protocol requires a new frozen
  publication rather than a report-only rerun.
