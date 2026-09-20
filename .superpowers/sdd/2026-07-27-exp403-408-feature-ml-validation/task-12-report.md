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

---

# Task 12 Fix Round 2: Exp408 Diagnostics and Immutable Manifest Boundary

## Delivered

- The independent Exp408 reconstruction now emits per-variant, per-alpha
  dataset-paired bootstrap 95% intervals for TP, FP, precision, FN recovery,
  and the Zero-F1 before-minus-after difference. The Korean report renders
  those diagnostics and the independently reconstructed Standard overlap,
  Priority overlap, and outside-existing review workload for every variant.
- The evaluation input manifest now records the verified database snapshot as
  ordered DB/WAL/SHM artifact hashes plus a canonical composite SHA256. The
  report manifest exposes that snapshot alongside current external DB, lane,
  provenance, input, prediction, evaluation, consensus, and report hashes.
- `MANIFEST.json` code hashes now cover the full execution path: contracts,
  foundation, feature registry/stability, classical committee, prediction,
  evaluation and label loading, learned/synthetic sources, consensus,
  pipeline, independent auditor, reporting, wrapper, queue runner, and
  dashboard integration.
- Artifact inventory changed from an output-directory scan to a fixed immutable
  allowlist. Logs, stdout, status, checkpoints, temporary files, and manifest
  self-references cannot enter the inventory.
- The Korean runbook now states the true order: verify frozen prediction and
  provenance, then generate/freeze evaluation CSVs, then independently audit
  them before publishing reports.

## Focused Verification

```text
python3 -m unittest \
  experiments.exp403_408.tests.test_evaluation.EvaluationTests.test_exp408_reaggregation_exposes_overlap_and_paired_bootstrap_diagnostics \
  experiments.exp403_408.tests.test_pipeline.PipelineTests.test_report_separates_autonomous_and_human_assisted_metrics \
  experiments.exp403_408.tests.test_pipeline.PipelineTests.test_manifest_allowlist_ignores_changing_stdout_and_renders_execution_context \
  experiments.exp403_408.tests.test_pipeline.PipelineTests.test_runbook_commands_reference_the_real_entrypoint -v
Ran 4 tests - OK

python3 -m unittest \
  experiments.exp403_408.tests.test_pipeline.PipelineTests.test_database_snapshot_composite_binds_db_wal_and_shm -v
Ran 1 test - OK

python3 scratch/test_exp403_408_queue.py
Exp403-408 queue and dashboard registration: PASS
```

## Scope Boundary

This remains retrospective-only. No deployment or prospective validation is
claimed, Exp405 does not change automatic Hard, and Exp406-408 remain
human-assisted review-only Shadow outputs.

---

# Task 12 Fix Round 3: Transitive Local Code Provenance

## Delivered

- Replaced the hand-maintained execution-code list with deterministic static
  AST resolution of repository-local `import` and `from ... import ...` edges.
  All Exp403-408 module roots plus the wrapper, queue runner, and dashboard
  entrypoints are traversed recursively.
- Only repository-local `.py` files are included. Standard-library and
  site-package modules remain excluded and package versions remain separately
  recorded in the runtime manifest.
- `MANIFEST.json` now records method
  `ast_recursive_repository_local_import_closure/v1`, explicit roots, every
  reachable local file SHA256, and a canonical closure SHA256. `MANIFEST.md`
  renders the same closure method, roots, composite, and per-file hashes.
- The closure includes the Exp226 adapters/config/contracts/database/
  preprocessing chain reached from Exp403-408 foundation code, along with any
  further statically imported repository-local module.

## TDD and Verification

```text
RED
python3 -m unittest \
  experiments.exp403_408.tests.test_pipeline.PipelineTests.test_transitive_code_provenance_covers_exp226_and_changes_with_preprocessing -v
ImportError: cannot import name 'execution_code_provenance'

GREEN
Ran 1 test - OK
```

The regression temporarily changes the real local Exp226 `preprocessing.py`,
restores it in `finally`, and proves both its inclusion and closure-hash
change. Full regression evidence is recorded in the task response.
