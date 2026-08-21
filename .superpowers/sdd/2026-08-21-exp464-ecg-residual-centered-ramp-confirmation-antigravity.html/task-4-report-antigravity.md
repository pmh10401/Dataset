# Exp464 Task 4 report antigravity

## Commit boundary

- Base product HEAD for this fix round: `2de05e639a0fc31c932ae14b9cf190d2fb5e0825`
- Functional parent already on branch before this fix round: `a8f3fa905fd07342cdc1c34355e9a90a24686c06`
- Functional commit: recorded in the final handoff after commit creation
- Map-only child: recorded in the final handoff after commit creation

## Code-map preflight

- versioned triplet checked before edits:
  - `docs/codemap/codemap_exp464_antigravity.html`
  - `docs/codemap/codemap_exp464_antigravity.json`
  - `docs/codemap/codemap_exp464_antigravity.lock`
- callers confirmed before edits:
  - Task 3 fold scores
  - new Task 4 focused tests
  - future Task 5 CLI / private qualification
- effects confirmed before edits:
  - independently recomputed qualification metrics
  - one fixed C6 decision
  - exactly seven canonical result leaves
  - semantic replay and aggregate-only dashboard export
  - Task 5 private qualification handoff after fresh review
- coverage confirmed for the updated boundary:
  - `test_qualification_antigravity.py`
  - `test_evaluation_antigravity.py`
  - `test_artifacts_antigravity.py`
  - `test_replay_antigravity.py`
  - `test_dashboard_export_antigravity.py`
  - Task 2–3 regressions and governance

## Scope completed

- created:
  - `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/qualification_antigravity.py`
  - `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/evaluation_antigravity.py`
  - `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/artifacts_antigravity.py`
  - `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/replay_antigravity.py`
  - `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/dashboard_export_antigravity.py`
  - five matching focused tests
  - this report
- updated truthful Task 4 review-pending state in:
  - `docs/EXP464_DASHBOARD_antigravity.html`
  - `docs/EXP464_PREREGISTRATION_antigravity.html`
  - `docs/EXP464_RUNBOOK_antigravity.html`
  - `docs/EXP464_IMPLEMENTATION_REPORT_antigravity.html`
  - `docs/superpowers/plans/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html`
  - `RESEARCH_STATE.json`
  - `EXPERIMENT_CATALOG.json`
  - `CHAMPION_LEDGER.json`
  - `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_governance_antigravity.py`
  - `scratch/build_codemap_exp464_antigravity.py`
  - generated Exp464 map triplet
- did not create:
  - Task 5 CLI or launcher
  - private TRAIN roots
  - official roots or official result hashes
  - TEST-label access

## RED evidence

Focused Task 4 suite before implementation:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_qualification_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_evaluation_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_artifacts_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_replay_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_dashboard_export_antigravity
```

Observed RED:

- `Ran 10 tests`
- `FAILED (errors=10)`
- every error was the expected missing Task 4 import surface

## GREEN evidence

Focused Task 4 suite, pass 1:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_qualification_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_evaluation_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_artifacts_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_replay_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_dashboard_export_antigravity
```

- `Ran 13 tests`
- `OK`

Focused Task 4 suite, pass 2:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_qualification_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_evaluation_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_artifacts_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_replay_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_dashboard_export_antigravity
```

- `Ran 13 tests`
- `OK`

Task 2–3 Exp464 regression once:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_config_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_contracts_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_predecessor_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_sources_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_centered_ramp_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_scoring_antigravity
```

- `Ran 23 tests`
- `OK`

Focused Task 4 suite after the frozen predecessor-authority fix:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_qualification_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_evaluation_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_artifacts_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_replay_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_dashboard_export_antigravity
```

- `Ran 13 tests`
- `OK`

Exp463 predecessor authority smoke plus Exp464 governance:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest \
  experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_predecessor_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_governance_antigravity
```

- `Ran 3 tests`
- `OK`

## Result integrity

- exact publication inventory:
  - `context_antigravity.json`
  - `method_fold_metrics_antigravity.jsonl`
  - `failed_folds_antigravity.jsonl`
  - `method_summary_antigravity.jsonl`
  - `decision_antigravity.json`
  - `dashboard_snapshot_antigravity.json`
  - `artifact_sha256_antigravity.json`
- replay tamper matrix covered:
  - `amplitude_auc`
  - `cosine_counts`
  - `pair_counts`
  - `summary`
  - `decision`
  - `dashboard`
  - `missing_leaf`
  - `extra_leaf`
  - `reordered_rows`
- frozen Exp463 control-row digest provenance:
  - reproduced under the fixed reviewed Exp463 runtime only after the canonical
    `method_fold_metrics_antigravity.jsonl` bytes rehashed to
    `84f610e437c59cb6dce4bd73e2ae8ec09c056d7a41178c3f7b790c0f01803fa6`
  - the accepted 40 C0/C1/C2/C7 per-fold digests are now frozen in
    `config_antigravity.py` and validated exactly by order, type, and value
  - `capture_exp463_authority_antigravity()` no longer caches a prior success
    and no longer rescored Exp463 or loaded TRAIN data during capture

## Additional verification

- `py_compile`: `OK`
- map refresh + validate-only: `OK` for the Exp464 triplet after rebuilding `codemap_exp464_antigravity.lock`
- `git diff --check`: `OK`
- strict HTML/JSON parse, official-root absence, zero official evaluate/inspect/label counts, and null official result truth across the three ledgers: `OK`

## Status truth

- status remains `task_4_implemented_awaiting_fresh_review`
- checkbox truth remains exactly `21/42`
- Task 4 heading and all Task 4 steps remain unchecked
- Task 5 stays blocked until fresh Task 4 review closes

## Concerns

- Exp463 predecessor authority smoke passed, but Exp463's own governance validate-only map tests currently enforce that experiment's older source-parent fingerprint rule and were not rewritten in this Task 4 fix round because they are outside the owned Exp464 surface.
- fresh Task 4 review remains the only product gate before any Task 5 work.
