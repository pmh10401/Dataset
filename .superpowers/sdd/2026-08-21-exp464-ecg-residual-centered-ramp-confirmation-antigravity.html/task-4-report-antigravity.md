# Exp464 Task 4 report antigravity

## Commit boundary

- Fix-round product base provided by controller: `2de05e639a0fc31c932ae14b9cf190d2fb5e0825`
- Pre-existing functional parent on this branch before round-2 fixes: `a8f3fa905fd07342cdc1c34355e9a90a24686c06`
- This report is part of the functional fix payload. The final functional and map-only child commit hashes are recorded in the terminal handoff because embedding a commit's own hash inside the same commit would change that hash.

## Code-map preflight

- Checked before edits:
  - `docs/codemap/codemap_exp464_antigravity.html`
  - `docs/codemap/codemap_exp464_antigravity.json`
  - `docs/codemap/codemap_exp464_antigravity.lock`
- Callers confirmed before edits:
  - Task 2 frozen authority/source modules
  - Task 3 centered-ramp and scoring modules
  - Task 4 qualification/evaluation/artifacts/replay/dashboard tests
  - future Task 5 private qualification handoff only after fresh review
- Effects confirmed before edits:
  - exact per-fold control binding from frozen Exp463 evidence
  - one fixed C6 decision only
  - exactly seven canonical publication leaves
  - semantic replay from raw evidence carriers
  - aggregate-only dashboard export

## Scope completed in this fix round

- Added frozen predecessor-evidence modules:
  - `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/control_row_evidence_antigravity.py`
  - `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/exp463_control_row_evidence_antigravity.jsonl`
- Added raw-evidence replay carrier:
  - `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/raw_evidence_antigravity.py`
- Updated Exp464 product modules:
  - `config_antigravity.py`
  - `contracts_antigravity.py`
  - `predecessor_antigravity.py`
  - `scoring_antigravity.py`
  - `qualification_antigravity.py`
  - `artifacts_antigravity.py`
  - `replay_antigravity.py`
- Updated Exp464 tests:
  - `test_config_antigravity.py`
  - `test_predecessor_antigravity.py`
  - `test_qualification_antigravity.py`
  - `test_evaluation_antigravity.py`
  - `test_replay_antigravity.py`
  - `test_dashboard_export_antigravity.py`
  - `test_governance_antigravity.py`
- Updated review-pending truth/docs:
  - `docs/EXP464_DASHBOARD_antigravity.html`
  - `docs/EXP464_IMPLEMENTATION_REPORT_antigravity.html`
  - `docs/superpowers/plans/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html`
  - `RESEARCH_STATE.json`
  - `EXPERIMENT_CATALOG.json`
  - `CHAMPION_LEDGER.json`
  - `scratch/build_codemap_exp464_antigravity.py`
- Did not perform:
  - Task 5 CLI work
  - Exp464 private TRAIN qualification
  - Exp464 official evaluate/inspect
  - TEST label queries
  - champion/pending/routing/promotion changes

## Exp463 one-off control-row recovery evidence

- Reviewed Exp463 closeout commit frozen by authority: `147225831e9c13e9df5043fa88757bb03e01b79a`
- Reviewed Exp463 runtime commit whose relevant tracked bytes matched exactly during the one-off recovery:
  - `923cc4ad6dd6737f84f4ea2c4f621eb652ac54b4`
- Fixed runtime used for the recovery and for current verification:
  - `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python`
- Frozen recovered leaf digest required by controller and preserved in tracked provenance:
  - `84f610e437c59cb6dce4bd73e2ae8ec09c056d7a41178c3f7b790c0f01803fa6`
- Frozen recovered artifact-map digest preserved in tracked provenance:
  - `25fb4ab237ae8f2601c2bc03b05f4e6a2d0cb9889876a83fa0f2c4ee7c4f0ea5`
- Tracked evidence file:
  - `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/exp463_control_row_evidence_antigravity.jsonl`
- Evidence-file digest:
  - `036d2bccd40d9beac4a20c2e5bc60ba60bd72d5a076d433aceb5e700b414f3c9`
- Current re-checks:
  - relevant Exp463 runtime files still diff-clean versus `923cc4ad6dd6737f84f4ea2c4f621eb652ac54b4`
  - temporary recovery root `/private/tmp/exp463_task5_control_row_recovery_antigravity` is absent
  - official Exp463 and Exp464 roots are absent

## RED evidence from this round

- Raw-evidence threading initially broke the focused source-free fixture:
  - missing raw fields in `FoldMethodScoresAntigravity` fixture construction
  - illegal raw blob carriage by C3/C4
  - replay parser assumed every row carried raw query blobs
- New semantic replay path initially exposed two real fixture bugs:
  - impossible zero-amplitude raw fit vectors
  - sign-inconsistent synthetic query amplitudes causing cosine replay drift
- Governance later exposed stale truth:
  - plan and docs still reported earlier focused-suite counts
  - code-map triplet became stale after truth/report changes

## GREEN evidence

### Focused Task 4 suite

Command:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest -q \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_qualification_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_evaluation_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_artifacts_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_replay_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_dashboard_export_antigravity
```

Observed current GREEN:

- `Ran 15 tests`
- `OK`

The focused suite now covers:

- source-free focused qualification that must not load TRAIN or call the production fold scorer
- exact 10/10 C6 positive decision fixture
- tie handling for pair diagnostics
- exact raw-evidence semantic replay for amplitude and cosine
- dashboard key/value allowlist behavior

### Exp464 Task 2–3 regression

Command:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest -q \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_config_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_contracts_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_predecessor_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_sources_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_centered_ramp_antigravity \
  experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_scoring_antigravity
```

Observed current GREEN:

- `Ran 25 tests`
- `OK`

### Exp463 scientific and authority regression

Command:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest -q \
  experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_ramp_features_antigravity \
  experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_scoring_antigravity \
  experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_predecessor_antigravity
```

Observed current GREEN:

- `Ran 10 tests`
- `OK`

### Full Exp464 package verification

Command:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest -q \
  $(rg --files 'experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests' | sed 's#/#.#g; s#.py$##')
```

Observed current GREEN:

- `Ran 41 tests`
- `OK`

## Direct publication / replay evidence

- Synthetic private temp-root publish→inspect equality:
  - `publish_result_antigravity(...)` succeeded at a fresh temp root
  - `inspect_replay_at_antigravity(...)` returned exact qualification equality
  - decision equality and artifact-map equality both held
- Direct temp-root semantic tamper checks rejected as expected:
  - `amplitude`
  - `cosine`
  - `reordered`

The broader semantic tamper matrix is also covered by the focused replay suite and remains GREEN there, including:

- rank-preserving amplitude mutation `+0.001`
- same-side cosine `1.0 -> 0.97`
- decision / summary / dashboard / missing-leaf / extra-leaf rejection

## Result integrity after fixes

- Publication still emits exactly these seven leaves:
  - `context_antigravity.json`
  - `method_fold_metrics_antigravity.jsonl`
  - `failed_folds_antigravity.jsonl`
  - `method_summary_antigravity.jsonl`
  - `decision_antigravity.json`
  - `dashboard_snapshot_antigravity.json`
  - `artifact_sha256_antigravity.json`
- Control validation now binds exact per-fold C0/C1/C2/C7 evidence from the tracked Exp463 recovery file instead of aggregate-only summaries.
- Semantic replay now derives score evidence from raw vector blobs for carrier methods and forces C3/C4 to borrow C6 raw provenance without carrying their own blobs.
- Dashboard export remains aggregate-only and registered-key/scalar-safe.

## Additional verification

- `py_compile` over Exp464 Python modules and `scratch/build_codemap_exp464_antigravity.py`: `OK`
- `git diff --check`: `OK`
- strict HTML parse:
  - `docs/EXP464_DASHBOARD_antigravity.html`
  - `docs/EXP464_PREREGISTRATION_antigravity.html`
  - `docs/EXP464_RUNBOOK_antigravity.html`
  - `docs/EXP464_IMPLEMENTATION_REPORT_antigravity.html`
  - `docs/superpowers/plans/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html`
  - `OK`
- strict JSON parse:
  - `RESEARCH_STATE.json`
  - `EXPERIMENT_CATALOG.json`
  - `CHAMPION_LEDGER.json`
  - `docs/codemap/codemap_exp464_antigravity.json`
  - `docs/codemap/codemap_exp464_antigravity.lock`
  - `OK`
- code-map rebuild and `--validate-only`: `OK`
- current root / label / process checks:
  - official Exp463 root present: `false`
  - official Exp464 root present: `false`
  - recovery temp root present: `false`
  - active Exp464 non-test process list: `[]`

## Status truth preserved

- status remains `task_4_implemented_awaiting_fresh_review`
- checkbox truth remains exactly `21/42`
- Task 4 heading and Task 4 steps remain unchecked
- no Exp464 private / official / TEST-label lifecycle was opened

## Remaining gate outside this report

- The only remaining product gate before any Task 5 work is fresh Task 4 review.
