# Exp464 Task 5 report antigravity

## Scope

- Phase A and Phase B.
- Implemented and hardened the closed Exp464 CLI boundary, outside-CWD launcher, and focused CLI tests.
- Sealed the private seam so it can inject only a distinct non-official root, and made evaluate preflight reject invalid private roots before clean validation or heavy compute.
- Moved the official lifecycle allowance onto the actual public dispatch path through an internal-only identity seam, so future Task 6 evaluate→inspect no longer depends on separate helper entrypoints.
- Fixed the duplicate-process guard so the current shell/time wrapper ancestor chain is ignored while a genuinely separate active Exp464 launcher/process is still rejected.
- Ran exactly two sequential private non-official evaluate→inspect reproductions at reviewed clean child commit `b8646d0fa9bd57dcb1f6ab5e251064d34911a41c`.
- Did not run any official command.
- Did not open any TEST label path.

## Pre-science disclosure

Retained wrapper failures that do not count as scientific reproductions:

- first pre-science stop: `reviewed commit must be the exact clean current commit`
  - stderr SHA-256: `8a3a459348603f9c79fd68fdd39cd39594dd87e63547e539a147bad29a4657cf`
  - time SHA-256: `e4a4cb4a2aa2fb6d1f64f82e137be621de64c2b92de18d41d25572d8949b90db`
  - stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  - root created: `false`
- corrected pre-science stop before the duplicate-process fix: `duplicate Exp464 process is already active`
  - stderr SHA-256: `e67bee7ce606726713bd7aaa6df6a2cf978094c3d68123372b4f3d9bb23be1a4`
  - time SHA-256: `539f1596e3b178a02613d497aadf76b6bf99d0df3bab746117cc74e40e28b68a`
  - stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  - root created: `false`

Both failures were pre-science, label-free, publication-free, and official-root-free.

## RED evidence

Focused CLI suite before the duplicate-process guard fix:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

- `Ran 16 tests`
- `FAILED (errors=1)`
- The new regression test reproduced the controller-observed false positive where the current `zsh` + `/usr/bin/time` ancestor wrapper chain still contained the launcher path and triggered `duplicate Exp464 process is already active`.

## GREEN evidence

Focused CLI suite after the duplicate-process guard fix, pass 1:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

- `Ran 16 tests`
- `OK`

Focused CLI suite after the duplicate-process guard fix, pass 2:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

- `Ran 16 tests`
- `OK`

Full Exp464 suite, pass 1 after the Task 5 truth refresh:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest discover -s experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests -p 'test_*_antigravity.py'
```

- `Ran 58 tests`
- `OK`

Full Exp464 suite, pass 2 after the Task 5 truth refresh:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest discover -s experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests -p 'test_*_antigravity.py'
```

- `Ran 58 tests`
- `OK`

Exp463 authority/science regression:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_config_antigravity experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_contracts_antigravity experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_predecessor_antigravity experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_sources_antigravity experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_ramp_features_antigravity experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_scoring_antigravity
```

- `Ran 19 tests`
- `OK`

Standalone governance suite after the truth refresh:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_governance_antigravity
```

- `Ran 1 test`
- `OK`

Additional verification:

- `compileall` for the Exp464 package plus launcher and map builder: `OK`
- `scratch/build_codemap_exp464_antigravity.py --validate-only`: `OK`
- strict JSON parse for the three ledgers and the Exp464 map JSON/lock: `OK`
- `git diff --check`: `OK`
- official root direct check: absent and non-symlink
- both explicit private roots direct check before cleanup: present and non-symlink
- both original explicit private roots direct check after cleanup: absent and non-symlink
- both recoverably moved private-root entries direct check after cleanup: present and non-symlink
- official evaluate / inspect counts: `0 / 0`
- label query count: `0`
- active Exp464 launcher/process count after verification: `0`

## Scientific run evidence

Run 1 evaluate:

- stdout SHA-256: `f41151e67a07b33bae607437501317e61c79d8f5508b62d0886b210b63180111`
- stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- time SHA-256: `4f84114f60ee79902c03f44cf0dfa1a8923422c1c6293100ef4c8f0f9fd4096b`
- elapsed: `real 269.67 / user 436.20 / sys 407.10`

Run 1 inspect:

- stdout SHA-256: `f41151e67a07b33bae607437501317e61c79d8f5508b62d0886b210b63180111`
- stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- time SHA-256: `55e175bd55f0c2d2661966c542a796665c1a131ef31d58358bc39cdadd2ee4a4`
- elapsed: `real 6.62 / user 6.41 / sys 0.18`

Run 2 evaluate:

- stdout SHA-256: `f41151e67a07b33bae607437501317e61c79d8f5508b62d0886b210b63180111`
- stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- time SHA-256: `d3e6de76747bcb23bc9200cdf3492c161cb0dc74a04e4777469fc20ccec56522`
- elapsed: `real 261.50 / user 440.15 / sys 415.37`

Run 2 inspect:

- stdout SHA-256: `f41151e67a07b33bae607437501317e61c79d8f5508b62d0886b210b63180111`
- stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- time SHA-256: `1a388f646b27c9dd8020120c1d332b2b3d45db40d6843d72a879def8b464bf6e`
- elapsed: `real 6.64 / user 6.41 / sys 0.20`

Exact equality:

- run1 evaluate == run1 inspect: `true`
- run2 evaluate == run2 inspect: `true`
- run1 == run2: `true`

Recoverably moved private roots after review:

- `/private/tmp/exp464_task5_private_removed_after_review_antigravity_20260821/exp464_task5_private_run1_antigravity`
- `/private/tmp/exp464_task5_private_removed_after_review_antigravity_20260821/exp464_task5_private_run2_antigravity`

Original explicit private roots are absent after the move:

- `/private/tmp/exp464_task5_private_run1_antigravity`
- `/private/tmp/exp464_task5_private_run2_antigravity`

Seven-leaf inventory in both roots:

- `artifact_sha256_antigravity.json`
- `context_antigravity.json`
- `dashboard_snapshot_antigravity.json`
- `decision_antigravity.json`
- `failed_folds_antigravity.jsonl`
- `method_fold_metrics_antigravity.jsonl`
- `method_summary_antigravity.jsonl`

Exact leaf bytes and SHA-256:

- `artifact_sha256_antigravity.json` — `605` bytes — `8e0cc3391f5acf970fb73a322a2ec313538cd2c4d153114e07b2d480f845bbca`
- `context_antigravity.json` — `1180` bytes — `577a8a989abaf7b64d69968578e3d15b2b597acb97618c7d71eeaaf45fc46653`
- `dashboard_snapshot_antigravity.json` — `3839` bytes — `da1e2333728b2886cea918f4c34ad6d084d10e1811b48c8903c0946ad167e1cb`
- `decision_antigravity.json` — `428` bytes — `d66a23ef0749833220700f7d6b3ec56422722a9167d443cf2f58d327ce9593cd`
- `failed_folds_antigravity.jsonl` — `5511663` bytes — `54eefdbb8bbc3d2495bfa1cd0d444c04ffa9845c5f12f4e79f6bc6e25717bff3`
- `method_fold_metrics_antigravity.jsonl` — `53043215` bytes — `51073f3caf4793a2a70602547d6f249e42bcc7dcd64709ab5a62ccfeba791bcf`
- `method_summary_antigravity.jsonl` — `3899` bytes — `3f7ad67d51c7dd63a31e9f373b14eb2ce2ffc616a81a611db2dd599e083f5540`

Exact scientific counts:

- metric rows: `90`
- failed rows: `30`
- summaries: `9`
- leaves: `7`

## Frozen decision

- Exp464 status: `stopped_before_official_evaluation_qualification_failed`
- reviewed scientific commit: `b8646d0fa9bd57dcb1f6ab5e251064d34911a41c`
- primary candidate: `template_residual_centered_ramp_confirmed_max`
- primary passed count: `9 / 10`
- decision status: `qualification_failed_blocked_before_routing_research`
- label query count: `0`
- routing applied: `false`
- eligible for routing research: `false`

Primary failure reasons:

- `primary_candidate_not_10_of_10`
- `primary_gradual_drift_auc_below_min`
- `primary_gradual_drift_min_below_0.85`

Key C6 metrics:

- min gradual_drift AUC: `0.8402557718424627`
- required gradual_drift AUC minimum: `0.85`
- min period_change AUC: `0.907392736783864`
- historical period floor requirements:
  - seed `20260721`, fold `3`, minimum `0.907392736783864`
  - seed `20261721`, fold `4`, minimum `0.9216148584075248`
- max normal Wilson upper: `0.013439124039399194`
- min mean AUC: `0.9212328982445551`

Non-degradation summary:

- gradual rows not below C0: `10`
- period rows not below C0: `10`

The frozen decision is qualification failure. Final review is SHIP (spec PASS, quality APPROVED, C/I/M0). Task 5 is now 7/7 checked, Task 6 is permanently forbidden and remains unchecked, and there is no next official action. Champion, pending candidate, routing, promotion, and scientific winner state remain unchanged.

## Implemented and touched files

- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/cli_antigravity.py`
- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_cli_antigravity.py`
- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_governance_antigravity.py`
- `scratch/build_codemap_exp464_antigravity.py`
- `docs/EXP464_DASHBOARD_antigravity.html`
- `docs/EXP464_PREREGISTRATION_antigravity.html`
- `docs/EXP464_RUNBOOK_antigravity.html`
- `docs/EXP464_IMPLEMENTATION_REPORT_antigravity.html`
- `docs/superpowers/plans/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html`
- `RESEARCH_STATE.json`
- `EXPERIMENT_CATALOG.json`
- `CHAMPION_LEDGER.json`
- `.superpowers/sdd/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/task-5-report-antigravity.md`

## Boundary preserved

- Public surface still accepts exactly:
  - `evaluate --reviewed-commit LOWER40HEX`
  - bare `inspect`
- Private seam injects only a non-official root.
- The duplicate-process guard ignores only the current PID and its ancestor chain.
- A genuinely separate active Exp464 launcher/process is still rejected.
- Official root remained absent throughout Task 5.
- Official evaluate / inspect counts stayed `0 / 0`.
- TEST label-query count stayed `0`.
- Both scientific private roots were reverified after review, then recoverably moved into `/private/tmp/exp464_task5_private_removed_after_review_antigravity_20260821`. Aggregate logs, report, and hashes remain retained.

## Commit boundary

- Functional truth/docs/governance commit hash: recorded in terminal handoff after commit.
- Map-only child hash: recorded in terminal handoff after commit.
- This report intentionally keeps those hashes out of the body until both commits are final.
