# Exp460 Task 3 report

- Scope:
  - `experiments/exp460_ecg_strict_fallback_antigravity/fallback_antigravity.py`
  - `experiments/exp460_ecg_strict_fallback_antigravity/evaluation_antigravity.py`
  - `experiments/exp460_ecg_strict_fallback_antigravity/contracts_antigravity.py`
  - `experiments/exp460_ecg_strict_fallback_antigravity/tests/test_fallback_antigravity.py`
  - `experiments/exp460_ecg_strict_fallback_antigravity/tests/test_evaluation_antigravity.py`
  - `experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py`
  - `docs/EXP460_DASHBOARD_antigravity.html`
  - `docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html`
  - `docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html`
  - `scratch/build_codemap_exp460_antigravity.py`
  - `docs/codemap/codemap_exp460_antigravity.{html,json,lock}`

- Exp460 codemap grounding before edits:
  - Callers: Task 2 predecessor capture feeds strict routing and four-rule evaluation; governance regression reads the dashboard, report, plan, and stored Exp460 map triplet.
  - Effects: Task 3 changes only the strict fallback scientific evaluation surface plus the bounded Exp460 docs/map progress surface.
  - Tests: `test_fallback_antigravity.py`, `test_evaluation_antigravity.py`, `test_governance_antigravity.py`, plus unchanged Task 2 contract/predecessor coverage.

- RED:
  - Count: 1 run.
  - Command: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest experiments.exp460_ecg_strict_fallback_antigravity.tests.test_fallback_antigravity experiments.exp460_ecg_strict_fallback_antigravity.tests.test_evaluation_antigravity`
  - Exact summary: `FAILED (errors=7)` because `fallback_antigravity.py`, `evaluation_antigravity.py`, and the immutable Task 3 result contracts did not exist yet.

- GREEN:
  - Focused Task 3 GREEN count: 2 runs.
  - Command: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest experiments.exp460_ecg_strict_fallback_antigravity.tests.test_fallback_antigravity experiments.exp460_ecg_strict_fallback_antigravity.tests.test_evaluation_antigravity`
  - Exact summary on both runs: `Ran 7 tests`, `OK`.

- Exact four-row current-authority result:
  - `always_baseline` → `baseline` → `TP=609 FP=66 FN=837 TN=70788 F1=0.5742574257425742`
  - `always_candidate` → `candidate` → `TP=617 FP=66 FN=829 TN=70788 F1=0.5796148426491311`
  - `majority_8_of_10` → `candidate` → `TP=617 FP=66 FN=829 TN=70788 F1=0.5796148426491311`
  - `strict_10_of_10` → `baseline` → `TP=609 FP=66 FN=837 TN=70788 F1=0.5742574257425742`

- Strict outcome:
  - Passed records: `8/10`
  - Worst `period_change` AUC: `0.8004124541019063`
  - Failed records:
    - `seed 20260721 fold 3 period_change_auc 0.8208339620743423`
    - `seed 20261721 fold 4 period_change_auc 0.8004124541019063`
  - Strict route: `baseline`
  - Strict final status: `fallback_triggered_performance_not_improved`
  - Joint success: `false`
  - Meaning: majority remains `candidate`, but the official strict gate rejects the candidate because not all 10 records pass and the strict row therefore reverts to baseline performance.

- Mutation and contract coverage:
  - Strict route table tests cover `10/10`, `9/10`, `8/10`, `7/10`, and `0/10`.
  - Majority uses candidate at `>=8/10`; strict uses candidate only at `10/10`.
  - Evaluation recomputes F1 from confusion counts and rejects stored/forged semantic drift.
  - Malformed authority rejection covers missing record count, reordered/incomplete rows, bool-as-int pass flags, and non-authority objects.
  - Result-contract rejection covers reordered rows, forged `strict_status`, forged `joint_success`, bool-as-int confusion counts, and non-finite metrics.

- Regression and verification:
  - Full Exp460 suite once: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest discover -s experiments/exp460_ecg_strict_fallback_antigravity/tests -p 'test_*_antigravity.py'`
  - Exact summary: `Ran 16 tests`, `OK`.
  - Affected Exp459 scientific regressions once: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_contracts_antigravity experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_evaluation_antigravity`
  - Exact summary: `Ran 10 tests`, `OK`.
  - `py_compile`: passed for Exp460 Task 2/3 modules, Exp460 Task 2/3/governance tests, and `scratch/build_codemap_exp460_antigravity.py`.
  - Exp460 stored map validate-only: passed.
  - `git diff --check`: passed.
  - Exp459 triplet byte guard: `git diff --exit-code 08838f34cff5fb64ab37c58cc38762eaf3b8e0d0 -- docs/codemap/codemap_exp459_antigravity.html docs/codemap/codemap_exp459_antigravity.json docs/codemap/codemap_exp459_antigravity.lock` passed.

- Map/docs/checkbox state:
  - Stored graph schema: `exp460_task3_strict_fallback_codemap_v1`
  - Stored lock schema: `exp460_task3_strict_fallback_codemap_lock_v1`
  - Source parent binding: `7a51c26d6b0faddbb6884e5d880f7fb327185308`
  - Node count: `24`
  - Edge count: `23`
  - New current Task 3 nodes: `task3_strict_route`, `task3_rule_evaluation`, `task3_rule_tests`
  - `docs/EXP460_DASHBOARD_antigravity.html` now reports `Task 3/6`, the exact four-row table, `fallback_triggered_performance_not_improved`, `joint success=false`, and unchanged official-zero state.
  - `docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html` now reports the Task 3 RED/GREEN counts, full Exp460/affected Exp459 regressions, and the exact current strict outcome.
  - Plan checkbox state:
    - Task 1 heading checked; Steps 1-6 checked.
    - Task 2 heading checked; Steps 1-6 checked.
    - Task 3 heading checked; Steps 1-6 checked.
    - Tasks 4-6 remain unchecked.

- Official/root/label state:
  - Official evaluate invocations: `0`
  - Official inspect invocations: `0`
  - Exp460 result root: `null`
  - Exp460 prediction/evaluation/decision/artifact-map/dashboard hashes: `null`
  - Label query count: `0`
  - Champion and pending candidate: unchanged.

- Clean status:
  - This report is included in the bounded Task 3 change.
  - Post-commit clean status is verified separately in the final handoff after the bounded Task 3 commit is created.

- Concerns:
  - Historical Exp459 map-parent governance provenance remains outside Task 3 scope. Task 3 did not modify any Exp459 codemap triplet bytes, and the explicit byte guard passed.

## Fix round 1

- Finding addressed:
  - Important: Task 3 joint-success validation previously trusted authority-carried baseline and candidate confusion rows after only type checks, so forged confusion drift could make the strict candidate path appear successful.
  - Minor: the regenerated Exp460 Task 3 codemap still carried stale Task 2 wording in some current-node roles and still used the earlier frozen `generated_at`.

- Fix:
  - `_task3_snapshot_from_authority_antigravity()` now rejects any baseline confusion other than the frozen Exp460 tuple `609/66/837/70788`.
  - `_task3_snapshot_from_authority_antigravity()` now rejects any candidate confusion other than the frozen Exp458 predecessor tuple `617/66/829/70788`.
  - This makes confusion drift fail before route evaluation or `joint_success` recomputation, preserving the exact real current result while closing the forged-baseline hole.
  - `scratch/build_codemap_exp460_antigravity.py` now regenerates `generated_at` with the current session timestamp and refreshes current-node roles to Task 3 wording.

- New regression coverage:
  - Focused RED: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest experiments.exp460_ecg_strict_fallback_antigravity.tests.test_evaluation_antigravity`
  - Exact RED summary: `FAILED (failures=1)` because the new forged-confusion regression was accepted before the fix.
  - Added drift cases:
    - all-pass authority with forged baseline confusion `609/70/900/70788` must raise `ValueError`
    - individual baseline drift `610/66/837/70788` must raise `ValueError`
    - individual candidate drift `617/66/830/70788` must raise `ValueError`

- Post-fix verification:
  - Focused GREEN: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest experiments.exp460_ecg_strict_fallback_antigravity.tests.test_fallback_antigravity experiments.exp460_ecg_strict_fallback_antigravity.tests.test_evaluation_antigravity` → `Ran 7 tests`, `OK`.
  - Full Exp460 regressions: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest discover -s experiments/exp460_ecg_strict_fallback_antigravity/tests -p 'test_*_antigravity.py'` → `Ran 16 tests`, `OK`.
  - Affected Exp459 scientific guard: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_contracts_antigravity experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_evaluation_antigravity` → `Ran 10 tests`, `OK`.
  - Exp460 map rebuild + validate-only: passed.
  - `git diff --check`: passed.
  - Exp459 triplet byte guard: passed.

- Refreshed map metadata:
  - Stored graph schema remains `exp460_task3_strict_fallback_codemap_v1`.
  - Stored lock schema remains `exp460_task3_strict_fallback_codemap_lock_v1`.
  - Refreshed `generated_at`: `2026-08-20T00:20:02Z`.
  - Stored source parent remains `1f519824754c31e0a2b8d1e95d2432663584b93a` for this fix round.
