# Exp460 Task 1 report

- Scope: governance, progress dashboard, planned ledger entries, Exp460-only codemap, governance regression, Task 1 plan checkbox updates.
- Exp459 codemap grounding:
  - Callers: governance review, future operators, implementation review, reader-facing review, catalog review, champion review.
  - Effects: planned-only Exp460 docs/ledgers/map now freeze rule order, predecessor unstable truth, official-zero/null state, and non-promotion boundary.
  - Tests: `experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py`.

- RED:
  - Count: 1 run.
  - Command: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest experiments.exp460_ecg_strict_fallback_antigravity.tests.test_governance_antigravity`
  - Exact summary: `FAILED (failures=1, errors=1)` because `docs/EXP460_PREREGISTRATION_antigravity.html` and `docs/EXP460_DASHBOARD_antigravity.html` did not exist.

- GREEN:
  - Count: 2 runs.
  - Command: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest experiments.exp460_ecg_strict_fallback_antigravity.tests.test_governance_antigravity`
  - Exact summary: `OK` on both runs (`Ran 2 tests`).

- Files changed:
  - `docs/EXP460_PREREGISTRATION_antigravity.html`
  - `docs/EXP460_RUNBOOK_antigravity.html`
  - `docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html`
  - `docs/EXP460_DASHBOARD_antigravity.html`
  - `RESEARCH_STATE.json`
  - `EXPERIMENT_CATALOG.json`
  - `CHAMPION_LEDGER.json`
  - `scratch/build_codemap_exp460_antigravity.py`
  - `docs/codemap/codemap_exp460_antigravity.html`
  - `docs/codemap/codemap_exp460_antigravity.json`
  - `docs/codemap/codemap_exp460_antigravity.lock`
  - `experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py`
  - `docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html`
  - `.superpowers/sdd/2026-08-20-exp460-ecg-strict-fallback-antigravity.html/task-1-report.md`

- Map:
  - Node count: 14.
  - Edge count: 12.
  - Binding: current Task 1 graph covers approved design, implementation plan, four HTML docs, three ledgers, governance regression, and the map builder; future downstream nodes are limited to predecessor capture, rule evaluation, and CLI/artifact flow.

- Verification:
  - Governance GREEN twice: passed.
  - HTML parse: passed for four Exp460 docs plus `docs/codemap/codemap_exp460_antigravity.html`.
  - Strict JSON parse with duplicate-key rejection: passed for `RESEARCH_STATE.json`, `EXPERIMENT_CATALOG.json`, `CHAMPION_LEDGER.json`, `docs/codemap/codemap_exp460_antigravity.json`, and `docs/codemap/codemap_exp460_antigravity.lock`.
  - Codemap validate-only: passed.
  - `git diff --check`: passed.
  - `git diff --exit-code 08838f34cff5fb64ab37c58cc38762eaf3b8e0d0 -- docs/codemap/codemap_exp459_antigravity.html docs/codemap/codemap_exp459_antigravity.json docs/codemap/codemap_exp459_antigravity.lock`: passed.

- Official/root/label state:
  - Official evaluate invocations: 0.
  - Official inspect invocations: 0.
  - Result root created: false.
  - Official result: null.
  - Prediction/evaluation/decision/artifact-map/dashboard hashes: null.
  - Label query count: 0.
  - Champion and pending candidate: unchanged.

- Checkbox state:
  - Task 1 heading checked.
  - Task 1 Steps 1-6 checked.
  - Future tasks remain unchecked.

- Commit:
  - Exact commit hash is reported in the final handoff message. Embedding it in this file before commit would require a follow-up amend.

- Concerns:
  - None on the Task 1 scope itself. The only procedural caveat is the report/commit-hash note above.

## Fix round 1

- Review scope:
  - Make `scratch/build_codemap_exp460_antigravity.py --validate-only` load and validate the stored HTML/JSON/lock triplet instead of rebuilding an in-memory map.
  - Represent future predecessor, rule/evaluation, artifacts/replay/dashboard, and CLI boundaries plus their future tests/evidence truthfully while keeping the graph bounded.

- RED:
  - Count: 1 run.
  - Command: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest experiments.exp460_ecg_strict_fallback_antigravity.tests.test_governance_antigravity`
  - Exact summary: `FAILED (failures=2)` because `future_predecessor_tests` was missing from the stored graph and `--validate-only` returned exit 0 even after packaged HTML tamper.

- GREEN:
  - Count: 2 runs.
  - Command: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -m unittest experiments.exp460_ecg_strict_fallback_antigravity.tests.test_governance_antigravity`
  - Exact summary: `OK` on both runs (`Ran 3 tests`).

- Builder/map changes:
  - `--validate-only` now loads `docs/codemap/codemap_exp460_antigravity.{html,json,lock}` from disk and validates those stored bytes.
  - HTML parity is now exact against the rendered graph, so packaged-byte tamper fails closed.
  - Future nodes now explicitly cover:
    - predecessor capture
    - predecessor tests
    - rule/evaluation
    - rule tests
    - artifacts/replay/dashboard
    - artifacts tests
    - CLI/launcher
    - CLI tests
  - Updated map size: 19 nodes, 17 edges.

- Verification:
  - Governance GREEN twice: passed.
  - Stored map `--validate-only`: passed.
  - HTML parse: passed.
  - Strict JSON duplicate-key parse: passed.
  - `git diff --check`: passed.
  - `git diff --exit-code 08838f34cff5fb64ab37c58cc38762eaf3b8e0d0 -- docs/codemap/codemap_exp459_antigravity.html docs/codemap/codemap_exp459_antigravity.json docs/codemap/codemap_exp459_antigravity.lock`: passed.

- Official/root/label state:
  - Unchanged from Task 1: official evaluate/inspect 0/0, result root false, hashes null, label query count 0, champion and pending candidate unchanged.

- Concerns:
  - None on fix-round scope.
