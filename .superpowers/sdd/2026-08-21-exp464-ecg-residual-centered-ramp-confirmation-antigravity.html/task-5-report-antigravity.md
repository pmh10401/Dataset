# Exp464 Task 5 report antigravity

## Scope

- Phase A only.
- Implemented and hardened the closed Exp464 CLI boundary, outside-CWD launcher, and focused CLI tests.
- Sealed the private seam so it can inject only a distinct non-official root.
- Added the internal-only official lifecycle authority path that future Task 6 evaluate→inspect can use without reopening the public CLI surface.
- Did not start either private TRAIN reproduction.
- Did not modify Exp464 result truth, ledgers, checkbox state, official root, or Task 6 state.

## RED evidence

Command:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

Observed RED before the hardening round:

- `Ran 10 tests`
- `FAILED (errors=3)`
- Expected boundary-only failures:
  - private seam still replaced frozen CLI dependencies and validation hooks
  - synthetic official-root tests still targeted frozen config replacement instead of the explicit official-root authority seam

## GREEN evidence

Focused CLI suite, pass 1:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

- `Ran 10 tests`
- `OK`

Focused CLI suite, pass 2:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

- `Ran 10 tests`
- `OK`

Full Exp464 discovery once after CLI integration:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest discover -s experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests -p 'test_*_antigravity.py'
```

- `Ran 52 tests`
- `OK`

Additional verification after the report/map updates:

- governance suite: `Ran 1 test` / `OK`
- `compileall` for Exp464 package plus Task 5 launcher/map builder: `OK`
- `scratch/build_codemap_exp464_antigravity.py --validate-only`: `OK`
- `git diff --check`: `OK`
- official root direct check: absent, and CLI/artifacts/replay root constants match

## Implemented files

- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/cli_antigravity.py`
- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/artifacts_antigravity.py`
- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/replay_antigravity.py`
- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_cli_antigravity.py`
- `scratch/run_exp464_ecg_residual_centered_ramp_confirmation_antigravity.py`
- `scratch/build_codemap_exp464_antigravity.py`
- `.superpowers/sdd/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/task-5-report-antigravity.md`

## Boundary preserved

- Public surface still accepts exactly:
  - `evaluate --reviewed-commit LOWER40HEX`
  - bare `inspect`
- Private seam can inject only a non-official root.
- Frozen clean-commit, duplicate-process, qualification, evaluation, publication, and replay paths remain bound inside the CLI and cannot be replaced by the private seam.
- Evaluate publishes exactly seven leaves and then emits the same canonical replay-backed JSON that inspect emits.
- Inspect remains source-free with no TRAIN load, no TEST-label path, and no public official-root access.
- The exact official root is structurally reachable only through an internal official lifecycle authority path reserved for future Task 6 evaluate→inspect.
- Official Exp464 root remains absent.
- No private run root was created in this Phase A report.

## Commit boundary

- Functional commit hash: recorded after commit in terminal handoff.
- Map-only child hash: recorded after commit in terminal handoff.
- This report intentionally does not embed its own commit hash or the child hash because those values are only final after the commits are created.
