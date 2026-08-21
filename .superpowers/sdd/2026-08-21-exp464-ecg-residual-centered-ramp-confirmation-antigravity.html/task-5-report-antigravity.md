# Exp464 Task 5 report antigravity

## Scope

- Phase A only.
- Implemented and hardened the closed Exp464 CLI boundary, outside-CWD launcher, and focused CLI tests.
- Sealed the private seam so it can inject only a distinct non-official root, and made evaluate preflight reject invalid private roots before clean validation or heavy compute.
- Moved the official lifecycle allowance onto the actual public dispatch path through an internal-only identity seam, so future Task 6 evaluate→inspect no longer depends on separate helper entrypoints.
- Did not start either private TRAIN reproduction.
- Did not modify Exp464 result truth, ledgers, checkbox state, official root, or Task 6 state.

## RED evidence

Command:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

Observed RED before round 2:

- `Ran 14 tests`
- `FAILED (failures=8)`
- Expected boundary-only failures:
  - private-root rejection still happened after clean validation or after qualification entry
  - public dispatch still routed official evaluate/inspect through helper-only seams instead of carrying the internal authority itself
  - public interface still exposed official-lifecycle minting helpers

## GREEN evidence

Focused CLI suite, pass 1:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

- `Ran 14 tests`
- `OK`

Focused CLI suite, pass 2:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

- `Ran 14 tests`
- `OK`

Full Exp464 discovery once after CLI integration:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest discover -s experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests -p 'test_*_antigravity.py'
```

- `Ran 56 tests`
- `OK`

Additional verification after the report/map updates:

- governance suite: `Ran 1 test` / `OK`
- `compileall` for Exp464 package plus Task 5 launcher/map builder: `OK`
- `scratch/build_codemap_exp464_antigravity.py`: `OK`
- `git diff --check`: `OK`
- official root direct check: absent, CLI/artifacts/replay root constants match, official authority factory is not exported, and public helper minting entrypoints are absent

## Implemented files

- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/cli_antigravity.py`
- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/artifacts_antigravity.py`
- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/replay_antigravity.py`
- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_cli_antigravity.py`
- `scratch/build_codemap_exp464_antigravity.py`
- `.superpowers/sdd/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/task-5-report-antigravity.md`

## Boundary preserved

- Public surface still accepts exactly:
  - `evaluate --reviewed-commit LOWER40HEX`
  - bare `inspect`
- Private seam can inject only a non-official root, and evaluate rejects non-Path, official-alias, symlink, existing, and rebound private roots before validation or qualification can proceed.
- Frozen clean-commit, duplicate-process, qualification, evaluation, publication, and replay paths remain bound inside the CLI and cannot be replaced by the private seam.
- Evaluate publishes exactly seven leaves and then emits the same canonical replay-backed JSON that inspect emits.
- Inspect remains source-free with no TRAIN load and no TEST-label path.
- The actual public dispatch path now carries the exact official root only through an internal official lifecycle identity that is absent from public interfaces and helper exports.
- Official Exp464 root remains absent.
- No private run root was created in this Phase A report.

## Commit boundary

- Functional commit hash: recorded after commit in terminal handoff.
- Map-only child hash: recorded after commit in terminal handoff.
- This report intentionally does not embed its own commit hash or the child hash because those values are only final after the commits are created.
