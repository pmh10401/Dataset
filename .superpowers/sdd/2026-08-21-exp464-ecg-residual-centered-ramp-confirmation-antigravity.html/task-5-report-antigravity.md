# Exp464 Task 5 report antigravity

## Scope

- Phase A only.
- Implemented the closed Exp464 CLI boundary, outside-CWD launcher, and focused CLI tests.
- Did not start either private TRAIN reproduction.
- Did not modify Exp464 result truth, ledgers, checkbox state, official root, or Task 6 state.

## RED evidence

Command:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

Observed RED before implementation:

- `Ran 7 tests`
- `FAILED (failures=4, errors=3)`
- Expected missing-boundary failures only:
  - missing `cli_antigravity`
  - missing `scratch/run_exp464_ecg_residual_centered_ramp_confirmation_antigravity.py`

## GREEN evidence

Focused CLI suite, pass 1:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

- `Ran 7 tests`
- `OK`

Focused CLI suite, pass 2:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_cli_antigravity
```

- `Ran 7 tests`
- `OK`

Full Exp464 discovery once after CLI integration:

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
"$PY" -m unittest discover -s experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests -p 'test_*_antigravity.py'
```

- `Ran 49 tests`
- `OK`

## Implemented files

- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/cli_antigravity.py`
- `experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_cli_antigravity.py`
- `scratch/run_exp464_ecg_residual_centered_ramp_confirmation_antigravity.py`

## Boundary preserved

- Public surface still accepts exactly:
  - `evaluate --reviewed-commit LOWER40HEX`
  - bare `inspect`
- Private seam can inject only a non-official root.
- Evaluate publishes exactly seven leaves and then emits the same canonical replay-backed JSON that inspect emits.
- Inspect remains source-free with no TRAIN load, no TEST-label path, and no official-root access.
- Official Exp464 root remains absent.
- No private run root was created in this Phase A report.

## Commit boundary

- Functional commit hash: recorded after commit in terminal handoff.
- Map-only child hash: recorded after commit in terminal handoff.
- This report intentionally does not embed its own commit hash or the child hash because those values are only final after the commits are created.
