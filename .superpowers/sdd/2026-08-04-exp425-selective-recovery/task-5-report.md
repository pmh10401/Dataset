# Task 5 — Selective MPS Runtime and Explicit Resume

Status: implementation in progress; the official population and readiness gate
were not started.

## TDD evidence

The initial focused command was intentionally red because neither owned module
existed:

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime \
  experiments.exp425_selective_recovery.tests.test_cli -v
ModuleNotFoundError: No module named 'experiments.exp425_selective_recovery.runtime'
```

The implemented coverage currently exercises atomic canonical progress/JSONL
writes, parent-log and PID creation, root lifecycle rejection for `new` and
`resume`, immutable inspection, and duplicate terminal-event rejection.

## MLX / MPS assessment

Exp425 adds no score source and keeps the frozen Exp424 MPS execution unchanged.
Its new runtime operations are descriptor-bound SQLite reads plus checkpoint and
JSON validation, so no exact MLX implementation can be substituted without a
new parity and timing qualification.  MLX is not adopted; the runtime state
records `adopted=false` and the parity/speed-gate fallback reason.  Native work
requires `torch.device("mps")`, a passing non-fallback MPS audit, and an
exclusive single-heavy-process lock.  CPU fallback is rejected.

## Verification to date

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime \
  experiments.exp425_selective_recovery.tests.test_cli \
  experiments.exp425_selective_recovery.tests.test_eligibility \
  experiments.exp425_selective_recovery.tests.test_capabilities \
  experiments.exp425_selective_recovery.tests.test_checkpoints \
  experiments.exp425_selective_recovery.tests.test_predecessor \
  experiments.exp425_selective_recovery.tests.test_database -v
PASS — 97 tests in 11.247s

python3 -m py_compile experiments/exp425_selective_recovery/runtime.py \
  scratch/run_exp425_selective_recovery.py
PASS

git diff --check
PASS
```

The requested TRAIN-only MPS parity smoke has not been run: this worktree has
no configured frozen launch assembly or verified source dataset selected for
that smoke, and running it without those arguments would violate the frozen
contract.  No Exp425 1,117-dataset population, readiness command, or TEST-label
query was started.

## Follow-up verification

```text
python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime \
  experiments.exp425_selective_recovery.tests.test_cli \
  experiments.exp417_causal_multihorizon_shadow.tests.test_runner \
  experiments.exp417_causal_multihorizon_shadow.tests.test_model \
  experiments.exp417_causal_multihorizon_shadow.tests.test_scoring \
  experiments.exp417_causal_multihorizon_shadow.tests.test_qualification -v
PASS — focused runtime/CLI plus frozen Exp417 runner/model/scoring/qualification regression.

python3 -m py_compile experiments/exp425_selective_recovery/*.py \
  scratch/run_exp425_selective_recovery.py
PASS

git diff --check
PASS
```
