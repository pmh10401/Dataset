# Task 1 Report: Exp267-273 Frozen Configuration, Dependencies, And Contracts

## Status

DONE_WITH_CONCERNS

## Implementation

Created the new `experiments.exp267_273_zero_support_fn_recovery` package with:

- Frozen policy constants: `SEED`, `SECOND_SEED`, `ALPHAS`, `PRIMARY_ALPHA`,
  `RFF_MIN_TRAIN`, `TCN_MIN_TRAIN`, `REVIEW_RISK_LIMIT`, `HARD_RISK_LIMIT`,
  and `MIN_RETROSPECTIVE_PRECISION`.
- Immutable `SourceSpec`, `SourceScore`, `FrozenSourcePrediction`,
  `RecoveryLanes`, and `PromotionAudit` dataclasses.
- A label-free `FrozenSourcePrediction` contract with exactly the requested
  freeze fields and `contains_test_labels=False` enforcement.
- Immutable existing Hard, Shadow, Priority, and Standard lane indices plus
  Recovery Shadow and Recovery Priority indices.
- `dependency_report()` using package metadata and PyTorch MPS capability
  reporting, without importing labels or database rows.

## Files

Created by this task:

- `experiments/exp267_273_zero_support_fn_recovery/__init__.py`
- `experiments/exp267_273_zero_support_fn_recovery/config.py`
- `experiments/exp267_273_zero_support_fn_recovery/contracts.py`
- `experiments/exp267_273_zero_support_fn_recovery/dependencies.py`
- `experiments/exp267_273_zero_support_fn_recovery/tests/test_contracts.py`
- `.superpowers/sdd/task-1-report-exp267.md`

No Exp253 or Exp266 artifacts were changed. Existing dirty `.superpowers`
files were not edited or staged.

## RED/GREEN Evidence

### RED

Command:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_contracts -v
```

Result before implementation: exit 1, with the expected missing-package
failure:

```text
ModuleNotFoundError: No module named 'experiments.exp267_273_zero_support_fn_recovery.config'
Ran 1 test in 0.000s
FAILED (errors=1)
```

### GREEN

Command:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_contracts -v
```

Fresh final result:

```text
Ran 5 tests in 0.000s
OK
```

The five passing tests cover the exact policy constants, forbidden evaluation
fields, label-free prediction default, frozen dataclass records, and lane
field contract.

Additional verification:

```bash
python3 -m py_compile experiments/exp267_273_zero_support_fn_recovery/__init__.py experiments/exp267_273_zero_support_fn_recovery/config.py experiments/exp267_273_zero_support_fn_recovery/contracts.py experiments/exp267_273_zero_support_fn_recovery/dependencies.py experiments/exp267_273_zero_support_fn_recovery/tests/test_contracts.py
```

Result: exit 0.

The unscoped `git diff --check` also inspected unrelated pre-existing dirty
content and reported only:

```text
.superpowers/sdd/progress.md:6: new blank line at EOF.
```

The package files themselves have no whitespace errors; the unrelated file was
left untouched.

## Dependency Status

Environment before the requested installation:

```text
Python 3.14.6
kymatio=MISSING (No package metadata was found for kymatio)
iisignature=MISSING (No package metadata was found for iisignature)
torch=2.13.0
mps_built=True
mps_available=True
```

Requested command:

```bash
python3 -m pip install --upgrade kymatio iisignature
```

Exact result: exit 1 while building `iisignature` requirements. The build
environment raised:

```text
ModuleNotFoundError: No module named 'numpy'
ERROR: Failed to build 'iisignature' when getting requirements to build wheel
```

After the failed install, both `kymatio` and `iisignature` remained missing.
Calling the implemented audit produced:

```text
PackageNotFoundError: No package metadata was found for kymatio
```

The audit contract was kept as specified; no alternate algorithm or dependency
substitution was introduced. A separate `BLOCKER_REPORT.md` was not created
because the brief restricts edits to the listed files and this report path.

## Self-Review

- The RED failure was observed before production implementation.
- The focused GREEN suite and Python compilation passed after implementation.
- `FrozenSourcePrediction` has no `y_test`, `tp`, `fp`, `fn`, `f1`, or
  `precision` fields and rejects `contains_test_labels=True`.
- The dependency module imports only `importlib.metadata` and `torch`; it does
  not read labels, SQLite state, or experiment-local modules.
- The package contains only the five brief-listed source/test files, apart from
  the required report.
- Existing dirty `.superpowers` content and Exp253/Exp266 artifacts remain
  outside the change set.

## Concerns

- Python 3.14 dependency installation failed for `iisignature`; the downstream
  scattering/signature stages cannot run until the dependency environment is
  repaired.
- `dependency_report()` therefore cannot return a complete version mapping in
  this environment, but it preserves the exact requested audit behavior.
- This task freezes contracts only. Source algorithms, population loading,
  scoring, routing, evaluation, and promotion audit logic remain for later
  tasks.
